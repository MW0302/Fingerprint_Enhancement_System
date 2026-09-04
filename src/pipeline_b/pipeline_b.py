"""
Pipeline B — Member B: Wavelet-Domain Contrast, Denoising, and Orientation-Steered
Morphology (Wavelet Contrast Enhancement + Wavelet Shrinkage Denoising +
Orientation-Steered Morphological Processing)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)                -> Wavelet-domain contrast
                                                       enhancement
    P2 — high random noise (DB3)                   -> Wavelet shrinkage denoising
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                                -> Orientation-steered
                                                       morphological processing

Segmentation and block-wise normalisation, revision history: dropped on 30
August 2026 because — with only P1/P2/P6 in scope — having all four
pipelines call the identical Otsu/Hong et al. steps as a counted technique
would have violated the lecturer's no-repeated-technique requirement. Added
back the same day (later revision) as a shared PREPROCESSING stage (Steps
0a-0b below) used by all four pipelines uniformly: this does not reopen the
repeated-technique issue because preprocessing that conditions the image
without independently solving P1/P2/P6 itself is treated the same way
orientation_field() already is (see below) — a supporting step, not one of
the three counted techniques. It was made uniform across all four pipelines,
rather than added to just one, to satisfy the group's own "Fair Experimental
Conditions" principle (Handover Notes) — every pipeline should start from
the same input so differences in the results reflect the three counted
techniques, not differing preprocessing. Adaptive thresholding (Sauvola) is
still not part of this pipeline's plan — it is not one of the three shared
problems.

Ridge orientation field estimation (structure tensor) is a supporting
calculation, not one of the three primary techniques — it is used
internally here (and by Pipelines A and C) only to steer the orientation
of the morphological structuring elements in Step 3. Find your own
citation for it and for each of the three techniques below when you write
this up; do not assume any particular citation is already settled.

CRITICAL: enhance() returns the GREYSCALE output (after denoising, before
binarisation), not a binarised output. NFIQ2 expects greyscale input — see
the caution in the analysis document, Section 6, Pipeline B.

Steps 0a-0b (normalisation, segmentation) are implemented as shared
preprocessing. Stage 1 wavelet-domain contrast enhancement is implemented as
a separately callable function for cumulative ablation. Stages 2 and 3 remain
TODO placeholders: wavelet shrinkage denoising and orientation-steered
morphological processing are not implemented in this revision.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import (  # noqa: E402
    orientation_field, normalize_image, segment,
    DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR,
)
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np
import pywt


def _wavelet_contrast(
    image,
    fg_mask_blocks=None,
    wavelet="db4",
    level=3,
    coarse_gain=1.60,
    fine_gain=1.00,
    coefficient_floor_percentile=25.0,
    blend=1.0,
):
    """Stage 1: enhance ridge detail in a multilevel wavelet representation.

    The approximation coefficients are retained unchanged so the operation
    does not become a second global intensity-normalisation step. Detail
    coefficients are amplified with a magnitude-adaptive mapping: strong
    structural coefficients approach the requested gain, while very small
    coefficients (which are more likely to be noise) remain close to their
    original magnitude. Coarser and finer detail levels have separate gains
    because fine-scale amplification is the most likely to boost noise before
    Pipeline B's later shrinkage-denoising stage.

    If the shared segmentation mask is provided, only foreground pixels
    receive the reconstructed detail increment. The function always returns
    a grayscale uint8 array with the same shape as ``image``.
    """
    source = np.clip(image, 0, 255).astype(np.float32)
    if source.ndim != 2:
        raise ValueError("_wavelet_contrast expects a 2D grayscale image")

    wavelet_obj = pywt.Wavelet(str(wavelet))
    requested_level = max(1, int(level))
    max_level = pywt.dwtn_max_level(source.shape, wavelet_obj)
    actual_level = min(requested_level, max_level)
    if actual_level < 1:
        return source.astype(np.uint8)

    coarse_gain = max(1.0, float(coarse_gain))
    fine_gain = max(1.0, float(fine_gain))
    floor_percentile = float(np.clip(coefficient_floor_percentile, 0.0, 100.0))
    blend = float(np.clip(blend, 0.0, 1.0))

    coeffs = pywt.wavedec2(
        source,
        wavelet_obj,
        mode="symmetric",
        level=actual_level,
    )
    mapped_coeffs = [coeffs[0]]

    # wavedec2 returns detail tuples from coarsest to finest. Interpolate the
    # requested gains in that same order so level changes remain predictable.
    level_gains = np.linspace(coarse_gain, fine_gain, actual_level)
    for details, gain in zip(coeffs[1:], level_gains):
        mapped_details = []
        for band in details:
            magnitude = np.abs(band)
            nonzero = magnitude[magnitude > 0]
            floor = (
                float(np.percentile(nonzero, floor_percentile))
                if nonzero.size
                else 0.0
            )
            if floor <= 1e-12:
                reliability = (magnitude > 0).astype(np.float32)
            else:
                reliability = magnitude / (magnitude + floor)
            scale = 1.0 + (float(gain) - 1.0) * reliability
            mapped_details.append(band * scale)
        mapped_coeffs.append(tuple(mapped_details))

    reconstructed = pywt.waverec2(
        mapped_coeffs,
        wavelet_obj,
        mode="symmetric",
    )[: source.shape[0], : source.shape[1]]
    increment = reconstructed.astype(np.float32) - source

    if fg_mask_blocks is None:
        foreground = np.ones_like(source, dtype=np.float32)
    else:
        foreground = cv2.resize(
            np.asarray(fg_mask_blocks, dtype=np.float32),
            (source.shape[1], source.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        foreground = np.clip(foreground, 0.0, 1.0)

    enhanced = source + foreground * blend * increment
    return np.clip(np.rint(enhanced), 0, 255).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 0a-0b: block-wise normalisation + Otsu segmentation (Hong, Wan, &
    # Jain, 1998) — shared preprocessing, not one of this pipeline's three
    # primary techniques (see module docstring). fg_mask_blocks is available
    # if your own Step 3/4 implementation wants to skip or de-emphasise
    # background blocks; it's not required.
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    fg_mask_blocks, _block_var = segment(normalized)

    # Stage 1: wavelet detail-coefficient contrast enhancement (P1). Kept as
    # a separately callable function for cumulative stage-wise ablation.
    contrast_enhanced = _wavelet_contrast(
        normalized,
        fg_mask_blocks=fg_mask_blocks,
        wavelet=params.get("wavelet", "db4"),
        level=params.get("wavelet_level", 3),
        coarse_gain=params.get("wavelet_coarse_gain", 1.60),
        fine_gain=params.get("wavelet_fine_gain", 1.00),
        coefficient_floor_percentile=params.get(
            "wavelet_coefficient_floor_percentile", 25.0
        ),
        blend=params.get("wavelet_contrast_blend", 1.0),
    )

    # Step 2: wavelet decomposition + soft-threshold shrinkage denoising
    # (P2 — high random noise) — TODO
    #   coeffs = pywt.wavedec2(contrast_enhanced, 'db4', level=2)
    #   ... apply soft thresholding to the detail coefficients ...
    #   denoised = pywt.waverec2(coeffs, 'db4')
    # Ask yourself: which wavelet family and decomposition level suit DB3's
    # noise level (roughly 3-4x every other subset)? How is the threshold
    # chosen (fixed vs. adaptive per level, e.g. universal/VisuShrink)?
    denoised = contrast_enhanced.copy()  # placeholder — replace once this step is implemented

    # Step 3: ridge orientation field estimation (supporting calculation,
    # not one of the three primary techniques) — used to steer Step 4.
    theta_field, coherence_field = orientation_field(denoised)

    # Step 4: orientation-steered morphological processing (P6 —
    # weak/inconsistent ridge orientation) — TODO
    # Idea: build directional structuring elements (e.g. short line-shaped
    # kernels) oriented along theta_field at each block, and use them for
    # local morphological closing/opening to bridge ridge gaps and remove
    # spurs along the correct local direction rather than a single fixed
    # global direction. cv2.morphologyEx and skimage.morphology are
    # reasonable starting points for the per-block operations themselves.
    # Ask yourself:
    #   - how do you build/rotate a structuring element per block?
    #   - how do you stitch per-block results back together without seams?
    #   - should low-coherence blocks (ambiguous orientation) be processed
    #     less aggressively, or skipped?
    enhanced = denoised  # placeholder — replace once this step is implemented

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_b_test_output.png", out)
        print("Saved pipeline_b_test_output.png (feed this to NFIQ2) — open it and "
              "compare against the input image.")
