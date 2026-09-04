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
a separately callable function for cumulative ablation. Stage 2 wavelet
shrinkage denoising is also independently callable. Stage 3 remains a TODO:
orientation-steered morphological processing is not implemented in this
revision.
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


_IMMERKAER_NOISE_KERNEL = np.array(
    [[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64
)


def _estimate_random_noise_sigma(image):
    """Estimate additive noise using Immerkær's fast 3x3 operator."""
    source = np.asarray(image, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("_estimate_random_noise_sigma expects a 2D grayscale image")
    height, width = source.shape
    if height < 3 or width < 3:
        return 0.0
    response = cv2.filter2D(source, -1, _IMMERKAER_NOISE_KERNEL)
    return float(
        np.sqrt(np.pi / 2.0)
        * np.abs(response).sum()
        / (6.0 * (width - 2) * (height - 2))
    )


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


def _wavelet_shrinkage_denoise(
    image,
    fg_mask_blocks=None,
    wavelet="db4",
    level=3,
    threshold_scale=1.00,
    denoise_finest_levels=1,
    blend=1.0,
    noise_adaptive=True,
    noise_reference_sigma=5.0,
    noise_adaptive_power=4.0,
    minimum_scale_factor=0.10,
):
    """Stage 2: subband-adaptive BayesShrink with soft thresholding.

    Noise sigma is robustly estimated from the finest diagonal-detail band
    using its median absolute deviation. For every selected detail subband,
    the BayesShrink threshold ``sigma_noise**2 / sigma_signal`` is computed
    independently and scaled by ``threshold_scale`` before soft shrinkage.
    Only the requested number of finest levels are thresholded by default so
    coarser fingerprint ridge structure is preserved. When ``noise_adaptive``
    is enabled, Immerkær's structure-robust image-noise estimate scales the
    threshold strength continuously. The configured minimum factor keeps the
    counted P2 technique active even on cleaner images.

    The reconstructed denoising increment is blended only into the shared
    foreground mask when supplied. Output is always a grayscale uint8 array
    with the same shape as ``image``.
    """
    source = np.clip(image, 0, 255).astype(np.float32)
    if source.ndim != 2:
        raise ValueError("_wavelet_shrinkage_denoise expects a 2D grayscale image")

    wavelet_obj = pywt.Wavelet(str(wavelet))
    requested_level = max(1, int(level))
    max_level = pywt.dwtn_max_level(source.shape, wavelet_obj)
    actual_level = min(requested_level, max_level)
    if actual_level < 1:
        return source.astype(np.uint8)

    threshold_scale = max(0.0, float(threshold_scale))
    blend = float(np.clip(blend, 0.0, 1.0))
    finest_levels = int(np.clip(denoise_finest_levels, 0, actual_level))
    if threshold_scale == 0.0 or blend == 0.0 or finest_levels == 0:
        return source.astype(np.uint8)

    if noise_adaptive:
        reference = max(float(noise_reference_sigma), 1e-12)
        power = max(float(noise_adaptive_power), 0.0)
        minimum_factor = float(np.clip(minimum_scale_factor, 0.0, 1.0))
        measured_noise = _estimate_random_noise_sigma(source)
        scale_factor = np.clip((measured_noise / reference) ** power, minimum_factor, 1.0)
        threshold_scale *= float(scale_factor)

    coeffs = pywt.wavedec2(
        source,
        wavelet_obj,
        mode="symmetric",
        level=actual_level,
    )

    finest_diagonal = coeffs[-1][2]
    if fg_mask_blocks is None:
        noise_samples = finest_diagonal.ravel()
    else:
        coefficient_mask = cv2.resize(
            np.asarray(fg_mask_blocks, dtype=np.uint8),
            (finest_diagonal.shape[1], finest_diagonal.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        noise_samples = finest_diagonal[coefficient_mask]
        if noise_samples.size == 0:
            noise_samples = finest_diagonal.ravel()
    sigma_noise = float(np.median(np.abs(noise_samples))) / 0.6745
    if sigma_noise <= 1e-12:
        return source.astype(np.uint8)
    noise_variance = sigma_noise * sigma_noise

    mapped_coeffs = [coeffs[0]]
    first_denoised_index = actual_level - finest_levels + 1
    for detail_index, details in enumerate(coeffs[1:], start=1):
        if detail_index < first_denoised_index:
            mapped_coeffs.append(details)
            continue

        mapped_details = []
        for band in details:
            observed_variance = float(np.mean(np.square(band, dtype=np.float64)))
            signal_sigma = np.sqrt(max(observed_variance - noise_variance, 0.0))
            if signal_sigma <= 1e-12:
                threshold = float(np.max(np.abs(band)))
            else:
                threshold = noise_variance / signal_sigma
            mapped_details.append(
                pywt.threshold(
                    band,
                    value=threshold_scale * threshold,
                    mode="soft",
                )
            )
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

    denoised = source + foreground * blend * increment
    return np.clip(np.rint(denoised), 0, 255).astype(np.uint8)


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

    # Stage 2: wavelet soft-threshold shrinkage denoising (P2). Kept separate
    # so Stage 2 minus Stage 1 measures this technique's contribution.
    denoised = _wavelet_shrinkage_denoise(
        contrast_enhanced,
        fg_mask_blocks=fg_mask_blocks,
        wavelet=params.get("denoise_wavelet", "db4"),
        level=params.get("denoise_wavelet_level", 3),
        threshold_scale=params.get("denoise_threshold_scale", 1.00),
        denoise_finest_levels=params.get("denoise_finest_levels", 1),
        blend=params.get("denoise_blend", 1.0),
        noise_adaptive=params.get("denoise_noise_adaptive", True),
        noise_reference_sigma=params.get("denoise_noise_reference_sigma", 5.0),
        noise_adaptive_power=params.get("denoise_noise_adaptive_power", 4.0),
        minimum_scale_factor=params.get("denoise_minimum_scale_factor", 0.10),
    )

    # Step 3: ridge orientation field estimation (supporting calculation,
    # not one of the three primary techniques) — used to steer Step 4.
    # theta_field, coherence_field = orientation_field(denoised)

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
