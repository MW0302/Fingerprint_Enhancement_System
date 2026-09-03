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
preprocessing. All three numbered steps below are still TODOs:
wavelet-domain contrast enhancement, wavelet shrinkage denoising, and
orientation-steered morphological processing are the techniques you are
responsible for researching and implementing yourself.
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

    # Step 1: wavelet decomposition + detail-coefficient contrast
    # enhancement (P1 — low global contrast) — TODO
    # A common Python starting point is PyWavelets (`pip install PyWavelets`):
    #   import pywt
    #   coeffs = pywt.wavedec2(normalized, 'db4', level=2)
    #   ... scale up the magnitude of the detail coefficients (cH, cV, cD)
    #       at one or more levels to sharpen local contrast ...
    #   contrast_enhanced = pywt.waverec2(coeffs, 'db4')
    # Ask yourself: which levels benefit most from boosting? Too aggressive
    # a boost will amplify noise before Step 2 has a chance to remove it —
    # is a mild boost here, relying on Step 2 to clean up afterwards, the
    # right order, or should denoising come first?
    contrast_enhanced = normalized.copy()  # placeholder — replace once this step is implemented

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
