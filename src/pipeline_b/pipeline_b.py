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

Segmentation and block-wise normalisation were dropped (30 August 2026):
they targeted P5/P7, which are no longer in scope now that only P1/P2/P6
are being solved, and — since Otsu segmentation and Hong et al.
normalisation would otherwise have been called identically by all four
pipelines — keeping them would have violated the lecturer's requirement
that no technique repeat across pipelines. Exactly three primary,
independently citable techniques per pipeline (one per shared problem) is
sufficient. Adaptive thresholding (Sauvola) is no longer part of this
pipeline's plan — it is not one of the three shared problems.

Ridge orientation field estimation (structure tensor) is a supporting
calculation, not one of the three primary techniques — it is used
internally here (and by Pipelines A and C) only to steer the orientation
of the morphological structuring elements in Step 3. Find your own
citation for it and for each of the three techniques below when you write
this up; do not assume any particular citation is already settled.

CRITICAL: enhance() returns the GREYSCALE output (after denoising, before
binarisation), not a binarised output. NFIQ2 expects greyscale input — see
the caution in the analysis document, Section 6, Pipeline B.

All three steps below are TODOs: wavelet-domain contrast enhancement,
wavelet shrinkage denoising, and orientation-steered morphological
processing are the techniques you are responsible for researching and
implementing yourself.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import orientation_field  # noqa: E402
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

    # Step 1: wavelet decomposition + detail-coefficient contrast
    # enhancement (P1 — low global contrast) — TODO
    # A common Python starting point is PyWavelets (`pip install PyWavelets`):
    #   import pywt
    #   coeffs = pywt.wavedec2(image, 'db4', level=2)
    #   ... scale up the magnitude of the detail coefficients (cH, cV, cD)
    #       at one or more levels to sharpen local contrast ...
    #   contrast_enhanced = pywt.waverec2(coeffs, 'db4')
    # Ask yourself: which levels benefit most from boosting? Too aggressive
    # a boost will amplify noise before Step 2 has a chance to remove it —
    # is a mild boost here, relying on Step 2 to clean up afterwards, the
    # right order, or should denoising come first?
    contrast_enhanced = image.copy()  # placeholder — replace once this step is implemented

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
