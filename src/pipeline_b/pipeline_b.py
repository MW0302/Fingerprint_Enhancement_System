"""
Pipeline B — Member B: Noise Suppression and Ridge Continuity
(Wavelet Denoising + Adaptive Thresholding + Morphological Processing)

Targets (see Dataset_Problem_Analysis_and_Revised_Pipelines.docx, Section 6):
    P2 — high random noise (DB3)
    Ridge gaps / continuity (not independently measured, but visually common
    after denoising — see the "Note" under Section 5 of the analysis document)

    NOTE: P2b (DB3 vertical banding) is intentionally NOT a target of this
    pipeline. It is documented as a qualitative-only finding — do not try to
    "fix" it here.

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Donoho & Johnstone (1994) — wavelet shrinkage denoising
    Sauvola & Pietikäinen (2000) — adaptive thresholding
    Gonzalez & Woods (2018) — morphological processing
    Hong, Wan, & Jain (1998) — shared segmentation step

CRITICAL: enhance() returns the GREYSCALE output (after denoising), not the
binarised output. NFIQ2 expects greyscale input — see the caution in the
analysis document, Section 6, Pipeline B. The binary/morphological result is
useful as an internal structural aid (e.g., to guide gap-filling) but should
be reported and saved separately, not fed to NFIQ2 as the "enhanced image".

Steps 1 is implemented (shared). Steps 2, 3, and 4 are TODOs: wavelet
denoising, adaptive thresholding, and morphological processing are the
techniques you are responsible for researching and implementing yourself.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import segment  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: (enhanced_greyscale, binary_intermediate)
        enhanced_greyscale — pass THIS to NFIQ2
        binary_intermediate — kept for your own visual/structural reporting only
    """
    params = params or {}

    # Step 1: segmentation (shared)
    fg_mask_blocks, block_var = segment(image)

    # Step 2: wavelet decomposition + soft-threshold denoising
    # (Donoho & Johnstone, 1994) — TODO
    # A common Python starting point is PyWavelets (`pip install PyWavelets`):
    #   import pywt
    #   coeffs = pywt.wavedec2(image, 'db4', level=2)
    #   ... apply soft thresholding to the detail coefficients ...
    #   denoised = pywt.waverec2(coeffs, 'db4')
    # Ask yourself: which wavelet family and decomposition level suit DB3's
    # noise level (roughly 3-4x every other subset)? How is the threshold
    # chosen (fixed vs. adaptive per level)?
    denoised = image.copy()  # placeholder — replace once wavelet denoising is implemented

    # Step 3: local adaptive thresholding (Sauvola & Pietikäinen, 2000) — TODO
    # scikit-image has this built in: skimage.filters.threshold_sauvola
    # This should operate on `denoised`, not the raw image.
    binary_intermediate = denoised.copy()  # placeholder — replace with real binarisation

    # Step 4: morphological processing (thinning, gap-bridging, spur removal)
    # (Gonzalez & Woods, 2018) — TODO
    # cv2.morphologyEx with MORPH_CLOSE / MORPH_OPEN, or skimage.morphology.thin,
    # are reasonable starting points. Apply this to guide repairs on
    # `denoised`, not as the final output itself.

    enhanced_greyscale = denoised  # <-- this greyscale version goes to NFIQ2
    return enhanced_greyscale, binary_intermediate


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        enhanced, binary = enhance(img)
        cv2.imwrite("pipeline_b_test_output.png", enhanced)
        cv2.imwrite("pipeline_b_test_binary.png", binary)
        print("Saved pipeline_b_test_output.png (feed this to NFIQ2) and "
              "pipeline_b_test_binary.png (structural aid only).")
