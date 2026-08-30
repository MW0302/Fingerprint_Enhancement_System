"""
Pipeline D — Member D: Frequency-Domain Contrast, Denoising, and
Orientation-Frequency Enhancement (FFT High-Frequency Emphasis Filtering +
Frequency-Domain Wiener/Notch Filtering + STFT-Based Joint
Orientation-Frequency Ridge Reconstruction)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)                -> FFT high-frequency
                                                       emphasis filtering
    P2 — high random noise (DB3)                   -> Frequency-domain
                                                       Wiener / notch filtering
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                                -> STFT-based joint
                                                       orientation-frequency
                                                       ridge reconstruction

Segmentation and block-wise normalisation were dropped (30 August 2026):
they targeted P5/P7, which are no longer in scope now that only P1/P2/P6
are being solved, and — since Otsu segmentation and Hong et al.
normalisation would otherwise have been called identically by all four
pipelines — keeping them would have violated the lecturer's requirement
that no technique repeat across pipelines. Exactly three primary,
independently citable techniques per pipeline (one per shared problem) is
sufficient, so the earlier "light morphological clean-up" step has also
been dropped — it was never one of the three shared problems.

All three steps below are TODOs: FFT high-frequency emphasis filtering,
frequency-domain Wiener/notch filtering, and STFT-based ridge
reconstruction are the techniques you are responsible for researching,
implementing, and being able to explain yourself. Find your own citation
for each; do not assume any particular citation is already settled.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
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

    # Step 1: FFT high-frequency emphasis filtering (P1 — low global
    # contrast) — TODO
    # Core idea: take the 2D FFT of the image (np.fft.fft2 / fftshift),
    # apply a high-frequency emphasis transfer function (boosts high
    # spatial frequencies relative to low ones, which sharpens local
    # contrast without a full log-domain homomorphic decomposition), then
    # inverse-transform back to the spatial domain. Ask yourself:
    #   - what cutoff separates "illumination-scale" from "ridge-scale"
    #     frequency content in this dataset?
    #   - how do you rescale the output back to a valid 0-255 range without
    #     a few outlier pixels dominating the stretch? (Consider a robust
    #     percentile-based rescale rather than true min/max.)
    contrast_enhanced = image.copy()  # placeholder — replace once this step is implemented

    # Step 2: frequency-domain Wiener / notch filtering (P2 — high random
    # noise) — TODO
    # Core idea: DB3's noise is not perfectly white — the qualitative
    # vertical-banding finding (P2b, see the analysis document Section 3)
    # suggests some structured/periodic component. A notch filter can
    # suppress specific frequency-domain peaks corresponding to that
    # banding, while a Wiener filter provides a more general statistical
    # denoising approach in the frequency domain. Ask yourself:
    #   - can you identify the banding frequency peak(s) directly from the
    #     2D FFT magnitude spectrum of a noisy DB3 image?
    #   - how do you estimate the noise/signal power ratio a Wiener filter
    #     needs, given you don't have a clean reference image?
    denoised = contrast_enhanced.copy()  # placeholder — replace once this step is implemented

    # Step 3: STFT-based joint orientation-frequency ridge reconstruction
    # (P6 — weak/inconsistent ridge orientation) — TODO
    # Core idea: slide a window over the image, take a 2D FFT of each window
    # (np.fft.fft2), and from its spectrum jointly estimate:
    #   - local ridge orientation (angle of the dominant frequency peak)
    #   - local ridge frequency (distance of the peak from the origin)
    #   - local energy (peak magnitude)
    # then reconstruct/enhance each window using this information and stitch
    # the windows back together (with overlap to avoid seams). Ask yourself:
    #   - what window size balances "local enough" against "enough ridges
    #     inside the window to get a clean frequency peak"?
    #   - how much should windows overlap, and how do you blend the overlap
    #     region to avoid visible block seams?
    enhanced = denoised  # placeholder — replace once this step is implemented

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB2_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_d_test_output.png", out)
        print("Saved pipeline_d_test_output.png — open it and compare against the input image.")
