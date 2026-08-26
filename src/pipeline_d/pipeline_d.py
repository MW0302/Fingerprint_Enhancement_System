"""
Pipeline D — Member D: Frequency-Domain Enhancement (STFT)

Targets (see Dataset_Problem_Analysis_and_Revised_Pipelines.docx, Section 6):
    P4 — blurred/soft ridges (DB2)
    P6 — weak/inconsistent ridge orientation (DB4)
    Local ridge-frequency variation

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Chikkerur, Cartwright, & Govindaraju (2007) — STFT-based ridge enhancement
    Gonzalez & Woods (2018) — morphological clean-up
    Hong, Wan, & Jain (1998) — shared segmentation/normalisation steps

Steps 1 and 2 are implemented (shared). Step 3 (STFT-based ridge enhancement)
is the technique you are responsible for researching and implementing
yourself — it is the core of this pipeline. Step 4 (morphological clean-up)
is a light supporting step.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import segment, normalize_image  # noqa: E402
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

    # Step 1: segmentation (shared)
    fg_mask_blocks, block_var = segment(image)

    # Step 2: block-wise normalisation (shared) — a consistent intensity
    # range improves the reliability of the frequency estimation in step 3.
    normalized = normalize_image(image)

    # Step 3: STFT-based ridge enhancement (Chikkerur, Cartwright, &
    # Govindaraju, 2007) — TODO
    # Core idea: slide a window over the image, take a 2D FFT of each window
    # (np.fft.fft2), and from its spectrum jointly estimate:
    #   - local ridge orientation (angle of the dominant frequency peak)
    #   - local ridge frequency (distance of the peak from the origin)
    #   - local energy (peak magnitude)
    # then reconstruct/enhance each window using this information and stitch
    # the windows back together (with overlap to avoid seams). Ask yourself:
    #   - what window size balances "local enough" against "enough ridges
    #     inside the window to get a clean frequency peak"?
    #   - how much should windows overlap?
    enhanced = normalized.copy()  # placeholder — replace once STFT step is implemented

    # Step 4: morphological clean-up (light opening/closing) (Gonzalez &
    # Woods, 2018) — removes small artefacts introduced by frequency-domain
    # reconstruction. Small kernel only; this is a clean-up step, not the
    # main enhancement.
    # TODO: e.g. cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel) then
    # cv2.MORPH_CLOSE with a small (e.g. 3x3) kernel.

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
