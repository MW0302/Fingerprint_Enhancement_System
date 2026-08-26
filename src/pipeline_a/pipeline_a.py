"""
Pipeline A — Member A: Contrast and Orientation Enhancement (CLAHE + Gabor)

Targets (see Dataset_Problem_Analysis_and_Revised_Pipelines.docx, Section 6):
    P1 — low global contrast (DB3)
    P3 — uneven illumination (DB3, DB2)
    P6 — weak/inconsistent ridge orientation (DB4, DB3)
    P7 — within-subset heterogeneity (DB1)

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Zuiderveld (1994) — CLAHE
    Hong, Wan, & Jain (1998) — orientation field + oriented Gabor filtering,
                                and the shared segmentation/normalisation steps

Every function below matches one step in the pipeline's step sequence.
Steps 1, 2, and 4 are already implemented (shared across pipelines).
Step 3 (CLAHE) is implemented as a starting point — tune it and be ready to
explain your parameter choices. Step 5 (oriented Gabor filtering) is left as
a TODO: this is the technique you are responsible for researching and
implementing yourself.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import segment, normalize_image, orientation_field  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array (e.g. from cv2.imread(path, cv2.IMREAD_GRAYSCALE))
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 1: segmentation (shared) — not applied to the pixels yet, just
    # computed here so you can use fg_mask_blocks later if your Gabor
    # implementation should skip background blocks.
    fg_mask_blocks, block_var = segment(image)

    # Step 2: block-wise normalisation (shared)
    normalized = normalize_image(image)

    # Step 3: CLAHE (Zuiderveld, 1994)
    clip_limit = params.get("clahe_clip", 2.0)
    grid_size = params.get("clahe_grid", 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    contrast_enhanced = clahe.apply(normalized)

    # Step 4: ridge orientation field estimation (shared; Hong et al., 1998)
    theta_field, coherence_field = orientation_field(contrast_enhanced)

    # Step 5: oriented Gabor filtering (Hong et al., 1998) — TODO
    # Idea: for each block, build a Gabor kernel oriented along theta_field
    # at that block (cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma)),
    # convolve that block (or a local neighbourhood) with its own kernel, and
    # stitch the results back together. Ask yourself:
    #   - what wavelength (lambd) matches typical ridge spacing in this dataset?
    #   - how do you handle block edges without visible seams?
    #   - should low-coherence blocks (noisy/ambiguous orientation) be
    #     filtered less aggressively, or skipped?
    enhanced = contrast_enhanced  # placeholder — replace once Gabor step is implemented

    return enhanced


if __name__ == "__main__":
    # Quick manual test on a single image before running the full 320-image batch.
    # Uses your own local data/raw/ copy (see README) — no path editing needed.
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_a_test_output.png", out)
        print("Saved pipeline_a_test_output.png — open it and compare against the input image.")
