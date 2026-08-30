"""
Pipeline A — Member A: Spatial-Domain Contrast, Noise, and Orientation Enhancement
(CLAHE + Median/Bilateral Filtering + Oriented Gabor Filtering)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)                -> CLAHE
    P2 — high random noise (DB3)                   -> Median / bilateral filtering
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                                -> Oriented Gabor filtering

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
techniques, not differing preprocessing.

Ridge orientation field estimation (structure tensor) is a supporting
calculation, not one of the three primary techniques — it is used
internally here (and by Pipelines B and C) only to steer the oriented
Gabor filtering step. Find your own citation for it and for the Gabor
filtering technique itself when you write this up; do not reuse Pipeline
C's citation list without checking it applies to your own implementation.

Steps 0a-0b (normalisation, segmentation) and 1 (CLAHE) are implemented as
a starting point — tune CLAHE's parameters and be ready to explain your
choices. Steps 2 (median/bilateral filtering) and 3 (oriented Gabor
filtering) are TODOs: these are the techniques you are responsible for
researching, implementing, and being able to explain yourself.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import orientation_field, normalize_image, segment  # noqa: E402
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

    # Step 0a-0b: block-wise normalisation + Otsu segmentation (Hong, Wan, &
    # Jain, 1998) — shared preprocessing, not one of this pipeline's three
    # primary techniques (see module docstring). fg_mask_blocks is available
    # if your own Step 2/3 implementation wants to skip or de-emphasise
    # background blocks; it's not required.
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", 100.0),
        target_var=params.get("normalize_target_var", 1600.0),
    )
    fg_mask_blocks, _block_var = segment(normalized)

    # Step 1: CLAHE (P1 — low global contrast)
    clip_limit = params.get("clahe_clip", 2.0)
    grid_size = params.get("clahe_grid", 8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    contrast_enhanced = clahe.apply(normalized)

    # Step 2: median / bilateral filtering (P2 — high random noise) — TODO
    # DB3's noise level is roughly 3-4x every other subset (see the analysis
    # document, Section 3). Ask yourself:
    #   - median filtering is simple and edge-preserving for salt-and-pepper
    #     style noise, but can blur fine ridge detail if the kernel is too
    #     large — what kernel size keeps ridges intact?
    #   - bilateral filtering preserves edges better (it weights neighbours
    #     by both spatial distance and intensity similarity) but is slower
    #     and has two parameters (sigma_color, sigma_space) to tune — is the
    #     extra cost worth it here?
    #   - should this run before or after CLAHE? Running CLAHE first can
    #     amplify noise it just contrast-stretched, which may argue for
    #     denoising first instead — try both and compare.
    denoised = contrast_enhanced.copy()  # placeholder — replace once this step is implemented

    # Step 3: ridge orientation field estimation (supporting calculation,
    # not one of the three primary techniques) — used to steer Step 4.
    theta_field, coherence_field = orientation_field(denoised)

    # Step 4: oriented Gabor filtering (P6 — weak/inconsistent ridge
    # orientation) — TODO
    # Idea: for each block, build a Gabor kernel oriented along theta_field
    # at that block (cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma)),
    # convolve that block (or a local neighbourhood) with its own kernel, and
    # stitch the results back together. Ask yourself:
    #   - what wavelength (lambd) matches typical ridge spacing in this dataset?
    #   - how do you handle block edges without visible seams?
    #   - should low-coherence blocks (noisy/ambiguous orientation) be
    #     filtered less aggressively, or skipped?
    enhanced = denoised  # placeholder — replace once Gabor step is implemented

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
