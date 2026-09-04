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
from common import (  # noqa: E402
    orientation_field, normalize_image, segment,
    DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR,
)
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def _clahe_contrast(image, clip_limit=2.0, grid_size=8):
    """Stage 1: improve local contrast with CLAHE."""
    grid_size = max(1, int(grid_size))
    clahe = cv2.createCLAHE(
        clipLimit=max(float(clip_limit), 0.0),
        tileGridSize=(grid_size, grid_size),
    )
    return clahe.apply(np.clip(image, 0, 255).astype(np.uint8))


def _bilateral_denoise(image, diameter=5, sigma_color=35.0, sigma_space=5.0):
    """Stage 2: suppress noise while retaining ridge/valley edges."""
    diameter = max(1, int(diameter))
    if diameter % 2 == 0:
        diameter += 1
    return cv2.bilateralFilter(
        np.clip(image, 0, 255).astype(np.uint8),
        diameter,
        max(float(sigma_color), 0.0),
        max(float(sigma_space), 0.0),
    )


def _oriented_gabor_filter(
    image,
    theta_field,
    coherence_field,
    fg_mask_blocks,
    kernel_size=17,
    sigma=4.0,
    wavelength=8.0,
    gamma=0.5,
    strength=0.7,
    orientation_bins=16,
    coherence_floor=0.2,
):
    """Stage 3: add ridge-aligned Gabor responses without block seams.

    A small bank of whole-image responses is computed and selected using an
    upsampled orientation field.  This avoids stitching independently
    filtered blocks, while coherence and segmentation prevent uncertain or
    background regions from receiving a strong artificial ridge pattern.
    """
    source = np.clip(image, 0, 255).astype(np.uint8)
    h, w = source.shape
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    orientation_bins = max(1, int(orientation_bins))

    # Ridge directions are pi-periodic. OpenCV's theta describes the Gabor
    # carrier normal, hence the additional pi/2 rotation from ridge angle.
    bin_angles = np.arange(orientation_bins, dtype=np.float32) * np.pi / orientation_bins
    responses = []
    for ridge_angle in bin_angles:
        kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            max(float(sigma), 1e-6),
            float(ridge_angle + np.pi / 2),
            max(float(wavelength), 1e-6),
            max(float(gamma), 1e-6),
            psi=0,
            ktype=cv2.CV_32F,
        )
        kernel -= kernel.mean()
        norm = np.sum(np.abs(kernel))
        if norm > 0:
            kernel /= norm
        responses.append(cv2.filter2D(source, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT))
    responses = np.stack(responses)

    # Interpolate the pi-periodic ridge direction through its double-angle
    # vector representation. Directly resizing theta would average angles on
    # opposite sides of the -pi/2 / pi/2 wrap boundary incorrectly.
    cos2_theta = cv2.resize(
        np.cos(2.0 * theta_field).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    sin2_theta = cv2.resize(
        np.sin(2.0 * theta_field).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    theta = 0.5 * np.arctan2(sin2_theta, cos2_theta)
    bin_index = np.rint(np.mod(theta, np.pi) * orientation_bins / np.pi).astype(np.int32)
    bin_index %= orientation_bins
    selected = np.take_along_axis(responses, bin_index[None, ...], axis=0)[0]

    coherence = cv2.resize(
        np.clip(coherence_field, 0.0, 1.0).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    foreground = cv2.resize(
        fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    confidence = np.clip(
        (coherence - float(coherence_floor)) / max(1.0 - float(coherence_floor), 1e-6),
        0.0,
        1.0,
    )
    gain = np.clip(foreground, 0.0, 1.0) * confidence * float(strength)
    enhanced = source.astype(np.float32) + gain * selected
    return np.clip(enhanced, 0, 255).astype(np.uint8)


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
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    fg_mask_blocks, _block_var = segment(normalized)

    # Stage 1: CLAHE (P1 — low global contrast). This named cumulative
    # output is intentionally retained for later step-wise ablation.
    contrast_enhanced = _clahe_contrast(
        normalized,
        clip_limit=params.get("clahe_clip", 2.0),
        grid_size=params.get("clahe_grid", 8),
    )

    # Stage 2: bilateral denoising (P2 — high random noise), applied to the
    # Stage 1 result so this output represents techniques 1+2 cumulatively.
    denoised = _bilateral_denoise(
        contrast_enhanced,
        diameter=params.get("bilateral_diameter", 5),
        sigma_color=params.get("bilateral_sigma_color", 35.0),
        sigma_space=params.get("bilateral_sigma_space", 5.0),
    )

    # Step 3: ridge orientation field estimation (supporting calculation,
    # not one of the three primary techniques) — used to steer Step 4.
    theta_field, coherence_field = orientation_field(denoised)

    # Stage 3: oriented Gabor filtering (P6). This is the cumulative final
    # stage (techniques 1+2+3) and remains enhance()'s public result.
    enhanced = _oriented_gabor_filter(
        denoised,
        theta_field,
        coherence_field,
        fg_mask_blocks,
        kernel_size=params.get("gabor_kernel_size", 17),
        sigma=params.get("gabor_sigma", 4.0),
        wavelength=params.get("gabor_wavelength", 8.0),
        gamma=params.get("gabor_gamma", 0.5),
        strength=params.get("gabor_strength", 0.7),
        orientation_bins=params.get("gabor_orientation_bins", 16),
        coherence_floor=params.get("gabor_coherence_floor", 0.2),
    )

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
