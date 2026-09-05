"""
Hybrid variant 2b (docs/hybrid_alternative_combinations.md): keeps P1/P2
paired the way Pipeline B itself validated them, avoiding Pipeline C's
diffusion entirely, to test whether the DB3_B scoreability collapse and
DB1_B net regression (both isolated to Pipeline C's diffusion in
docs/hybrid_validation_findings.md) disappear once P2 comes from the same
pipeline as P1 instead.

    P1 (contrast)     -> Pipeline B's wavelet contrast enhancement,
                         UNCHANGED (reuses hybrid.stage1_contrast() directly)
    P2 (noise)        -> Pipeline B's OWN wavelet shrinkage denoising
                         (_wavelet_shrinkage_denoise, NOT Pipeline C's
                         diffusion -- src/pipeline_b/pipeline_b.py)
    P6 (orientation)  -> Pipeline A's oriented Gabor filtering, UNCHANGED
                         (reuses hybrid.stage3_orientation() directly)

Does NOT modify pipeline_a.py/pipeline_b.py/pipeline_c.py/hybrid.py --
imports and calls their real internal functions/stage functions directly.
Structured the same way hybrid.py already is: each stage is its own
separately-callable function, so scripts/hybrid_ablation.py's cumulative
per-stage/minus-one-component method works unchanged.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_a"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_b"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_c"))

import cv2  # noqa: E402

from config import RAW_DIR  # noqa: E402
from pipeline_b import _wavelet_shrinkage_denoise  # noqa: E402

import hybrid  # noqa: E402

stage0_preprocess = hybrid.stage0_preprocess
stage1_contrast = hybrid.stage1_contrast
stage3_orientation = hybrid.stage3_orientation


def stage2_noise_b(stage1_output, fg_mask_blocks, params=None):
    """Stage 2 (P2): Pipeline B's own _wavelet_shrinkage_denoise, on Stage
    1's output (Pipeline B's wavelet contrast). Real locked defaults
    confirmed against src/pipeline_b/pipeline_b.py's enhance() (the
    "Stage 2: wavelet soft-threshold shrinkage denoising" call) --
    identical to the function's own signature defaults, i.e. pipeline_b.py
    never overrides them either: wavelet="db4", level=3,
    threshold_scale=1.00, denoise_finest_levels=1, blend=1.0,
    noise_adaptive=True, noise_reference_sigma=5.0,
    noise_adaptive_power=4.0, minimum_scale_factor=0.10.

    Unlike hybrid.stage2_noise() (Pipeline C's diffusion), this function
    needs no alpha probe on the pre-P1 image -- Pipeline B's own denoise
    step has no such quality-adaptive scheme, it runs directly on
    stage1_output with fixed parameters.
    """
    params = params or {}
    return _wavelet_shrinkage_denoise(
        stage1_output,
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


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image

    Hybrid variant 2b: Step 0 -> Pipeline B's wavelet contrast (P1,
    unchanged) -> Pipeline B's own wavelet shrinkage denoising (P2) ->
    Pipeline A's oriented Gabor filtering (P6, unchanged).
    """
    params = params or {}
    normalized, fg_mask_blocks = stage0_preprocess(image, params)
    stage1_output = stage1_contrast(normalized, fg_mask_blocks, params)
    stage2_output = stage2_noise_b(stage1_output, fg_mask_blocks, params)
    stage3_output = stage3_orientation(stage2_output, fg_mask_blocks, params)
    return stage3_output


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("hybrid_v2b_test_output.png", out)
        print("Saved hybrid_v2b_test_output.png — open it and compare against the input image.")
