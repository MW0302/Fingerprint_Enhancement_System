"""
Hybrid variant 2a (docs/hybrid_alternative_combinations.md): restores the
mechanical Section 2.4 P1 winner instead of the project lead's override, to
test whether Pipeline C's diffusion (P2) only breaks when paired with a P1
technique other than the one it was tuned alongside.

    P1 (contrast)     -> Pipeline C's own P1 stage: _homomorphic_filter +
                         its Step 1b background-feathering blend (BOTH
                         parts -- src/pipeline_c/pipeline_c.py)
    P2 (noise)        -> Pipeline C's coherence-enhancing diffusion,
                         UNCHANGED (reuses hybrid.stage2_noise() directly)
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

import numpy as np  # noqa: E402
import cv2  # noqa: E402

from config import RAW_DIR  # noqa: E402
from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _homomorphic_filter,
    _HOMOMORPHIC_GAMMA_HIGH_RANGE,
)
from common import orientation_field  # noqa: E402

import hybrid  # noqa: E402

stage0_preprocess = hybrid.stage0_preprocess
stage2_noise = hybrid.stage2_noise
stage3_orientation = hybrid.stage3_orientation


def stage1_contrast_c(normalized, fg_mask_blocks, params=None):
    """Stage 1 (P1): Pipeline C's own P1 stage -- _homomorphic_filter,
    immediately followed by its Step 1b background-feathering blend.
    Reproduces pipeline_c.py's enhance() (the block from "# Step 1:
    homomorphic filtering" through immediately before "# Step 2: ridge
    orientation field estimation") exactly, confirmed line by line:

      - Step 0c's alpha probe (pipeline_c.py, right before Step 1):
        orientation_field() on the shared, pre-P1 Step 0 `normalized`
        image, followed by _aggressiveness_alpha(normalized's coherence,
        fg_mask_blocks). This duplicates hybrid.stage2_noise()'s own
        identical alpha computation -- expected and harmless, since it is
        a deterministic function of `normalized`/`fg_mask_blocks` alone,
        not of anything this stage produces.
      - _homomorphic_filter(normalized, cutoff=0.06, gamma_low=0.5,
        gamma_high=_lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha),
        sharpness=1.0): pipeline_c.py's own real locked defaults for this
        call, reused unchanged (gamma_high is the one alpha-scaled value;
        _HOMOMORPHIC_GAMMA_HIGH_RANGE and _lerp are pipeline_c.py's own,
        imported directly, not re-derived).
      - Step 1b feathering blend: fg_mask_blocks resized to image shape
        (cv2.INTER_LINEAR), Gaussian-blurred (sigma=8.0), clipped to
        [0, 1] as fg_alpha, then contrast_enhanced = fg_alpha *
        contrast_enhanced_raw + (1 - fg_alpha) * normalized, clipped to
        [0, 255] uint8 -- pipeline_c.py's exact blend, reused unchanged.
        This is the piece the current Hybrid (Pipeline B's P1) lacks, and
        the one docs/hybrid_validation_findings.md identified as the
        likely reason Pipeline C's diffusion (P2) breaks when paired with
        a different P1 technique.

    Returns the blended contrast_enhanced image (uint8), the same return
    shape as hybrid.stage1_contrast()'s _wavelet_contrast output, so it is
    a drop-in P1 replacement for hybrid.stage2_noise()'s stage1_output
    argument.
    """
    params = params or {}
    _theta_probe, coherence_probe = orientation_field(normalized)
    alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)

    contrast_enhanced_raw = _homomorphic_filter(
        normalized,
        cutoff=params.get("homomorphic_cutoff", 0.06),
        gamma_low=params.get("homomorphic_gamma_low", 0.5),
        gamma_high=params.get(
            "homomorphic_gamma_high", _lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha)
        ),
        sharpness=params.get("homomorphic_sharpness", 1.0),
    )

    h, w = normalized.shape
    fg_alpha = cv2.resize(
        fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
    )
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), 8.0)
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
    contrast_enhanced = (
        fg_alpha * contrast_enhanced_raw.astype(np.float64)
        + (1 - fg_alpha) * normalized.astype(np.float64)
    )
    contrast_enhanced = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)
    return contrast_enhanced


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image

    Hybrid variant 2a: Step 0 -> Pipeline C's own homomorphic + feathered P1
    -> Pipeline C's coherence diffusion (P2, unchanged) -> Pipeline A's
    oriented Gabor filtering (P6, unchanged).
    """
    params = params or {}
    normalized, fg_mask_blocks = stage0_preprocess(image, params)
    stage1_output = stage1_contrast_c(normalized, fg_mask_blocks, params)
    stage2_output, _alpha = stage2_noise(normalized, stage1_output, fg_mask_blocks, params)
    stage3_output = stage3_orientation(stage2_output, fg_mask_blocks, params)
    return stage3_output


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("hybrid_v2a_test_output.png", out)
        print("Saved hybrid_v2a_test_output.png — open it and compare against the input image.")
