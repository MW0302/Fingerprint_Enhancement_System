"""
Hybrid pipeline (Handover Notes Priority 5 / Section 17): FIXED, not
adaptive -- one single pipeline (Step0 -> chosen P1 -> chosen P2 -> chosen
P6) applied identically to every image. Assembled from the three per-slot
winners selected in Section 2.4 steps 1-3 (docs/section_2_4_findings.md)
and adjusted by the project lead's paired-statistical review (paired
t-test/Wilcoxon on Pipeline B vs C's delta_P1 -- C's higher mean was not
significant, while B is clearly more reliable on every other metric):

    P1 (contrast)     -> Pipeline B's wavelet contrast enhancement
                         (_wavelet_contrast, src/pipeline_b/pipeline_b.py)
    P2 (noise)        -> Pipeline C's coherence-enhancing diffusion
                         (_coherence_diffusion, src/pipeline_c/pipeline_c.py)
    P6 (orientation)  -> Pipeline A's oriented Gabor filtering
                         (_oriented_gabor_filter, src/pipeline_a/pipeline_a.py)

Does NOT modify pipeline_a.py/pipeline_b.py/pipeline_c.py -- imports and
calls their real internal functions directly, with their real locked
default parameter values (read from each file's own enhance(), confirmed
line by line, not re-derived or guessed -- see each stage function's own
docstring below for exactly what was confirmed and against which source
lines).

Step 0 (shared preprocessing, identical to every pipeline): normalize_image()
+ segment() from common.py.

Structured the same way A/B/C/D already are: each of the three stages is
its own separately-callable function, so cumulative per-stage NFIQ2 can be
measured the same way every other pipeline's ablation data already is (see
scripts/hybrid_ablation.py).
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_a"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_b"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_c"))

from common import (  # noqa: E402
    normalize_image, segment, orientation_field,
    DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR,
)
from config import RAW_DIR  # noqa: E402

from pipeline_b import _wavelet_contrast  # noqa: E402
from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _coherence_diffusion,
    _DIFFUSION_ITERATIONS_RANGE,
    _DIFFUSION_KAPPA_RANGE,
)
from pipeline_a import _oriented_gabor_filter  # noqa: E402

import cv2


def stage0_preprocess(image, params=None):
    """Step 0: shared normalize_image() + segment(), identical to every
    pipeline's own Step 0a-0b (confirmed against pipeline_c.py's enhance(),
    lines 803-816 -- the same two calls, same default constants, appear
    verbatim in every one of A/B/C/D's own enhance())."""
    params = params or {}
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    fg_mask_blocks, _block_var = segment(normalized)
    return normalized, fg_mask_blocks


def stage1_contrast(normalized, fg_mask_blocks, params=None):
    """Stage 1 (P1): Pipeline B's _wavelet_contrast, on the shared Step 0
    output. Real locked defaults confirmed against src/pipeline_b/
    pipeline_b.py's enhance() (lines 450-461): wavelet="db4", level=3,
    coarse_gain=1.60, fine_gain=1.00, coefficient_floor_percentile=25.0,
    blend=1.0 -- identical to the function's own signature defaults, i.e.
    pipeline_b.py never overrides them either."""
    params = params or {}
    return _wavelet_contrast(
        normalized,
        fg_mask_blocks=fg_mask_blocks,
        wavelet=params.get("wavelet", "db4"),
        level=params.get("wavelet_level", 3),
        coarse_gain=params.get("wavelet_coarse_gain", 1.60),
        fine_gain=params.get("wavelet_fine_gain", 1.00),
        coefficient_floor_percentile=params.get("wavelet_coefficient_floor_percentile", 25.0),
        blend=params.get("wavelet_contrast_blend", 1.0),
    )


def stage2_noise(normalized, stage1_output, fg_mask_blocks, params=None):
    """Stage 2 (P2): Pipeline C's _coherence_diffusion, on Stage 1's
    output. Reproduces pipeline_c.py's enhance() (lines 818-958) exactly
    for everything this stage needs, confirmed line by line:

      - Step 0c's alpha probe (lines 818-827): orientation_field() on the
        shared, pre-P1 Step 0 `normalized` image -- NOT stage1_output --
        followed by _aggressiveness_alpha(). Confirmed this runs on the
        normalised-but-unfiltered image regardless of which P1 technique
        conditions the image afterward (pipeline_c.py's own comment: "This
        probe orientation_field() call is separate from step 2's [...],
        this one only exists to set alpha"), so it is unaffected by
        substituting Pipeline B's P1 for Pipeline C's own homomorphic P1.
      - Step 2's real orientation field (line 911): orientation_field() on
        the post-P1 image, the same relative position pipeline_c.py itself
        uses (right after its own P1 step, right before diffusion) --
        substituting the hybrid's own Stage 1 output (stage1_output) in
        place of pipeline_c's homomorphic output, per the explicit
        instruction not to invent a different order or a different field.
      - diffusion_iterations/kappa (lines 945, 947): alpha-scaled via the
        exact same _lerp/_DIFFUSION_ITERATIONS_RANGE/_DIFFUSION_KAPPA_RANGE
        pipeline_c.py itself uses -- imported directly from pipeline_c, not
        duplicated or re-derived.
      - dt=0.2, confidence_ceiling=0.45 (lines 946, 958): pipeline_c.py's
        own fixed (non-adaptive) defaults for this call, reused unchanged.

    Passing `stage1_output=normalized` (i.e. the Step 0 output itself, with
    no P1 technique applied) reproduces the Hybrid-minus-P1 ablation
    variant with no special-casing needed here -- the "what to diffuse"
    input is already a separate argument from "what sets alpha", exactly
    because pipeline_c.py's own two orientation_field() calls are already
    on two different images.

    Returns (diffused_image, alpha) -- alpha is exposed for tests/
    diagnostics, not used by enhance() itself.
    """
    params = params or {}
    _theta_probe, coherence_probe = orientation_field(normalized)
    alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)

    theta_field, coherence_field = orientation_field(stage1_output)

    diffusion_iterations = params.get(
        "diffusion_iterations", int(round(_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha)))
    )
    diffusion_dt = params.get("diffusion_dt", 0.2)
    diffusion_kappa = params.get("diffusion_kappa", _lerp(_DIFFUSION_KAPPA_RANGE, alpha))
    diffused = _coherence_diffusion(
        stage1_output,
        theta_field,
        coherence_field,
        iterations=diffusion_iterations,
        dt=diffusion_dt,
        kappa=diffusion_kappa,
        confidence_ceiling=params.get("diffusion_confidence_ceiling", 0.45),
    )
    return diffused, alpha


def stage3_orientation(stage2_output, fg_mask_blocks, params=None):
    """Stage 3 (P6): Pipeline A's _oriented_gabor_filter, on Stage 2's
    output. orientation_field() is computed on stage2_output first, the
    same relative position pipeline_a.py itself uses (lines 204-208: its
    own `theta_field, coherence_field = orientation_field(denoised)`
    immediately before calling _oriented_gabor_filter, where `denoised` is
    its own Stage 2/bilateral-denoise output) -- substituting the hybrid's
    Stage 2 output in place of pipeline_a's own denoised output. Real
    locked defaults confirmed against pipeline_a.py's enhance() (lines
    208-219): kernel_size=17, sigma=4.0, wavelength=8.0, gamma=0.5,
    strength=0.7, orientation_bins=16, coherence_floor=0.2 -- identical to
    the function's own signature defaults."""
    params = params or {}
    theta_field, coherence_field = orientation_field(stage2_output)
    return _oriented_gabor_filter(
        stage2_output, theta_field, coherence_field, fg_mask_blocks,
        kernel_size=params.get("gabor_kernel_size", 17),
        sigma=params.get("gabor_sigma", 4.0),
        wavelength=params.get("gabor_wavelength", 8.0),
        gamma=params.get("gabor_gamma", 0.5),
        strength=params.get("gabor_strength", 0.7),
        orientation_bins=params.get("gabor_orientation_bins", 16),
        coherence_floor=params.get("gabor_coherence_floor", 0.2),
    )


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image

    Fixed hybrid pipeline (Handover Notes Priority 5 / Section 17): Step 0
    -> Pipeline B's wavelet contrast (P1) -> Pipeline C's coherence
    diffusion (P2) -> Pipeline A's oriented Gabor filtering (P6), applied
    identically to every image -- not adaptive/per-image technique
    selection; the fixed 3-stage composition Section 2.4 selected.
    """
    params = params or {}
    normalized, fg_mask_blocks = stage0_preprocess(image, params)
    stage1_output = stage1_contrast(normalized, fg_mask_blocks, params)
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
        cv2.imwrite("hybrid_test_output.png", out)
        print("Saved hybrid_test_output.png — open it and compare against the input image.")
