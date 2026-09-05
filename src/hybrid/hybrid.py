"""
Hybrid pipeline (Handover Notes Priority 5 / Section 17): FIXED, not
adaptive -- one single pipeline (Step0 -> chosen P1 -> chosen P2 -> chosen
P6) applied identically to every image.

P1 was originally Pipeline B's wavelet contrast (the project lead's
paired-statistical override of Section 2.4's mechanical P1 winner -- paired
t-test/Wilcoxon on Pipeline B vs C's delta_P1 showed C's higher mean was not
significant, while B was more reliable on every other metric). Real 320-
image NFIQ2 validation of that assembly (docs/hybrid_validation_findings.md)
then found two real problems traced to P1/P2 being drawn from different
pipelines: a 20/80 (25%) DB3_B NFIQ2-scoreability collapse at Stage 2+, and
DB1_B netting -3.28 overall. A follow-up validation of alternative P1/P2
pairings (docs/hybrid_alternative_combinations.md, src/hybrid/hybrid_v2a.py)
confirmed the cause and fixed it: pairing Pipeline C's diffusion back with
Pipeline C's OWN P1 (homomorphic filtering + its Step 1b background-
feathering blend, not just the homomorphic filter alone) restores the
79/80 DB3_B scoreability ceiling every other pipeline already has, and
edges the overall mean above Pipeline C alone (51.95 vs 51.53). P1 was
switched to this pairing as a result -- see that doc for DB1_B's own
tradeoff under this composition (net -7.20, the worst DB1_B of any
combination tested, reported there in full and not hidden here):

    P1 (contrast)     -> Pipeline C's homomorphic filter + Step 1b
                         background-feathering blend
                         (_homomorphic_filter, src/pipeline_c/pipeline_c.py)
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
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "pipeline_c"))

from common import (  # noqa: E402
    normalize_image, segment, orientation_field,
    DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR,
)
from config import RAW_DIR  # noqa: E402

from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _homomorphic_filter,
    _HOMOMORPHIC_GAMMA_HIGH_RANGE,
    _coherence_diffusion,
    _DIFFUSION_ITERATIONS_RANGE,
    _DIFFUSION_KAPPA_RANGE,
)
from pipeline_a import _oriented_gabor_filter  # noqa: E402

import numpy as np
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
    """Stage 1 (P1): Pipeline C's own P1 stage -- _homomorphic_filter,
    immediately followed by its Step 1b background-feathering blend.
    Reproduces pipeline_c.py's enhance() (the block from "# Step 1:
    homomorphic filtering" through immediately before "# Step 2: ridge
    orientation field estimation") exactly, confirmed line by line:

      - Step 0c's alpha probe (pipeline_c.py, right before Step 1):
        orientation_field() on the shared, pre-P1 Step 0 `normalized`
        image, followed by _aggressiveness_alpha(normalized's coherence,
        fg_mask_blocks). This duplicates stage2_noise()'s own identical
        alpha computation below -- expected and harmless, since it is a
        deterministic function of `normalized`/`fg_mask_blocks` alone, not
        of anything this stage produces.
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
        Keeps background/low-confidence regions close to the pre-filter
        brightness rather than the raw homomorphic output; validated
        (docs/hybrid_alternative_combinations.md Section 3) as the piece
        that lets Stage 2's diffusion run safely on DB3_B's marginal-
        foreground images -- swapping in a different P1 technique ahead
        of the same diffusion call, without this blend, is what caused
        the DB3_B scoreability collapse documented in
        docs/hybrid_validation_findings.md."""
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
    -> Pipeline C's homomorphic filter + Step 1b feathering (P1) ->
    Pipeline C's coherence diffusion (P2) -> Pipeline A's oriented Gabor
    filtering (P6), applied identically to every image -- not adaptive/
    per-image technique selection; see this module's own docstring above
    for how P1 arrived at Pipeline C's own technique instead of Section
    2.4's original Pipeline B selection.
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
