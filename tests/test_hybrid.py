"""Regression tests for the fixed Hybrid pipeline's independently callable
Stage 0/1/2/3 functions (src/hybrid/hybrid.py).

Since each stage is a thin, read-only wrapper around an already-tested
function from pipeline_a/b/c (no reimplemented logic), the main risk this
suite guards against is WIRING bugs -- passing the wrong image into the
wrong call, or computing a supporting field (orientation_field) on the
wrong stage's output -- not the underlying algorithms themselves (already
covered by each source pipeline's own tests)."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "hybrid"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_a"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_c"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from common import orientation_field, segment  # noqa: E402
from pipeline_a import _oriented_gabor_filter  # noqa: E402
from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _homomorphic_filter,
    _HOMOMORPHIC_GAMMA_HIGH_RANGE,
    _coherence_diffusion,
    _DIFFUSION_ITERATIONS_RANGE,
    _DIFFUSION_KAPPA_RANGE,
)

import hybrid  # noqa: E402


def _synthetic_fingerprint(height=96, width=112, seed=7):
    """A synthetic ridge-like pattern large enough (multiple of BLOCK=16)
    for segment()/orientation_field() to produce meaningful, non-degenerate
    output -- not a real fingerprint, just structured enough to exercise
    every stage without erroring."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    ridges = 128 + 70 * np.sin(x / 6.0 + y / 30.0)
    noise = rng.normal(0.0, 6.0, (height, width))
    return np.clip(ridges + noise, 0, 255).astype(np.uint8)


class Stage0PreprocessTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()

    def test_normalized_preserves_shape_and_dtype(self):
        normalized, fg_mask_blocks = hybrid.stage0_preprocess(self.image)
        self.assertEqual(normalized.shape, self.image.shape)
        self.assertEqual(normalized.dtype, np.uint8)

    def test_fg_mask_blocks_matches_segment_block_geometry(self):
        normalized, fg_mask_blocks = hybrid.stage0_preprocess(self.image)
        expected_shape = (self.image.shape[0] // 16, self.image.shape[1] // 16)
        self.assertEqual(fg_mask_blocks.shape, expected_shape)


class Stage1ContrastTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()
        self.normalized, self.fg_mask_blocks = hybrid.stage0_preprocess(self.image)

    def test_matches_direct_pipeline_c_call_exactly(self):
        """Stage 1 must be bit-identical to manually reconstructing
        pipeline_c.py's own Step 0c alpha probe + Step 1 homomorphic
        filter + Step 1b feather blend, using its own real defaults --
        confirms the wrapper isn't silently altering any parameter."""
        _theta_probe, coherence_probe = orientation_field(self.normalized)
        alpha = _aggressiveness_alpha(coherence_probe, self.fg_mask_blocks)
        contrast_enhanced_raw = _homomorphic_filter(
            self.normalized,
            cutoff=0.06, gamma_low=0.5,
            gamma_high=_lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha),
            sharpness=1.0,
        )
        h, w = self.normalized.shape
        fg_alpha = cv2.resize(
            self.fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
        fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), 8.0)
        fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
        expected = (
            fg_alpha * contrast_enhanced_raw.astype(np.float64)
            + (1 - fg_alpha) * self.normalized.astype(np.float64)
        )
        expected = np.clip(expected, 0, 255).astype(np.uint8)

        actual = hybrid.stage1_contrast(self.normalized, self.fg_mask_blocks)
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_grayscale_contract(self):
        actual = hybrid.stage1_contrast(self.normalized, self.fg_mask_blocks)
        self.assertEqual(actual.shape, self.normalized.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_background_is_fed_back_toward_normalized(self):
        """Step 1b's whole purpose is to keep background regions close to
        `normalized` rather than the raw homomorphic output -- confirm the
        all-background corner (fg_mask_blocks all zero there) stays much
        closer to `normalized` than an unblended homomorphic filter would
        leave it."""
        fg_mask_blocks = np.ones_like(self.fg_mask_blocks)
        fg_mask_blocks[:2, :2] = 0  # force a background corner
        actual = hybrid.stage1_contrast(self.normalized, fg_mask_blocks)
        raw = _homomorphic_filter(self.normalized, cutoff=0.06, gamma_low=0.5, gamma_high=2.0, sharpness=1.0)
        corner_slice = (slice(0, 16), slice(0, 16))
        dist_to_normalized = np.abs(
            actual[corner_slice].astype(np.int32) - self.normalized[corner_slice].astype(np.int32)
        ).mean()
        dist_to_raw_homomorphic = np.abs(
            actual[corner_slice].astype(np.int32) - raw[corner_slice].astype(np.int32)
        ).mean()
        self.assertLess(dist_to_normalized, dist_to_raw_homomorphic)

    def test_rejects_colour_input_via_underlying_function(self):
        colour = cv2.cvtColor(self.normalized, cv2.COLOR_GRAY2BGR)
        with self.assertRaises(ValueError):
            hybrid.stage1_contrast(colour, self.fg_mask_blocks)


class Stage2NoiseTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()
        self.normalized, self.fg_mask_blocks = hybrid.stage0_preprocess(self.image)
        self.stage1_output = hybrid.stage1_contrast(self.normalized, self.fg_mask_blocks)

    def test_matches_direct_pipeline_c_wiring_exactly(self):
        """Reproduces pipeline_c.py's own Step 0c probe + Step 2 field +
        diffusion call manually, using the same alpha this stage should
        compute, and checks the result is bit-identical."""
        _theta_probe, coherence_probe = orientation_field(self.normalized)
        alpha = _aggressiveness_alpha(coherence_probe, self.fg_mask_blocks)
        theta_field, coherence_field = orientation_field(self.stage1_output)
        expected = _coherence_diffusion(
            self.stage1_output, theta_field, coherence_field,
            iterations=int(round(_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha))),
            dt=0.2,
            kappa=_lerp(_DIFFUSION_KAPPA_RANGE, alpha),
            confidence_ceiling=0.45,
        )
        actual, actual_alpha = hybrid.stage2_noise(self.normalized, self.stage1_output, self.fg_mask_blocks)
        self.assertAlmostEqual(actual_alpha, alpha, places=9)
        np.testing.assert_array_equal(actual, expected)

    def test_alpha_depends_only_on_normalized_not_on_stage1_output(self):
        """Critical design property: the alpha probe must use the shared
        Step 0 `normalized` image, not whatever Stage 1 produced -- so
        swapping in a very different "Stage 1 output" must not change
        alpha at all."""
        _, alpha_with_real_stage1 = hybrid.stage2_noise(
            self.normalized, self.stage1_output, self.fg_mask_blocks
        )
        decoy_stage1 = np.clip(self.stage1_output.astype(np.int32) + 40, 0, 255).astype(np.uint8)
        _, alpha_with_decoy_stage1 = hybrid.stage2_noise(
            self.normalized, decoy_stage1, self.fg_mask_blocks
        )
        self.assertAlmostEqual(alpha_with_real_stage1, alpha_with_decoy_stage1, places=9)

    def test_minus_p1_variant_diffuses_normalized_directly(self):
        """Hybrid-minus-P1 is produced by calling stage2_noise with
        stage1_output=normalized (skipping Stage 1 entirely) -- confirm
        this actually diffuses `normalized`, not the real stage1_output,
        by checking the two variants disagree given P1 measurably changed
        the image."""
        actual_full, _ = hybrid.stage2_noise(self.normalized, self.stage1_output, self.fg_mask_blocks)
        actual_minus_p1, _ = hybrid.stage2_noise(self.normalized, self.normalized, self.fg_mask_blocks)
        self.assertFalse(np.array_equal(actual_full, actual_minus_p1))

    def test_preserves_grayscale_contract(self):
        actual, _alpha = hybrid.stage2_noise(self.normalized, self.stage1_output, self.fg_mask_blocks)
        self.assertEqual(actual.shape, self.normalized.shape)
        self.assertEqual(actual.dtype, np.uint8)


class Stage3OrientationTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()
        self.normalized, self.fg_mask_blocks = hybrid.stage0_preprocess(self.image)
        self.stage1_output = hybrid.stage1_contrast(self.normalized, self.fg_mask_blocks)
        self.stage2_output, _alpha = hybrid.stage2_noise(
            self.normalized, self.stage1_output, self.fg_mask_blocks
        )

    def test_matches_direct_pipeline_a_call_exactly(self):
        theta_field, coherence_field = orientation_field(self.stage2_output)
        expected = _oriented_gabor_filter(
            self.stage2_output, theta_field, coherence_field, self.fg_mask_blocks,
            kernel_size=17, sigma=4.0, wavelength=8.0, gamma=0.5,
            strength=0.7, orientation_bins=16, coherence_floor=0.2,
        )
        actual = hybrid.stage3_orientation(self.stage2_output, self.fg_mask_blocks)
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_grayscale_contract(self):
        actual = hybrid.stage3_orientation(self.stage2_output, self.fg_mask_blocks)
        self.assertEqual(actual.shape, self.stage2_output.shape)
        self.assertEqual(actual.dtype, np.uint8)


class EnhanceEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()

    def test_retains_input_contract(self):
        actual = hybrid.enhance(self.image)
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_output_differs_from_input(self):
        """All three stages are non-identity by default -- the fixed
        hybrid should actually change a real image, not pass it through."""
        actual = hybrid.enhance(self.image)
        self.assertFalse(np.array_equal(actual, self.image))

    def test_equals_manual_three_stage_composition(self):
        """enhance() must be exactly stage0 -> stage1 -> stage2 -> stage3
        chained, nothing more, nothing reordered."""
        normalized, fg_mask_blocks = hybrid.stage0_preprocess(self.image)
        stage1 = hybrid.stage1_contrast(normalized, fg_mask_blocks)
        stage2, _alpha = hybrid.stage2_noise(normalized, stage1, fg_mask_blocks)
        stage3 = hybrid.stage3_orientation(stage2, fg_mask_blocks)
        actual = hybrid.enhance(self.image)
        np.testing.assert_array_equal(actual, stage3)


if __name__ == "__main__":
    unittest.main()
