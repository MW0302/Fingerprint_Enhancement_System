"""Regression tests for Hybrid variant 2b's independently callable Stage
0/1/2/3 functions (src/hybrid/hybrid_v2b.py) -- P1/P6 unchanged from
src/hybrid/hybrid.py, P2 replaced with Pipeline B's own wavelet shrinkage
denoise (instead of Pipeline C's diffusion).

Since Stage 0/1/3 are reused directly from hybrid.py (already tested in
tests/test_hybrid.py) and the new Stage 2 is a thin wrapper around one
already-tested pipeline_b.py primitive (no reimplemented logic), the main
risk this suite guards against is WIRING bugs in the new stage2_noise_b --
wrong input image, wrong defaults -- not the underlying algorithm itself."""

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "hybrid"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_a"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_c"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from pipeline_b import _wavelet_shrinkage_denoise  # noqa: E402

import hybrid_v2b  # noqa: E402


def _synthetic_fingerprint(height=96, width=112, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    ridges = 128 + 70 * np.sin(x / 6.0 + y / 30.0)
    noise = rng.normal(0.0, 6.0, (height, width))
    return np.clip(ridges + noise, 0, 255).astype(np.uint8)


class Stage2NoiseBTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()
        self.normalized, self.fg_mask_blocks = hybrid_v2b.stage0_preprocess(self.image)
        self.stage1_output = hybrid_v2b.stage1_contrast(self.normalized, self.fg_mask_blocks)

    def test_matches_direct_pipeline_b_call_exactly(self):
        """Stage 2 must be bit-identical to calling Pipeline B's own
        _wavelet_shrinkage_denoise directly with its own real defaults --
        confirms the wrapper isn't silently altering any parameter."""
        actual = hybrid_v2b.stage2_noise_b(self.stage1_output, self.fg_mask_blocks)
        expected = _wavelet_shrinkage_denoise(
            self.stage1_output,
            fg_mask_blocks=self.fg_mask_blocks,
            wavelet="db4", level=3, threshold_scale=1.00,
            denoise_finest_levels=1, blend=1.0, noise_adaptive=True,
            noise_reference_sigma=5.0, noise_adaptive_power=4.0,
            minimum_scale_factor=0.10,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_grayscale_contract(self):
        actual = hybrid_v2b.stage2_noise_b(self.stage1_output, self.fg_mask_blocks)
        self.assertEqual(actual.shape, self.stage1_output.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_operates_on_stage1_output_not_normalized(self):
        """Confirms Stage 2 diffuses/denoises Stage 1's output, not the
        pre-P1 normalized image -- the two must disagree given P1
        measurably changed the image."""
        actual_on_stage1 = hybrid_v2b.stage2_noise_b(self.stage1_output, self.fg_mask_blocks)
        actual_on_normalized = hybrid_v2b.stage2_noise_b(self.normalized, self.fg_mask_blocks)
        self.assertFalse(np.array_equal(actual_on_stage1, actual_on_normalized))

    def test_no_alpha_probe_needed_unlike_pipeline_c_diffusion(self):
        """Unlike hybrid.stage2_noise() (Pipeline C's diffusion, which
        needs the pre-P1 `normalized` image to compute alpha),
        stage2_noise_b takes no such argument -- confirms this stage's
        signature doesn't smuggle in a dependency on `normalized`."""
        import inspect
        params = list(inspect.signature(hybrid_v2b.stage2_noise_b).parameters)
        self.assertEqual(params, ["stage1_output", "fg_mask_blocks", "params"])


class EnhanceEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()

    def test_retains_input_contract(self):
        actual = hybrid_v2b.enhance(self.image)
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_output_differs_from_input(self):
        actual = hybrid_v2b.enhance(self.image)
        self.assertFalse(np.array_equal(actual, self.image))

    def test_equals_manual_three_stage_composition(self):
        """enhance() must be exactly stage0 -> stage1_contrast (hybrid.py's,
        unchanged) -> stage2_noise_b -> stage3_orientation (hybrid.py's,
        unchanged) chained, nothing more, nothing reordered."""
        normalized, fg_mask_blocks = hybrid_v2b.stage0_preprocess(self.image)
        stage1 = hybrid_v2b.stage1_contrast(normalized, fg_mask_blocks)
        stage2 = hybrid_v2b.stage2_noise_b(stage1, fg_mask_blocks)
        stage3 = hybrid_v2b.stage3_orientation(stage2, fg_mask_blocks)
        actual = hybrid_v2b.enhance(self.image)
        np.testing.assert_array_equal(actual, stage3)


if __name__ == "__main__":
    unittest.main()
