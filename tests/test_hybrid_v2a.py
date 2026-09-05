"""Regression tests for Hybrid variant 2a's independently callable Stage
0/1/2/3 functions (src/hybrid/hybrid_v2a.py) -- P1 restored to Pipeline C's
own homomorphic+feathered stage, P2/P6 unchanged from src/hybrid/hybrid.py.

Since Stage 0/2/3 are reused directly from hybrid.py (already tested in
tests/test_hybrid.py) and the new Stage 1 is a thin wrapper around two
already-tested pipeline_c.py primitives (no reimplemented logic), the main
risk this suite guards against is WIRING bugs in the new stage1_contrast_c
-- wrong alpha source, wrong blend order, wrong constants -- not the
underlying algorithms themselves."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "hybrid"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_a"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_c"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from common import orientation_field  # noqa: E402
from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _homomorphic_filter,
    _HOMOMORPHIC_GAMMA_HIGH_RANGE,
)

import hybrid_v2a  # noqa: E402


def _synthetic_fingerprint(height=96, width=112, seed=7):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    ridges = 128 + 70 * np.sin(x / 6.0 + y / 30.0)
    noise = rng.normal(0.0, 6.0, (height, width))
    return np.clip(ridges + noise, 0, 255).astype(np.uint8)


class Stage1ContrastCTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()
        self.normalized, self.fg_mask_blocks = hybrid_v2a.stage0_preprocess(self.image)

    def test_matches_direct_pipeline_c_call_exactly(self):
        """Bit-identical to manually reconstructing pipeline_c.py's own
        Step 0c alpha probe + Step 1 homomorphic filter + Step 1b feather
        blend, using its own real defaults."""
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

        actual = hybrid_v2a.stage1_contrast_c(self.normalized, self.fg_mask_blocks)
        np.testing.assert_array_equal(actual, expected)

    def test_preserves_grayscale_contract(self):
        actual = hybrid_v2a.stage1_contrast_c(self.normalized, self.fg_mask_blocks)
        self.assertEqual(actual.shape, self.normalized.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_differs_from_pipeline_b_stage1(self):
        """Sanity check that this really is a different P1 technique from
        the current hybrid.py's Stage 1 (Pipeline B's wavelet contrast),
        not an accidental no-op or duplicate."""
        import hybrid
        stage1_b = hybrid.stage1_contrast(self.normalized, self.fg_mask_blocks)
        stage1_c = hybrid_v2a.stage1_contrast_c(self.normalized, self.fg_mask_blocks)
        self.assertFalse(np.array_equal(stage1_b, stage1_c))

    def test_background_is_fed_back_toward_normalized(self):
        """Step 1b's whole purpose is to keep background regions close to
        `normalized` rather than the raw homomorphic output -- confirm the
        all-background corner (fg_mask_blocks all zero there) stays much
        closer to `normalized` than an unblended homomorphic filter would
        leave it."""
        fg_mask_blocks = np.ones_like(self.fg_mask_blocks)
        fg_mask_blocks[:2, :2] = 0  # force a background corner
        actual = hybrid_v2a.stage1_contrast_c(self.normalized, fg_mask_blocks)
        raw = _homomorphic_filter(self.normalized, cutoff=0.06, gamma_low=0.5, gamma_high=2.0, sharpness=1.0)
        corner_slice = (slice(0, 16), slice(0, 16))
        dist_to_normalized = np.abs(
            actual[corner_slice].astype(np.int32) - self.normalized[corner_slice].astype(np.int32)
        ).mean()
        dist_to_raw_homomorphic = np.abs(
            actual[corner_slice].astype(np.int32) - raw[corner_slice].astype(np.int32)
        ).mean()
        self.assertLess(dist_to_normalized, dist_to_raw_homomorphic)


class EnhanceEndToEndTests(unittest.TestCase):
    def setUp(self):
        self.image = _synthetic_fingerprint()

    def test_retains_input_contract(self):
        actual = hybrid_v2a.enhance(self.image)
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_output_differs_from_input(self):
        actual = hybrid_v2a.enhance(self.image)
        self.assertFalse(np.array_equal(actual, self.image))

    def test_equals_manual_three_stage_composition(self):
        """enhance() must be exactly stage0 -> stage1_contrast_c ->
        stage2_noise (hybrid.py's, unchanged) -> stage3_orientation
        (hybrid.py's, unchanged) chained, nothing more, nothing reordered."""
        normalized, fg_mask_blocks = hybrid_v2a.stage0_preprocess(self.image)
        stage1 = hybrid_v2a.stage1_contrast_c(normalized, fg_mask_blocks)
        stage2, _alpha = hybrid_v2a.stage2_noise(normalized, stage1, fg_mask_blocks)
        stage3 = hybrid_v2a.stage3_orientation(stage2, fg_mask_blocks)
        actual = hybrid_v2a.enhance(self.image)
        np.testing.assert_array_equal(actual, stage3)


if __name__ == "__main__":
    unittest.main()
