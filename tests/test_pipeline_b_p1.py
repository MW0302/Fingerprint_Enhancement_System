"""Focused regression tests for Pipeline B's independently callable P1 stage."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from pipeline_b import _wavelet_contrast, enhance  # noqa: E402


class WaveletContrastTests(unittest.TestCase):
    def setUp(self):
        x = np.linspace(35, 220, 91, dtype=np.float32)
        y = np.linspace(0, 8 * np.pi, 77, dtype=np.float32)[:, None]
        self.image = np.clip(x + 22 * np.sin(y), 0, 255).astype(np.uint8)

    def test_identity_parameters_are_exact(self):
        actual = _wavelet_contrast(
            self.image,
            wavelet="db4",
            level=3,
            coarse_gain=1.0,
            fine_gain=1.0,
        )
        np.testing.assert_array_equal(actual, self.image)

    def test_preserves_grayscale_contract_on_odd_shape(self):
        actual = _wavelet_contrast(self.image)
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)
        self.assertGreaterEqual(int(actual.min()), 0)
        self.assertLessEqual(int(actual.max()), 255)

    def test_zero_foreground_mask_leaves_image_unchanged(self):
        mask = np.zeros((5, 7), dtype=np.uint8)
        actual = _wavelet_contrast(self.image, fg_mask_blocks=mask)
        np.testing.assert_array_equal(actual, self.image)

    def test_rejects_colour_input(self):
        colour = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        with self.assertRaises(ValueError):
            _wavelet_contrast(colour)

    def test_pipeline_enhance_retains_input_contract(self):
        actual = enhance(self.image)
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
