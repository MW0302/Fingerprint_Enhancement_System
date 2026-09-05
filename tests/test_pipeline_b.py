"""Regression tests for Pipeline B's independently callable P1/P2/P6 stages."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))

from pipeline_b import (  # noqa: E402
    _wavelet_contrast,
    _wavelet_shrinkage_denoise,
    _orientation_steered_morphology,
    enhance,
)


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


class WaveletDenoiseTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2133)
        clean = np.full((77, 91), 128.0, dtype=np.float32)
        self.noisy = np.clip(clean + rng.normal(0.0, 14.0, clean.shape), 0, 255).astype(np.uint8)

    def test_zero_threshold_is_exact_identity(self):
        actual = _wavelet_shrinkage_denoise(self.noisy, threshold_scale=0.0)
        np.testing.assert_array_equal(actual, self.noisy)

    def test_preserves_grayscale_contract(self):
        actual = _wavelet_shrinkage_denoise(self.noisy)
        self.assertEqual(actual.shape, self.noisy.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_reduces_synthetic_random_noise(self):
        actual = _wavelet_shrinkage_denoise(
            self.noisy,
            threshold_scale=1.0,
            denoise_finest_levels=2,
        )
        self.assertLess(float(actual.std()), float(self.noisy.std()))

    def test_zero_foreground_mask_leaves_image_unchanged(self):
        actual = _wavelet_shrinkage_denoise(
            self.noisy,
            fg_mask_blocks=np.zeros((4, 6), dtype=np.uint8),
        )
        np.testing.assert_array_equal(actual, self.noisy)

    def test_rejects_colour_input(self):
        colour = cv2.cvtColor(self.noisy, cv2.COLOR_GRAY2BGR)
        with self.assertRaises(ValueError):
            _wavelet_shrinkage_denoise(colour)


class OrientationSteeredMorphologyTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((65, 65), 220, dtype=np.uint8)
        self.image[31:34, 8:57] = 35
        self.image[31:34, 31:34] = 220
        self.theta = np.zeros((4, 4), dtype=np.float32)
        self.coherence = np.ones((4, 4), dtype=np.float32)

    def test_horizontal_orientation_fills_horizontal_ridge_gap(self):
        actual = _orientation_steered_morphology(
            self.image,
            self.theta,
            self.coherence,
            kernel_length=7,
            strength=1.0,
            max_darkening=255.0,
        )
        self.assertLess(int(actual[32, 32]), int(self.image[32, 32]))

    def test_perpendicular_orientation_does_not_fill_horizontal_gap(self):
        vertical_theta = np.full_like(self.theta, np.pi / 2)
        actual = _orientation_steered_morphology(
            self.image,
            vertical_theta,
            self.coherence,
            kernel_length=7,
            strength=1.0,
            max_darkening=255.0,
        )
        self.assertEqual(int(actual[32, 32]), int(self.image[32, 32]))

    def test_identity_strength_is_exact(self):
        actual = _orientation_steered_morphology(
            self.image,
            self.theta,
            self.coherence,
            strength=0.0,
        )
        np.testing.assert_array_equal(actual, self.image)

    def test_zero_foreground_mask_leaves_image_unchanged(self):
        actual = _orientation_steered_morphology(
            self.image,
            self.theta,
            self.coherence,
            fg_mask_blocks=np.zeros((4, 4), dtype=np.uint8),
        )
        np.testing.assert_array_equal(actual, self.image)

    def test_preserves_grayscale_contract(self):
        actual = _orientation_steered_morphology(
            self.image,
            self.theta,
            self.coherence,
        )
        self.assertEqual(actual.shape, self.image.shape)
        self.assertEqual(actual.dtype, np.uint8)

    def test_rejects_mismatched_fields(self):
        with self.assertRaises(ValueError):
            _orientation_steered_morphology(
                self.image,
                self.theta,
                np.ones((3, 4), dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
