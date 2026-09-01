"""
Pipeline D — Member D: Frequency-Domain Contrast, Denoising, and
Orientation-Frequency Enhancement (FFT High-Frequency Emphasis Filtering +
Frequency-Domain Wiener/Notch Filtering + STFT-Based Joint
Orientation-Frequency Ridge Reconstruction)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)                -> FFT high-frequency
                                                       emphasis filtering
    P2 — high random noise (DB3)                   -> Frequency-domain
                                                       Wiener / notch filtering
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                                -> STFT-based joint
                                                       orientation-frequency
                                                       ridge reconstruction

Segmentation and block-wise normalisation, revision history: dropped on 30
August 2026 because — with only P1/P2/P6 in scope — having all four
pipelines call the identical Otsu/Hong et al. steps as a counted technique
would have violated the lecturer's no-repeated-technique requirement. Added
back the same day (later revision) as a shared PREPROCESSING stage (Steps
0a-0b below) used by all four pipelines uniformly: this does not reopen the
repeated-technique issue because preprocessing that conditions the image
without independently solving P1/P2/P6 itself is not one of the three
counted techniques (see common.py's module docstring — the same reasoning
already applied to orientation_field(), shared by Pipelines A, B, and C).
It was made uniform across all four pipelines, rather than added to just
one, to satisfy the group's own "Fair Experimental Conditions" principle
(Handover Notes) — every pipeline should start from the same input so
differences in the results reflect the three counted techniques, not
differing preprocessing. The earlier "light morphological clean-up" step
is still dropped — it was never one of the three shared problems.

Steps 0a-0b (normalisation, segmentation) are implemented as shared
preprocessing. Step 1's FFT high-frequency emphasis filtering is implemented;
Steps 2-3 (frequency-domain Wiener/notch filtering and STFT-based ridge
reconstruction) remain TODOs. Find and verify an appropriate citation for
each implemented technique rather than assuming one is already settled.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import normalize_image, segment  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def _fft_high_frequency_emphasis(
    image,
    cutoff_ratio,
    low_gain,
    high_boost,
    percentile_low,
    percentile_high,
    blend,
):
    """Enhance local contrast using two-dimensional FFT emphasis filtering.

    ``cutoff_ratio`` sets the Gaussian high-pass cutoff as a fraction of the
    image's shorter dimension: lower values admit more mid-frequency detail,
    while higher values restrict emphasis to finer structure. ``low_gain``
    retains the low-frequency image content and ``high_boost`` adds
    increasing gain above the cutoff.

    The inverse FFT can contain a small number of extreme pixels, so robust
    percentile rescaling is used instead of a true min/max stretch. ``blend``
    mixes the rescaled result with the input to keep the pilot defaults
    conservative and reduce over-enhancement of already high-contrast DB1
    images.
    """
    image_float = np.asarray(image, dtype=np.float64)
    if image_float.ndim != 2 or image_float.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")

    parameter_values = (
        cutoff_ratio,
        low_gain,
        high_boost,
        percentile_low,
        percentile_high,
        blend,
    )
    if not all(np.isfinite(value) for value in parameter_values):
        raise ValueError("FFT emphasis parameters must be finite")
    if cutoff_ratio <= 0:
        raise ValueError("cutoff_ratio must be greater than zero")
    if low_gain < 0 or high_boost < 0:
        raise ValueError("low_gain and high_boost must be non-negative")
    if not 0 <= percentile_low < percentile_high <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    if not 0 <= blend <= 1:
        raise ValueError("blend must be between zero and one")

    image_float = np.nan_to_num(image_float, nan=0.0, posinf=255.0, neginf=0.0)
    input_range = np.clip(image_float, 0.0, 255.0)

    rows, cols = image_float.shape
    spectrum = np.fft.fftshift(np.fft.fft2(image_float))

    row_coordinates = np.arange(rows, dtype=np.float64) - rows // 2
    col_coordinates = np.arange(cols, dtype=np.float64) - cols // 2
    distance_squared = (
        row_coordinates[:, np.newaxis] ** 2
        + col_coordinates[np.newaxis, :] ** 2
    )
    cutoff = cutoff_ratio * min(rows, cols)
    high_pass = 1.0 - np.exp(-distance_squared / (2.0 * cutoff**2))
    transfer = low_gain + high_boost * high_pass

    filtered_spectrum = spectrum * transfer
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_spectrum)).real
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)

    lower, upper = np.percentile(filtered, [percentile_low, percentile_high])
    if np.isclose(upper, lower, rtol=1e-12, atol=1e-12):
        rescaled = input_range
    else:
        rescaled = (filtered - lower) * (255.0 / (upper - lower))
        rescaled = np.clip(rescaled, 0.0, 255.0)

    blended = (1.0 - blend) * input_range + blend * rescaled
    blended = np.nan_to_num(blended, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 0a-0b: block-wise normalisation + Otsu segmentation (Hong, Wan, &
    # Jain, 1998) — shared preprocessing, not one of this pipeline's three
    # primary techniques (see module docstring). fg_mask_blocks is available
    # if your own Step 2/3 implementation wants to skip or de-emphasise
    # background blocks; it's not required. Unlike Pipelines A/B/C, this
    # pipeline still doesn't need orientation_field() — Step 3's STFT
    # technique estimates orientation and frequency jointly on its own.
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", 100.0),
        target_var=params.get("normalize_target_var", 1600.0),
    )
    fg_mask_blocks, _block_var = segment(normalized)

    # Step 1: FFT high-frequency emphasis filtering (P1 — low global
    # contrast).
    # Core idea: take the 2D FFT of the image (np.fft.fft2 / fftshift),
    # apply a high-frequency emphasis transfer function (boosts high
    # spatial frequencies relative to low ones, which sharpens local
    # contrast without a full log-domain homomorphic decomposition), then
    # inverse-transform back to the spatial domain. Ask yourself:
    #   - what cutoff separates "illumination-scale" from "ridge-scale"
    #     frequency content in this dataset?
    #   - how do you rescale the output back to a valid 0-255 range without
    #     a few outlier pixels dominating the stretch? (Consider a robust
    #     percentile-based rescale rather than true min/max.)
    contrast_enhanced = _fft_high_frequency_emphasis(
        normalized,
        cutoff_ratio=params.get("fft_cutoff_ratio", 0.06),
        low_gain=params.get("fft_low_gain", 0.95),
        high_boost=params.get("fft_high_boost", 0.75),
        percentile_low=params.get("fft_percentile_low", 1.0),
        percentile_high=params.get("fft_percentile_high", 99.0),
        blend=params.get("fft_blend", 0.55),
    )

    # Step 2: frequency-domain Wiener / notch filtering (P2 — high random
    # noise) — TODO
    # Core idea: DB3's noise is not perfectly white — the qualitative
    # vertical-banding finding (P2b, see the analysis document Section 3)
    # suggests some structured/periodic component. A notch filter can
    # suppress specific frequency-domain peaks corresponding to that
    # banding, while a Wiener filter provides a more general statistical
    # denoising approach in the frequency domain. Ask yourself:
    #   - can you identify the banding frequency peak(s) directly from the
    #     2D FFT magnitude spectrum of a noisy DB3 image?
    #   - how do you estimate the noise/signal power ratio a Wiener filter
    #     needs, given you don't have a clean reference image?
    denoised = contrast_enhanced.copy()  # placeholder — replace once this step is implemented

    # Step 3: STFT-based joint orientation-frequency ridge reconstruction
    # (P6 — weak/inconsistent ridge orientation) — TODO
    # Core idea: slide a window over the image, take a 2D FFT of each window
    # (np.fft.fft2), and from its spectrum jointly estimate:
    #   - local ridge orientation (angle of the dominant frequency peak)
    #   - local ridge frequency (distance of the peak from the origin)
    #   - local energy (peak magnitude)
    # then reconstruct/enhance each window using this information and stitch
    # the windows back together (with overlap to avoid seams). Ask yourself:
    #   - what window size balances "local enough" against "enough ridges
    #     inside the window to get a clean frequency peak"?
    #   - how much should windows overlap, and how do you blend the overlap
    #     region to avoid visible block seams?
    enhanced = denoised  # placeholder — replace once this step is implemented

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB2_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_d_test_output.png", out)
        print("Saved pipeline_d_test_output.png — open it and compare against the input image.")
