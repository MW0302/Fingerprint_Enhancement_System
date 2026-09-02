"""
Pipeline D — Member D: Frequency-Domain Contrast, Denoising, and
Orientation-Frequency Enhancement (FFT High-Frequency Emphasis Filtering +
Frequency-Domain Wiener Filtering + STFT-Based Joint
Orientation-Frequency Ridge Reconstruction)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)                -> FFT high-frequency
                                                       emphasis filtering
    P2 — high random noise (DB3)                   -> Frequency-domain
                                                       Wiener filtering
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
preprocessing. Steps 1-2 (FFT high-frequency emphasis and frequency-domain
Wiener filtering) are implemented. Step 3 (STFT-based ridge reconstruction)
remains a TODO. Find and verify an appropriate citation for each implemented
technique rather than assuming one is already settled.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import BLOCK, normalize_image, segment  # noqa: E402
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


def _frequency_domain_wiener_filter(
    image,
    fg_mask_blocks,
    noise_radius_low,
    noise_radius_high,
    noise_percentile,
    psd_smooth_radius,
    dc_protect_ratio,
    ridge_radius_low,
    ridge_radius_high,
    ridge_min_gain,
    min_gain,
    blend,
    pad_ratio,
):
    """Suppress broadband noise using a frequency-domain Wiener filter.

    All radius parameters are radial frequencies in cycles per pixel, built
    from :func:`numpy.fft.fftfreq`; they are unrelated to Step 1's
    frequency-pixel ``cutoff_ratio``. The scalar noise power is estimated
    robustly from a high-frequency annulus, while conservative gain floors
    protect the DC component and the normal fingerprint ridge band.

    ``fg_mask_blocks`` is applied only after the inverse FFT. Each mask cell
    controls its exact ``BLOCK`` by ``BLOCK`` image region, so incomplete
    right and bottom blocks remain unchanged. The observed power spectrum is
    locally averaged to stabilise its estimate; this smoothing acts on the
    frequency-domain PSD, not on the spatial image, and is therefore not
    spatial Gaussian denoising. No percentile intensity rescaling is applied.
    """
    image_float = np.asarray(image, dtype=np.float64)
    if image_float.ndim != 2 or image_float.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")

    mask = np.asarray(fg_mask_blocks)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("fg_mask_blocks must be a non-empty 2D array")

    scalar_parameters = (
        noise_radius_low,
        noise_radius_high,
        noise_percentile,
        dc_protect_ratio,
        ridge_radius_low,
        ridge_radius_high,
        ridge_min_gain,
        min_gain,
        blend,
        pad_ratio,
    )
    if not all(
        np.isscalar(value)
        and not np.iscomplexobj(value)
        and np.isfinite(value)
        for value in scalar_parameters
    ):
        raise ValueError("Wiener filter scalar parameters must be finite")
    if not 0 <= noise_radius_low < noise_radius_high <= 0.5:
        raise ValueError(
            "noise radii must satisfy 0 <= low < high <= 0.5 cycles/pixel"
        )
    if not 0 <= noise_percentile <= 100:
        raise ValueError("noise_percentile must be between zero and 100")
    if not 0 <= dc_protect_ratio:
        raise ValueError("dc_protect_ratio must be non-negative")
    if not 0 <= ridge_radius_low < ridge_radius_high <= 0.5:
        raise ValueError(
            "ridge radii must satisfy 0 <= low < high <= 0.5 cycles/pixel"
        )
    if isinstance(psd_smooth_radius, (bool, np.bool_)) or not isinstance(
        psd_smooth_radius, (int, np.integer)
    ):
        raise ValueError("psd_smooth_radius must be a non-negative integer")
    if psd_smooth_radius < 0:
        raise ValueError("psd_smooth_radius must be a non-negative integer")
    if not 0 <= ridge_min_gain <= 1:
        raise ValueError("ridge_min_gain must be between zero and one")
    if not 0 <= min_gain <= 1:
        raise ValueError("min_gain must be between zero and one")
    if not 0 <= blend <= 1:
        raise ValueError("blend must be between zero and one")
    if pad_ratio < 0:
        raise ValueError("pad_ratio must be non-negative")

    image_float = np.nan_to_num(
        image_float,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    input_range = np.clip(image_float, 0.0, 255.0)
    rows, cols = input_range.shape

    # Very small arrays do not contain a useful high-frequency annulus and
    # cannot always support reflect padding safely.
    if min(rows, cols) < 8:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    expected_mask_shape = (rows // BLOCK, cols // BLOCK)
    if mask.shape != expected_mask_shape:
        raise ValueError(
            "fg_mask_blocks shape must match the image's complete BLOCK geometry"
        )
    mask = mask.astype(bool, copy=False)
    if not np.any(mask) or np.isclose(np.var(input_range), 0.0):
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    pad_rows = int(round(pad_ratio * rows))
    pad_cols = int(round(pad_ratio * cols))
    if pad_rows >= rows or pad_cols >= cols:
        raise ValueError("reflect padding must be smaller than each image dimension")

    if pad_rows or pad_cols:
        padded = np.pad(
            input_range,
            ((pad_rows, pad_rows), (pad_cols, pad_cols)),
            mode="reflect",
        )
    else:
        padded = input_range

    padded_rows, padded_cols = padded.shape
    spectrum = np.fft.fftshift(np.fft.fft2(padded))
    observed_psd = np.abs(spectrum) ** 2 / padded.size

    if psd_smooth_radius:
        kernel_size = 2 * psd_smooth_radius + 1
        # This local average stabilises the frequency-domain power estimate;
        # it never smooths the spatial fingerprint image.
        smoothed_psd = cv2.blur(
            observed_psd,
            (kernel_size, kernel_size),
            borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        smoothed_psd = observed_psd

    fy = np.fft.fftshift(np.fft.fftfreq(padded_rows))[:, np.newaxis]
    fx = np.fft.fftshift(np.fft.fftfreq(padded_cols))[np.newaxis, :]
    radial_frequency = np.sqrt(fy**2 + fx**2)
    noise_annulus = (
        (radial_frequency >= noise_radius_low)
        & (radial_frequency <= noise_radius_high)
    )
    if not np.any(noise_annulus):
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    noise_power = float(
        np.percentile(smoothed_psd[noise_annulus], noise_percentile)
    )
    if not np.isfinite(noise_power) or noise_power <= np.finfo(np.float64).eps:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    signal_psd = np.maximum(smoothed_psd - noise_power, 0.0)
    epsilon = np.finfo(np.float64).eps * max(1.0, noise_power)
    gain = signal_psd / (signal_psd + noise_power + epsilon)
    gain = np.maximum(gain, min_gain)

    ridge_band = (
        (radial_frequency >= ridge_radius_low)
        & (radial_frequency <= ridge_radius_high)
    )
    gain[ridge_band] = np.maximum(gain[ridge_band], ridge_min_gain)
    gain[radial_frequency <= dc_protect_ratio] = 1.0

    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * gain)).real
    if pad_rows:
        row_slice = slice(pad_rows, pad_rows + rows)
    else:
        row_slice = slice(0, rows)
    if pad_cols:
        col_slice = slice(pad_cols, pad_cols + cols)
    else:
        col_slice = slice(0, cols)
    filtered = filtered[row_slice, col_slice]
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=255.0, neginf=0.0)

    alpha = np.zeros((rows, cols), dtype=np.float64)
    expanded_mask = np.repeat(np.repeat(mask, BLOCK, axis=0), BLOCK, axis=1)
    covered_rows = expected_mask_shape[0] * BLOCK
    covered_cols = expected_mask_shape[1] * BLOCK
    alpha[:covered_rows, :covered_cols] = expanded_mask

    output = input_range + alpha * blend * (filtered - input_range)
    output = np.nan_to_num(output, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


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

    # Step 2: frequency-domain Wiener filtering (P2 — high random noise).
    # Phase A diagnostics found a broadband high-frequency excess in DB3,
    # but no stable non-ridge spectral peaks across images, so this step uses
    # Wiener filtering only and deliberately contains no notch filter.
    denoised = _frequency_domain_wiener_filter(
        contrast_enhanced,
        fg_mask_blocks,
        noise_radius_low=params.get("wiener_noise_radius_low", 0.35),
        noise_radius_high=params.get("wiener_noise_radius_high", 0.48),
        noise_percentile=params.get("wiener_noise_percentile", 25.0),
        psd_smooth_radius=params.get("wiener_psd_smooth_radius", 2),
        dc_protect_ratio=params.get("wiener_dc_protect_ratio", 0.03),
        ridge_radius_low=params.get("wiener_ridge_radius_low", 0.04),
        ridge_radius_high=params.get("wiener_ridge_radius_high", 0.25),
        ridge_min_gain=params.get("wiener_ridge_min_gain", 0.80),
        min_gain=params.get("wiener_min_gain", 0.35),
        blend=params.get("wiener_blend", 0.20),
        pad_ratio=params.get("wiener_pad_ratio", 0.10),
    )

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
