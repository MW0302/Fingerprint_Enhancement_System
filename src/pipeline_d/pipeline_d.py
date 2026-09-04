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
preprocessing. Steps 1-3 are implemented as three separate internal helpers:
FFT high-frequency emphasis, frequency-domain Wiener filtering, and STFT-based
joint orientation-frequency ridge reconstruction. This separation supports a
later cumulative-ablation wrapper without making hybrid selection part of this
module. The estimator, spectral mask, inverse transform, and weighted
overlap-add are integral parts of the single counted STFT technique.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import (  # noqa: E402
    BLOCK,
    DEFAULT_NORMALIZE_TARGET_MEAN,
    DEFAULT_NORMALIZE_TARGET_VAR,
    normalize_image,
    segment,
)
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


def _stft_orientation_frequency_reconstruct(
    image,
    fg_mask_blocks,
    window_size,
    overlap_ratio,
    frequency_low,
    frequency_high,
    radial_bandwidth,
    angular_bandwidth,
    min_reliability,
    reconstruction_blend,
    pad_mode,
):
    """Reconstruct reliable local ridge spectra using overlapping 2D STFTs.

    Each locally mean-centred, square-root Hann-windowed spectrum supplies a
    joint probabilistic estimate of its ridge orientation and frequency. The
    estimate uses all energy in ``frequency_low`` to ``frequency_high``
    cycles/pixel rather than a single peak bin. A smooth, conjugate-symmetric
    radial/angular mask is then applied within the same STFT reconstruction.

    Reliability continuously blends the transfer function back towards the
    identity, so uncertain, curved, damaged, or low-energy windows are left
    conservative. Square-root Hann synthesis and denominator-normalised
    weighted overlap-add avoid block seams and preserve local brightness. The
    shared foreground mask is never multiplied into the image before an FFT;
    it only gates estimation and the final reconstructed pixels. Incomplete
    right/bottom ``BLOCK`` remainders consequently retain the input exactly.
    """
    image_float = np.asarray(image, dtype=np.float64)
    if image_float.ndim != 2 or image_float.size == 0:
        raise ValueError("image must be a non-empty 2D grayscale array")

    mask = np.asarray(fg_mask_blocks)
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("fg_mask_blocks must be a non-empty 2D array")

    if isinstance(window_size, (bool, np.bool_)) or not isinstance(
        window_size, (int, np.integer)
    ):
        raise ValueError("window_size must be an even integer of at least 8")
    if window_size < 8 or window_size % 2:
        raise ValueError("window_size must be an even integer of at least 8")

    scalar_parameters = (
        overlap_ratio,
        frequency_low,
        frequency_high,
        radial_bandwidth,
        angular_bandwidth,
        min_reliability,
        reconstruction_blend,
    )
    if not all(
        np.isscalar(value)
        and not np.iscomplexobj(value)
        and np.isfinite(value)
        for value in scalar_parameters
    ):
        raise ValueError("STFT reconstruction scalar parameters must be finite")
    if not 0 <= overlap_ratio < 1:
        raise ValueError("overlap_ratio must satisfy 0 <= overlap_ratio < 1")
    if not 0 <= frequency_low < frequency_high <= 0.5:
        raise ValueError(
            "frequencies must satisfy 0 <= low < high <= 0.5 cycles/pixel"
        )
    if radial_bandwidth <= 0:
        raise ValueError("radial_bandwidth must be greater than zero")
    if not 0 < angular_bandwidth <= 90:
        raise ValueError("angular_bandwidth must satisfy 0 < value <= 90 degrees")
    if not 0 <= min_reliability <= 1:
        raise ValueError("min_reliability must be between zero and one")
    if not 0 <= reconstruction_blend <= 1:
        raise ValueError("reconstruction_blend must be between zero and one")
    if pad_mode != "reflect":
        raise ValueError("pad_mode must be 'reflect'")

    hop = int(round(window_size * (1.0 - overlap_ratio)))
    if hop < 1:
        raise ValueError("overlap_ratio must produce a hop of at least one pixel")

    image_float = np.nan_to_num(
        image_float,
        nan=0.0,
        posinf=255.0,
        neginf=0.0,
    )
    input_range = np.clip(image_float, 0.0, 255.0)
    rows, cols = input_range.shape

    # Small images cannot provide a full analysis window or legal reflect
    # padding. They remain a conservative pass-through.
    if min(rows, cols) < window_size:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    expected_mask_shape = (rows // BLOCK, cols // BLOCK)
    if mask.shape != expected_mask_shape:
        raise ValueError(
            "fg_mask_blocks shape must match the image's complete BLOCK geometry"
        )
    mask = mask.astype(bool, copy=False)
    if (
        not np.any(mask)
        or np.isclose(np.var(input_range), 0.0)
        or reconstruction_blend == 0
        or min_reliability == 1
    ):
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    foreground = np.zeros((rows, cols), dtype=bool)
    expanded_mask = np.repeat(np.repeat(mask, BLOCK, axis=0), BLOCK, axis=1)
    covered_rows = expected_mask_shape[0] * BLOCK
    covered_cols = expected_mask_shape[1] * BLOCK
    foreground[:covered_rows, :covered_cols] = expanded_mask

    pad = window_size // 2
    if pad >= rows or pad >= cols:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)
    padded = np.pad(input_range, ((pad, pad), (pad, pad)), mode=pad_mode)
    padded_foreground = np.pad(
        foreground.astype(np.float64),
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )
    real_support = np.pad(
        np.ones((rows, cols), dtype=np.float64),
        ((pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0.0,
    )

    hann_1d = np.hanning(window_size)
    analysis_window = np.sqrt(np.outer(hann_1d, hann_1d))
    synthesis_window = analysis_window
    window_product = analysis_window * synthesis_window
    window_weight = float(np.sum(window_product))
    epsilon = np.finfo(np.float64).eps
    if window_weight <= epsilon:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    fy = np.fft.fftshift(np.fft.fftfreq(window_size))[:, np.newaxis]
    fx = np.fft.fftshift(np.fft.fftfreq(window_size))[np.newaxis, :]
    radial_frequency = np.hypot(fy, fx)
    spectral_angle = np.arctan2(fy, fx)
    ridge_band = (
        (radial_frequency >= frequency_low)
        & (radial_frequency <= frequency_high)
    )
    non_dc = radial_frequency >= max(1.0 / window_size, frequency_low)
    if not np.any(ridge_band):
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    padded_rows, padded_cols = padded.shape
    row_starts = list(range(0, padded_rows - window_size + 1, hop))
    col_starts = list(range(0, padded_cols - window_size + 1, hop))
    final_row_start = padded_rows - window_size
    final_col_start = padded_cols - window_size
    if row_starts[-1] != final_row_start:
        row_starts.append(final_row_start)
    if col_starts[-1] != final_col_start:
        col_starts.append(final_col_start)

    numerator = np.zeros_like(padded, dtype=np.float64)
    denominator = np.zeros_like(padded, dtype=np.float64)
    reliable_windows = 0
    angular_sigma = np.deg2rad(angular_bandwidth)
    radial_scale = max((frequency_high - frequency_low) / 4.0, epsilon)

    for row_start in row_starts:
        row_slice = slice(row_start, row_start + window_size)
        for col_start in col_starts:
            col_slice = slice(col_start, col_start + window_size)
            patch = padded[row_slice, col_slice]
            local_mean = float(np.mean(patch))
            identity_patch = patch * window_product
            reconstructed_patch = identity_patch

            foreground_support = float(
                np.sum(padded_foreground[row_slice, col_slice] * window_product)
                / window_weight
            )
            if foreground_support >= 0.5:
                tapered = (patch - local_mean) * analysis_window
                spectrum = np.fft.fftshift(np.fft.fft2(tapered))
                power = np.abs(spectrum) ** 2
                band_power = np.where(ridge_band, power, 0.0)
                total_band_power = float(np.sum(band_power))
                total_non_dc_power = float(np.sum(power[non_dc]))

                if total_band_power > epsilon and total_non_dc_power > epsilon:
                    probability = band_power / total_band_power
                    doubled_axis = np.sum(
                        probability * np.exp(2j * spectral_angle)
                    )
                    angular_reliability = float(np.abs(doubled_axis))
                    axis_angle = 0.5 * float(np.angle(doubled_axis))
                    ridge_orientation = (axis_angle + np.pi / 2.0) % np.pi
                    estimated_frequency = float(
                        np.sum(probability * radial_frequency)
                    )
                    radial_variance = float(
                        np.sum(
                            probability
                            * (radial_frequency - estimated_frequency) ** 2
                        )
                    )
                    radial_std = np.sqrt(max(0.0, radial_variance))
                    radial_reliability = float(
                        np.clip(1.0 - radial_std / radial_scale, 0.0, 1.0)
                    )
                    band_fraction = float(
                        np.clip(total_band_power / total_non_dc_power, 0.0, 1.0)
                    )
                    reliability = (
                        angular_reliability
                        * radial_reliability
                        * np.sqrt(band_fraction)
                    )

                    if reliability > min_reliability:
                        reliability_ramp = np.clip(
                            (reliability - min_reliability)
                            / (1.0 - min_reliability),
                            0.0,
                            1.0,
                        )
                        edge_support = float(
                            np.sum(real_support[row_slice, col_slice] * window_product)
                            / window_weight
                        )
                        reconstruction_strength = (
                            reconstruction_blend
                            * reliability_ramp
                            * edge_support
                        )

                        radial_mask = np.exp(
                            -(
                                (radial_frequency - estimated_frequency) ** 2
                            )
                            / (2.0 * radial_bandwidth**2)
                        )
                        spectral_axis = ridge_orientation - np.pi / 2.0
                        axis_distance = np.abs(
                            (
                                spectral_angle
                                - spectral_axis
                                + np.pi / 2.0
                            )
                            % np.pi
                            - np.pi / 2.0
                        )
                        angular_mask = np.exp(
                            -(axis_distance**2) / (2.0 * angular_sigma**2)
                        )
                        spectral_mask = radial_mask * angular_mask
                        transfer = (
                            1.0 - reconstruction_strength
                            + reconstruction_strength * spectral_mask
                        )

                        reconstructed_complex = np.fft.ifft2(
                            np.fft.ifftshift(spectrum * transfer)
                        )
                        imaginary_residual = float(
                            np.max(np.abs(reconstructed_complex.imag))
                        )
                        real_scale = max(
                            1.0,
                            float(np.max(np.abs(reconstructed_complex.real))),
                        )
                        if imaginary_residual <= 1e-10 * real_scale:
                            restored_tapered = (
                                reconstructed_complex.real
                                + local_mean * analysis_window
                            )
                            reconstructed_patch = (
                                restored_tapered * synthesis_window
                            )
                            reliable_windows += 1

            numerator[row_slice, col_slice] += reconstructed_patch
            denominator[row_slice, col_slice] += window_product

    if reliable_windows == 0:
        return np.clip(np.rint(input_range), 0, 255).astype(np.uint8)

    reconstructed_padded = np.divide(
        numerator,
        denominator,
        out=padded.copy(),
        where=denominator > epsilon,
    )
    reconstructed = reconstructed_padded[pad : pad + rows, pad : pad + cols]
    output = input_range.copy()
    output[foreground] = reconstructed[foreground]
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
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
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
    # (P6 — weak/inconsistent ridge orientation). Its estimator, smooth
    # spectral mask, inverse FFT, and weighted overlap-add are one counted
    # STFT reconstruction technique. No shared orientation field is used.
    enhanced = _stft_orientation_frequency_reconstruct(
        denoised,
        fg_mask_blocks,
        window_size=params.get("stft_window_size", 32),
        overlap_ratio=params.get("stft_overlap_ratio", 0.75),
        frequency_low=params.get("stft_frequency_low", 0.04),
        frequency_high=params.get("stft_frequency_high", 0.25),
        radial_bandwidth=params.get("stft_radial_bandwidth", 0.025),
        angular_bandwidth=params.get("stft_angular_bandwidth", 20.0),
        min_reliability=params.get("stft_min_reliability", 0.12),
        reconstruction_blend=params.get("stft_reconstruction_blend", 0.25),
        pad_mode=params.get("stft_pad_mode", "reflect"),
    )

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
