"""
EXPERIMENT 2 — confidence-gated Log-Gabor "replace" mode.

This is a STANDALONE COPY of src/pipeline_c/pipeline_c.py (verified version),
NOT an edit of the original — the original is untouched. The only change is
in _log_gabor_enhance(): a new `mode="replace_confidence_gated"` option
(default stays "additive", i.e. byte-identical to the verified pipeline
unless explicitly requested via params).

Background: the verified pipeline's docstring records that a direct LR2-
style REPLACE (output = filtered bandpass response, at any rescaling) was
tried three times and failed all three, for a structural reason: a Log-Gabor
filter keeps one narrow frequency band (~a sine wave); a real ridge/valley
pattern is closer to a square wave (energy at the fundamental + harmonics),
so replacing always trades away sharp contrast for a smoother sinusoidal
look, regardless of which blocks it's applied to. All three prior attempts
gated replace by fg_mask_blocks (foreground/background) or nothing at all —
never by whether THIS block's own local orientation/frequency estimate is
actually trustworthy.

This experiment reuses _log_gabor_enhance's existing FIFTH-revision
per-block gain_confidence (built from coherence_field, already computed for
the additive mode) to gate replace itself: only blocks where confidence is
high (>= replace_confidence_high) get replaced outright (rescaled to that
block's own local std in the diffused image); blocks where confidence is low
(<= replace_confidence_low) are left as the diffused image, completely
untouched (neither replaced nor additively boosted); blocks in between are
linearly blended between the two, to avoid a hard seam at the confidence
threshold's block boundaries.

Run via scripts/experiments/run_experiment_batch.py with
params={"log_gabor_mode": "replace_confidence_gated"}, not directly.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src", "utils"))
from common import orientation_field, normalize_image, segment  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


_COHERENCE_FULL_GENTLE = 0.68
_COHERENCE_FULL_AGGRESSIVE = 0.50

_HOMOMORPHIC_GAMMA_HIGH_RANGE = (1.2, 2.0)
_DIFFUSION_ITERATIONS_RANGE = (8, 25)
_DIFFUSION_KAPPA_RANGE = (10.0, 25.0)
_LOG_GABOR_ADD_GAIN_RANGE = (0.8, 2.5)


def _aggressiveness_alpha(coherence_field, fg_mask_blocks):
    if fg_mask_blocks is not None and fg_mask_blocks.any():
        mean_coherence = float(coherence_field[fg_mask_blocks].mean())
    else:
        mean_coherence = float(coherence_field.mean())
    span = _COHERENCE_FULL_GENTLE - _COHERENCE_FULL_AGGRESSIVE
    alpha = (_COHERENCE_FULL_GENTLE - mean_coherence) / span
    return float(np.clip(alpha, 0.0, 1.0))


def _lerp(range_, alpha):
    gentle, aggressive = range_
    return gentle + alpha * (aggressive - gentle)


def _homomorphic_filter(image, cutoff=0.06, gamma_low=0.5, gamma_high=2.0, sharpness=1.0):
    img = image.astype(np.float64) + 1.0
    log_img = np.log(img)

    h, w = image.shape
    Fshift = np.fft.fftshift(np.fft.fft2(log_img))

    u = (np.arange(w) - w // 2) / w
    v = (np.arange(h) - h // 2) / h
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)

    H = (gamma_high - gamma_low) * (1 - np.exp(-sharpness * (D ** 2) / (cutoff ** 2 + 1e-12))) + gamma_low

    filtered = np.fft.ifft2(np.fft.ifftshift(Fshift * H))
    result = np.exp(np.real(filtered)) - 1.0

    lo, hi = np.percentile(result, [1, 99])
    if hi > lo:
        result = (result - lo) / (hi - lo) * 255.0
    else:
        result = result - result.min()
    return np.clip(result, 0, 255).astype(np.uint8)


def _coherence_diffusion(image, theta_field, coherence_field, iterations=15, dt=0.2, kappa=15.0,
                          confidence_ceiling=0.45):
    img = image.astype(np.float32)
    h, w = img.shape

    theta_px = cv2.resize(theta_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = cv2.resize(coherence_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = np.clip(coherence_px, 0.0, 1.0)

    confidence_px = np.clip(coherence_px / confidence_ceiling, 0.0, 1.0)

    cos_t = np.cos(theta_px)
    sin_t = np.sin(theta_px)

    for _ in range(iterations):
        Iy, Ix = np.gradient(img)

        g_along = Ix * cos_t + Iy * sin_t
        g_across = -Ix * sin_t + Iy * cos_t

        c_across = np.exp(-(g_across / kappa) ** 2)
        c_along = np.exp(-(g_along / (kappa * 4.0)) ** 2)

        c_along = coherence_px * c_along + (1 - coherence_px) * c_across

        d11 = confidence_px * (c_along * cos_t ** 2 + c_across * sin_t ** 2)
        d22 = confidence_px * (c_along * sin_t ** 2 + c_across * cos_t ** 2)
        d12 = confidence_px * (c_along - c_across) * cos_t * sin_t

        Fx = d11 * Ix + d12 * Iy
        Fy = d12 * Ix + d22 * Iy

        _, dFx_dx = np.gradient(Fx)
        dFy_dy, _ = np.gradient(Fy)

        img = img + dt * (dFx_dx + dFy_dy)

    return np.clip(img, 0, 255).astype(np.uint8)


def _estimate_ridge_frequency(image, theta_field, block=16, window_len=32, window_width=16,
                               min_period=3, max_period=25, default_freq=1.0 / 9.0):
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape
    freq_field = np.full_like(theta_field, default_freq, dtype=np.float64)

    half_len = window_len // 2
    half_width = window_width // 2
    xs = np.arange(-half_width, half_width)
    ys = np.arange(-half_len, half_len)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")

    for by in range(nby):
        for bx in range(nbx):
            theta = theta_field[by, bx]
            cy = by * block + block // 2
            cx = bx * block + block // 2

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            sample_x = cx + xv * (-sin_t) + yv * cos_t
            sample_y = cy + xv * cos_t + yv * sin_t

            if (sample_x.min() < 0 or sample_x.max() >= w - 1 or
                    sample_y.min() < 0 or sample_y.max() >= h - 1):
                continue

            x0 = np.floor(sample_x).astype(int)
            y0 = np.floor(sample_y).astype(int)
            fx = sample_x - x0
            fy = sample_y - y0
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            x0 = np.clip(x0, 0, w - 1)
            y0 = np.clip(y0, 0, h - 1)

            Ia, Ib = img[y0, x0], img[y0, x1]
            Ic, Id = img[y1, x0], img[y1, x1]
            sampled = (Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy) +
                       Ic * (1 - fx) * fy + Id * fx * fy)

            x_signature = sampled.mean(axis=0)

            peaks = [k for k in range(1, len(x_signature) - 1)
                     if x_signature[k] > x_signature[k - 1] and x_signature[k] >= x_signature[k + 1]]

            if len(peaks) >= 2:
                period = float(np.mean(np.diff(peaks)))
                if min_period <= period <= max_period:
                    freq_field[by, bx] = 1.0 / period

    return freq_field


def _log_gabor_filter_2d(shape, freq0, theta0, sigma_onf=0.65, sigma_theta_deg=20.0):
    rows, cols = shape
    u = (np.arange(cols) - cols // 2) / cols
    v = (np.arange(rows) - rows // 2) / rows
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)
    radius[rows // 2, cols // 2] = 1.0

    theta_grid = np.arctan2(V, U)

    log_radius_ratio = np.log(radius / freq0 + 1e-12)
    radial = np.exp(-(log_radius_ratio ** 2) / (2 * np.log(sigma_onf) ** 2))
    radial[rows // 2, cols // 2] = 0.0

    sigma_theta = np.radians(sigma_theta_deg)
    dtheta = theta_grid - theta0
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
    dtheta = np.minimum(np.abs(dtheta), np.abs(np.abs(dtheta) - np.pi))
    angular = np.exp(-(dtheta ** 2) / (2 * sigma_theta ** 2))

    H = radial * angular
    return np.fft.ifftshift(H)


def _log_gabor_enhance(image, theta_field, freq_field, fg_mask_blocks, block=16, window=40,
                        sigma_onf=0.75, sigma_theta_deg=25.0, add_gain=2.5,
                        field_blur_sigma=1.0, coherence_field=None,
                        confidence_ceiling=0.45,
                        mode="additive",
                        replace_confidence_high=0.7,
                        replace_confidence_low=0.3):
    """
    [EXPERIMENT 2] Added `mode`, `replace_confidence_high`,
    `replace_confidence_low`. mode="additive" (default) is byte-identical to
    the verified pipeline's behaviour. mode="replace_confidence_gated" is
    new: per foreground block, uses gain_confidence (the same FIFTH-revision
    per-block confidence used to throttle additive gain) to decide whether
    to REPLACE that block with its own bandpass response (rescaled to the
    block's local std in the diffused image) — confidence >=
    replace_confidence_high -> full replace; confidence <=
    replace_confidence_low -> block left as the diffused image, completely
    untouched (no replace, no additive boost either); in between, a linear
    blend of the two, to avoid a hard seam at the confidence threshold.
    Background blocks are unaffected by `mode` — they're already left
    untouched in both modes.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape

    cos2 = np.cos(2 * theta_field)
    sin2 = np.sin(2 * theta_field)
    cos2_s = cv2.GaussianBlur(cos2.astype(np.float32), (0, 0), field_blur_sigma)
    sin2_s = cv2.GaussianBlur(sin2.astype(np.float32), (0, 0), field_blur_sigma)
    theta_field = 0.5 * np.arctan2(sin2_s, cos2_s)
    freq_field = cv2.GaussianBlur(freq_field.astype(np.float32), (0, 0), field_blur_sigma).astype(np.float64)

    if coherence_field is not None:
        coherence_smoothed = cv2.GaussianBlur(
            np.clip(coherence_field, 0.0, 1.0).astype(np.float32), (0, 0), field_blur_sigma
        ).astype(np.float64)
        gain_confidence = np.clip(coherence_smoothed / confidence_ceiling, 0.0, 1.0)
    else:
        gain_confidence = np.ones_like(theta_field)

    pad = window // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    taper_1d = np.hanning(window)
    taper_1d = np.clip(taper_1d, 1e-3, None)
    taper = np.outer(taper_1d, taper_1d)

    accum = np.zeros_like(padded)
    weight = np.zeros_like(padded)

    for by in range(nby):
        for bx in range(nbx):
            theta = theta_field[by, bx]
            freq = freq_field[by, bx]
            if not np.isfinite(freq) or freq <= 0:
                freq = 1.0 / 9.0

            cy = by * block + block // 2 + pad
            cx = bx * block + block // 2 + pad
            y0, y1 = cy - pad, cy - pad + window
            x0, x1 = cx - pad, cx - pad + window
            win = padded[y0:y1, x0:x1]
            if win.shape != (window, window):
                continue

            is_fg = fg_mask_blocks is None or fg_mask_blocks[by, bx]
            if is_fg:
                H = _log_gabor_filter_2d((window, window), freq, theta,
                                          sigma_onf=sigma_onf, sigma_theta_deg=sigma_theta_deg)
                F = np.fft.fft2(win)
                filtered = np.real(np.fft.ifft2(F * H))

                if mode == "replace_confidence_gated":
                    conf = gain_confidence[by, bx]
                    win_std = win.std()
                    filt_std = filtered.std()
                    if filt_std > 1e-6:
                        replace_win = win.mean() + (filtered - filtered.mean()) * (win_std / filt_std)
                    else:
                        replace_win = win.copy()
                    replace_win = np.clip(replace_win, 0, 255)

                    if conf >= replace_confidence_high:
                        enhanced_win = replace_win
                    elif conf <= replace_confidence_low:
                        enhanced_win = win
                    else:
                        t = (conf - replace_confidence_low) / (replace_confidence_high - replace_confidence_low)
                        enhanced_win = t * replace_win + (1 - t) * win
                else:
                    # ADDITIVE (default, unchanged from verified pipeline)
                    block_gain = add_gain * gain_confidence[by, bx]
                    enhanced_win = win + block_gain * filtered
            else:
                enhanced_win = win

            accum[y0:y1, x0:x1] += enhanced_win * taper
            weight[y0:y1, x0:x1] += taper

    weight[weight == 0] = 1.0
    merged = (accum / weight)[pad:pad + h, pad:pad + w]
    return np.clip(merged, 0, 255).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image

    [EXPERIMENT 2] New optional params: log_gabor_mode (default "additive",
    matching the verified pipeline; pass "replace_confidence_gated" to
    activate the experiment), log_gabor_replace_confidence_high (default
    0.7), log_gabor_replace_confidence_low (default 0.3).
    """
    params = params or {}

    # Step 0a
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", 100.0),
        target_var=params.get("normalize_target_var", 1600.0),
    )

    # Step 0b
    fg_mask_blocks, _block_var = segment(normalized)

    # Step 0c
    _theta_probe, coherence_probe = orientation_field(normalized)
    alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)

    # Step 1
    contrast_enhanced_raw = _homomorphic_filter(
        normalized,
        cutoff=params.get("homomorphic_cutoff", 0.06),
        gamma_low=params.get("homomorphic_gamma_low", 0.5),
        gamma_high=params.get("homomorphic_gamma_high", _lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha)),
        sharpness=params.get("homomorphic_sharpness", 1.0),
    )

    # Step 1b (unchanged from verified pipeline — this experiment only
    # touches _log_gabor_enhance)
    h, w = normalized.shape
    fg_alpha = cv2.resize(fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), 8.0)
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
    contrast_enhanced = (
        fg_alpha * contrast_enhanced_raw.astype(np.float64)
        + (1 - fg_alpha) * normalized.astype(np.float64)
    )
    contrast_enhanced = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)

    # Step 2
    theta_field, coherence_field = orientation_field(contrast_enhanced)

    # Step 3
    diffusion_iterations = params.get("diffusion_iterations", int(round(_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha))))
    diffusion_dt = params.get("diffusion_dt", 0.2)
    diffusion_kappa = params.get("diffusion_kappa", _lerp(_DIFFUSION_KAPPA_RANGE, alpha))
    diffused = _coherence_diffusion(
        contrast_enhanced,
        theta_field,
        coherence_field,
        iterations=diffusion_iterations,
        dt=diffusion_dt,
        kappa=diffusion_kappa,
        confidence_ceiling=params.get("diffusion_confidence_ceiling", 0.45),
    )

    # Step 4a
    freq_field = _estimate_ridge_frequency(
        diffused,
        theta_field,
        window_len=params.get("freq_window_len", 32),
        window_width=params.get("freq_window_width", 16),
        min_period=params.get("freq_min_period", 3),
        max_period=params.get("freq_max_period", 25),
    )

    # Step 4b [EXPERIMENT 2]: mode + replace-confidence thresholds now
    # passed through; default mode is "additive" (unchanged behaviour).
    enhanced = _log_gabor_enhance(
        diffused,
        theta_field,
        freq_field,
        fg_mask_blocks,
        window=params.get("log_gabor_window", 40),
        sigma_onf=params.get("log_gabor_sigma_onf", 0.75),
        sigma_theta_deg=params.get("log_gabor_sigma_theta_deg", 25.0),
        add_gain=params.get("log_gabor_add_gain", _lerp(_LOG_GABOR_ADD_GAIN_RANGE, alpha)),
        coherence_field=coherence_field,
        confidence_ceiling=params.get("log_gabor_confidence_ceiling", 0.45),
        mode=params.get("log_gabor_mode", "additive"),
        replace_confidence_high=params.get("log_gabor_replace_confidence_high", 0.7),
        replace_confidence_low=params.get("log_gabor_replace_confidence_low", 0.3),
    )

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img, params={"log_gabor_mode": "replace_confidence_gated"})
        cv2.imwrite("pipeline_c_exp2_test_output.png", out)
        print("Saved pipeline_c_exp2_test_output.png")
