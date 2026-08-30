"""
Pipeline C — Member C: Coherence-Preserving, Multi-Filter Enhancement
(Homomorphic Filtering + Coherence Diffusion + 2D Log-Gabor Filtering)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)              -> Homomorphic filtering
    P2 — high random noise (DB3)                 -> Coherence diffusion
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                              -> 2D Log-Gabor filtering

Segmentation and block-wise normalisation were dropped (30 August 2026):
they targeted P5/P7, which are no longer in scope now that only P1/P2/P6
are being solved, and — since Otsu segmentation and Hong et al.
normalisation would otherwise have been called identically by all four
pipelines — keeping them would have violated the lecturer's requirement
that no technique repeat across pipelines. Exactly three primary,
independently citable techniques per pipeline (one per shared problem) is
sufficient.

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Oppenheim, Schafer, & Stockham (1968) — homomorphic filtering
    Perona & Malik (1990) — anisotropic / coherence-enhancing diffusion
    Field (1987) — Log-Gabor filtering
    Shams et al. (2023) — methodological inspiration (this is the group's own
                           Literature Review 2 — anchor your lit review here)
    Hong, Wan, & Jain (1998) — structure-tensor orientation field (a
                                supporting calculation, not one of the three
                                primary techniques — see module docstring
                                note above on why it doesn't count as a
                                repeated technique)

All four steps are implemented:
    1. Homomorphic filtering (P1 — contrast)
    2. Orientation field (supporting calculation; steers steps 3 & 4)
    3. Coherence-enhancing anisotropic diffusion (P2 — noise)
    4. Ridge-frequency estimation + 2D Log-Gabor filtering (P6 — orientation)

Run this file directly (`python pipeline_c.py`) to sanity-check it against
one test image before running it over the full dataset.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import orientation_field  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def _homomorphic_filter(image, cutoff=0.06, gamma_low=0.5, gamma_high=2.0, sharpness=1.0):
    """
    Step 1: homomorphic filtering (Oppenheim, Schafer, & Stockham, 1968).

    Models an image as illumination(x,y) * reflectance(x,y), where
    illumination varies slowly (low spatial frequency) and reflectance
    carries the fine ridge/valley detail (high spatial frequency). Taking a
    log turns that product into a sum, which a single frequency-domain
    filter can then split apart: attenuate the low-frequency illumination
    term (gamma_low < 1) while boosting the high-frequency reflectance term
    (gamma_high > 1), then undo the log with exp().

    This directly targets P1 (DB3's systematically low, narrow-SD global
    contrast — Section 3.2) and, as a side effect, P3 (uneven illumination),
    since both are driven by the same low-frequency component this filter
    suppresses. It runs first in the pipeline (before orientation estimation
    in step 2) so the downstream diffusion and Log-Gabor steps are steered
    using a contrast-corrected image rather than DB3's original flat one.
    """
    img = image.astype(np.float64) + 1.0  # +1 avoids log(0) on pure-black pixels
    log_img = np.log(img)

    h, w = image.shape
    Fshift = np.fft.fftshift(np.fft.fft2(log_img))

    u = (np.arange(w) - w // 2) / w
    v = (np.arange(h) - h // 2) / h
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)

    # High-pass emphasis transfer function: ~gamma_low near DC (attenuate
    # illumination), rising smoothly to ~gamma_high at high frequencies
    # (boost reflectance/detail); `cutoff` sets where the transition sits,
    # `sharpness` how quickly it happens.
    H = (gamma_high - gamma_low) * (1 - np.exp(-sharpness * (D ** 2) / (cutoff ** 2 + 1e-12))) + gamma_low

    filtered = np.fft.ifft2(np.fft.ifftshift(Fshift * H))
    result = np.exp(np.real(filtered)) - 1.0

    # Rescale to 0-255 using the 1st/99th percentile, not the true min/max:
    # without a prior normalisation step, a handful of outlier pixels (e.g.
    # DB1's bright platen corners) can otherwise dominate a min/max stretch
    # and crush the entire ridge/valley structure into a narrow dark band —
    # this happened in practice on DB1 once block-wise normalisation was
    # dropped from the pipeline. Same robust-percentile idea already used
    # for Michelson contrast in Section 2 of the dataset analysis.
    lo, hi = np.percentile(result, [1, 99])
    if hi > lo:
        result = (result - lo) / (hi - lo) * 255.0
    else:
        result = result - result.min()
    return np.clip(result, 0, 255).astype(np.uint8)


def _coherence_diffusion(image, theta_field, coherence_field, iterations=15, dt=0.2, kappa=15.0):
    """
    Coherence-enhancing anisotropic diffusion (Perona & Malik, 1990), steered
    by the local ridge orientation from Step 2.

    At every pixel we build a 2x2 diffusion tensor whose two eigen-directions
    are the local ridge direction (theta_field) and the direction
    perpendicular to it. The two matching eigenvalues (c_along, c_across)
    control how freely intensity is allowed to blur in each direction:
        - along the ridge:  smooth A LOT. A real ridge is fairly uniform
          along its own length, so blurring along it removes noise without
          destroying real structure.
        - across the ridge: smooth CAUTIOUSLY, using a Perona-Malik
          conductance function that backs off wherever the cross-ridge
          gradient is large (i.e. a real ridge/valley boundary), so that
          boundary is preserved rather than blurred away.
    Wherever coherence_field says the orientation estimate is unreliable, the
    tensor is blended back toward isotropic diffusion, so we never smooth
    confidently along a direction we can't actually trust.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    # theta_field / coherence_field are one value per 16x16 block (see
    # orientation_field in common.py) — upsample to one value per pixel so
    # every pixel is steered by its local block's orientation.
    theta_px = cv2.resize(theta_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = cv2.resize(coherence_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = np.clip(coherence_px, 0.0, 1.0)

    cos_t = np.cos(theta_px)
    sin_t = np.sin(theta_px)

    for _ in range(iterations):
        Iy, Ix = np.gradient(img)  # np.gradient returns (d/d(row), d/d(col)) = (dI/dy, dI/dx)

        g_along = Ix * cos_t + Iy * sin_t     # gradient measured along the ridge direction
        g_across = -Ix * sin_t + Iy * cos_t   # gradient measured across the ridge direction

        # Perona-Malik conductance: c(x) = exp(-(x/kappa)^2).
        # small gradient -> conductance near 1 (smooth freely)
        # large gradient -> conductance near 0 (treat as a real edge, stop)
        c_across = np.exp(-(g_across / kappa) ** 2)
        c_along = np.exp(-(g_along / (kappa * 4.0)) ** 2)  # much more tolerant: keep smoothing along the ridge

        # low coherence = orientation estimate is shaky here -> fall back
        # toward isotropic (c_along ~= c_across) instead of trusting it
        c_along = coherence_px * c_along + (1 - coherence_px) * c_across

        # 2x2 diffusion tensor per pixel, built from the two eigenvalues
        # (c_along, c_across) and eigenvectors (ridge direction, perpendicular)
        d11 = c_along * cos_t ** 2 + c_across * sin_t ** 2
        d22 = c_along * sin_t ** 2 + c_across * cos_t ** 2
        d12 = (c_along - c_across) * cos_t * sin_t

        Fx = d11 * Ix + d12 * Iy
        Fy = d12 * Ix + d22 * Iy

        _, dFx_dx = np.gradient(Fx)
        dFy_dy, _ = np.gradient(Fy)

        img = img + dt * (dFx_dx + dFy_dy)

    return np.clip(img, 0, 255).astype(np.uint8)


def _estimate_ridge_frequency(image, theta_field, block=16, window_len=32, window_width=16,
                               min_period=3, max_period=25, default_freq=1.0 / 9.0):
    """
    Local ridge-frequency estimation, following the x-signature procedure in
    Hong, Wan, & Jain (1998).

    For each block, an oriented window aligned with the block's ridge
    direction (theta_field) is sampled: `window_width` parallel lines run
    ALONG the ridge, each `window_len` samples long, and are averaged pixel-
    for-pixel to build a 1D "x-signature" that profiles intensity ACROSS the
    ridges/valleys (averaging along the ridge cancels most of the noise a
    single scanline would carry). The average spacing between consecutive
    peaks in that profile is the local ridge period in pixels; frequency is
    1/period.

    Blocks whose x-signature doesn't yield at least two peaks with a
    plausible fingerprint period (e.g. background, or a low-coherence block
    where "the ridge direction" isn't meaningful) fall back to
    `default_freq` rather than reporting a wild estimate — a bad frequency
    would otherwise mistune the Log-Gabor filter for that whole block in
    step 5.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape
    freq_field = np.full_like(theta_field, default_freq, dtype=np.float64)

    half_len = window_len // 2
    half_width = window_width // 2
    xs = np.arange(-half_width, half_width)
    ys = np.arange(-half_len, half_len)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")  # shape (window_len, window_width)

    for by in range(nby):
        for bx in range(nbx):
            theta = theta_field[by, bx]
            cy = by * block + block // 2
            cx = bx * block + block // 2

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # xv is the across-ridge offset, yv the along-ridge offset;
            # rotate that ridge-aligned frame into image (x, y) coordinates.
            sample_x = cx + xv * (-sin_t) + yv * cos_t
            sample_y = cy + xv * cos_t + yv * sin_t

            if (sample_x.min() < 0 or sample_x.max() >= w - 1 or
                    sample_y.min() < 0 or sample_y.max() >= h - 1):
                continue  # oriented window falls off the image; keep default_freq

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

            x_signature = sampled.mean(axis=0)  # average along the ridge -> profile across ridges

            peaks = [k for k in range(1, len(x_signature) - 1)
                     if x_signature[k] > x_signature[k - 1] and x_signature[k] >= x_signature[k + 1]]

            if len(peaks) >= 2:
                period = float(np.mean(np.diff(peaks)))
                if min_period <= period <= max_period:
                    freq_field[by, bx] = 1.0 / period
                # else: implausible spacing for a real ridge -> keep default_freq

    return freq_field


def _log_gabor_filter_2d(shape, freq0, theta0, sigma_onf=0.65, sigma_theta_deg=20.0):
    """
    Builds one 2D Log-Gabor transfer function H(u,v) over a frequency-domain
    grid of the given shape, tuned to a single local ridge frequency (freq0,
    cycles/pixel, from step 4) and orientation (theta0, radians, from step 2)
    — Field (1987).

    A Log-Gabor filter is defined directly as a Gaussian in LOG-frequency
    space rather than linear frequency space (the ordinary Gabor filters
    Pipeline B already uses). That avoids the DC bias / limited bandwidth a
    linear-domain Gaussian carries, so the same filter shape stays well
    behaved whether it's tuned to DB3's tight high-frequency ridges or DB4's
    coarser ones — useful here since Pipeline C's own problem set (P2, P6)
    spans both.
    """
    rows, cols = shape
    u = (np.arange(cols) - cols // 2) / cols
    v = (np.arange(rows) - rows // 2) / rows
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)
    radius[rows // 2, cols // 2] = 1.0  # placeholder so log() below doesn't hit log(0) at DC

    theta_grid = np.arctan2(V, U)

    # radial component: Gaussian over log(radius/freq0)
    log_radius_ratio = np.log(radius / freq0 + 1e-12)
    radial = np.exp(-(log_radius_ratio ** 2) / (2 * np.log(sigma_onf) ** 2))
    radial[rows // 2, cols // 2] = 0.0  # zero the DC term explicitly

    # angular component: Gaussian around theta0. Ridge orientation is a
    # direction, not a vector (theta0 and theta0 + pi describe the same
    # ridge), so both lobes of the frequency plane 180 degrees apart get
    # folded onto the same angular response.
    sigma_theta = np.radians(sigma_theta_deg)
    dtheta = theta_grid - theta0
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap to [-pi, pi]
    dtheta = np.minimum(np.abs(dtheta), np.abs(np.abs(dtheta) - np.pi))
    angular = np.exp(-(dtheta ** 2) / (2 * sigma_theta ** 2))

    H = radial * angular
    return np.fft.ifftshift(H)  # match np.fft.fft2's unshifted (DC-at-corner) layout


def _log_gabor_enhance(image, theta_field, freq_field, block=16, window=40,
                        sigma_onf=0.5, sigma_theta_deg=12.0):
    """
    Step 4: block-wise 2D Log-Gabor filtering (Field, 1987), each block
    tuned to its own local orientation (theta_field, step 2) and ridge
    frequency (freq_field, step 4) — following the general locally-tuned,
    block-wise filtering strategy in Shams et al. (2023), built here from
    Log-Gabor transfer functions rather than Hong et al.'s (1998) spatial
    Gabor kernels (that spatial-domain approach is Pipeline B's technique;
    this frequency-domain one is Pipeline C's point of comparison against it
    for the same P2/P6 problems).

    Each block is filtered inside a larger `window`-sized neighbourhood, not
    just its own `block` pixels, so the FFT has enough context to represent
    the tuned frequency cleanly. Since `window` is twice `block`, neighbouring
    blocks' windows overlap; rather than hard-cutting each to its own centre
    region (which leaves a visible seam at every block boundary — adjacent
    blocks are tuned to slightly different theta/freq, so their filtered
    outputs don't land on the same scale), every window is blended into the
    output with a 2D Hanning taper via overlap-add, weighted-averaging the
    overlapping contributions instead of hard-cutting them.

    Log-Gabor filtering also removes the DC term, so a block's raw filtered
    values carry no fixed relationship to the input's 0-255 range; each
    block is rescaled to match the mean/std of its OWN local input window
    before blending, rather than one global min-max stretch at the end
    (which was the other source of visible per-block banding).
    """
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape

    pad = window // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # 2D raised-cosine taper: 0 at a window's own edges, 1 at its centre, so
    # overlap-add blends neighbouring windows smoothly instead of seaming.
    taper_1d = np.hanning(window)
    taper_1d = np.clip(taper_1d, 1e-3, None)  # keep every contribution weighted, never exactly zero
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
                continue  # shouldn't happen given the reflect-padding, but be defensive

            H = _log_gabor_filter_2d((window, window), freq, theta,
                                      sigma_onf=sigma_onf, sigma_theta_deg=sigma_theta_deg)
            F = np.fft.fft2(win)
            filtered = np.real(np.fft.ifft2(F * H))

            local_mean, local_std = win.mean(), win.std() + 1e-6
            f_mean, f_std = filtered.mean(), filtered.std() + 1e-6
            filtered = (filtered - f_mean) / f_std * local_std + local_mean

            accum[y0:y1, x0:x1] += filtered * taper
            weight[y0:y1, x0:x1] += taper

    weight[weight == 0] = 1.0
    merged = (accum / weight)[pad:pad + h, pad:pad + w]
    return np.clip(merged, 0, 255).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 1: homomorphic filtering (Oppenheim, Schafer, & Stockham, 1968)
    # — this pipeline's technique for P1 (low global contrast, systematic
    # to DB3). Runs directly on the raw image (segmentation and block-wise
    # normalisation were dropped — see module docstring: they targeted
    # P5/P7, which are out of scope now that only P1/P2/P6 are shared
    # across pipelines, and keeping them would have meant all four
    # pipelines calling the identical Otsu/Hong et al. steps, which the
    # lecturer's no-repeated-technique requirement rules out).
    contrast_enhanced = _homomorphic_filter(
        image,
        cutoff=params.get("homomorphic_cutoff", 0.06),
        gamma_low=params.get("homomorphic_gamma_low", 0.5),
        gamma_high=params.get("homomorphic_gamma_high", 2.0),
        sharpness=params.get("homomorphic_sharpness", 1.0),
    )

    # Step 2: ridge orientation field estimation (Hong et al., 1998) — a
    # supporting calculation, not one of this pipeline's three primary
    # techniques (see module docstring). Computed on the contrast-corrected
    # image so steps 3-4 are steered by a cleaner orientation estimate than
    # DB3's original flat contrast would give.
    theta_field, coherence_field = orientation_field(contrast_enhanced)

    # Step 3: coherence-enhancing anisotropic diffusion (Perona & Malik,
    # 1990), steered by theta_field — this pipeline's technique for P2
    # (high random noise, systematic to DB3). Smooths MORE along the ridge
    # direction and LESS across it, preserving the ridge/valley boundary
    # itself.
    diffusion_iterations = params.get("diffusion_iterations", 15)
    diffusion_dt = params.get("diffusion_dt", 0.2)
    diffusion_kappa = params.get("diffusion_kappa", 15.0)
    diffused = _coherence_diffusion(
        contrast_enhanced,
        theta_field,
        coherence_field,
        iterations=diffusion_iterations,
        dt=diffusion_dt,
        kappa=diffusion_kappa,
    )

    # Step 4a: local ridge-frequency estimation (supporting calculation for
    # step 4b; Hong, Wan, & Jain, 1998 x-signature method)
    freq_field = _estimate_ridge_frequency(
        diffused,
        theta_field,
        window_len=params.get("freq_window_len", 32),
        window_width=params.get("freq_window_width", 16),
        min_period=params.get("freq_min_period", 3),
        max_period=params.get("freq_max_period", 25),
    )

    # Step 4b: 2D Log-Gabor filtering (Field, 1987; Shams et al., 2023),
    # tuned per block by theta_field (step 2) and freq_field (step 4a) —
    # this pipeline's technique for P6 (weak/inconsistent ridge orientation,
    # DB4/DB3).
    enhanced = _log_gabor_enhance(
        diffused,
        theta_field,
        freq_field,
        window=params.get("log_gabor_window", 40),
        sigma_onf=params.get("log_gabor_sigma_onf", 0.5),
        sigma_theta_deg=params.get("log_gabor_sigma_theta_deg", 12.0),
    )

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_c_test_output.png", out)
        print("Saved pipeline_c_test_output.png — open it and compare against the input image.")
