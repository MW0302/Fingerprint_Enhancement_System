"""
Pipeline C — Member C: Coherence-Preserving Denoising
(Coherence Diffusion + 2D Log-Gabor Filtering)

Targets (see Dataset_Problem_Analysis_and_Revised_Pipelines.docx, Section 6):
    P2 — high random noise (DB3)
    P6 — weak/inconsistent ridge orientation (DB4, DB3)
    P4 — blurred/soft ridges (secondary, cross-check against Pipeline D)

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Perona & Malik (1990) — anisotropic / coherence-enhancing diffusion
    Field (1987) — Log-Gabor filtering
    Shams et al. (2023) — methodological inspiration (this is the group's own
                           Literature Review 2 — anchor your lit review here)
    Hong, Wan, & Jain (1998) — shared segmentation + orientation field steps

All 5 steps are implemented: (1) segmentation, (2) orientation field (both
shared), (3) coherence-enhancing anisotropic diffusion, (4) local
ridge-frequency estimation, (5) 2D Log-Gabor filtering.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import segment, orientation_field  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import (
    distance_transform_edt,
    binary_fill_holes,
    binary_closing,
    binary_dilation,
    label,
)


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


def _fill_nan_nearest(field):
    """Fills NaN entries in `field` with the value of their nearest non-NaN
    neighbor, so freq_field has no gaps for Step 5's Log-Gabor filter."""
    invalid = np.isnan(field)
    if not invalid.any():
        return field
    if invalid.all():
        return np.full_like(field, 1.0 / 9.0)
    nearest_idx = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return field[tuple(nearest_idx)]


def _estimate_ridge_frequency(image, theta_field, block=16, window=32,
                               min_freq=1.0 / 25.0, max_freq=1.0 / 3.0):
    """Local ridge-frequency estimation via the X-signature method (Hong,
    Wan, & Jain, 1998): for each block, rotate a window around it so the
    local ridge direction becomes vertical, sum intensities column-wise to
    get a 1D "X-signature" (one peak per ridge), and take the average
    spacing between consecutive peaks as the ridge period."""
    h, w = image.shape
    n_rows, n_cols = theta_field.shape
    half = window // 2
    img_f = image.astype(np.float64)

    freq_field = np.full((n_rows, n_cols), np.nan, dtype=np.float64)

    for bi in range(n_rows):
        for bj in range(n_cols):
            cy = bi * block + block // 2
            cx = bj * block + block // 2

            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
            pad_top, pad_bottom = max(0, -y0), max(0, y1 - h)
            pad_left, pad_right = max(0, -x0), max(0, x1 - w)
            ys0, ys1 = max(0, y0), min(h, y1)
            xs0, xs1 = max(0, x0), min(w, x1)
            patch = img_f[ys0:ys1, xs0:xs1]
            if pad_top or pad_bottom or pad_left or pad_right:
                if pad_top >= patch.shape[0] or pad_bottom >= patch.shape[0] or \
                        pad_left >= patch.shape[1] or pad_right >= patch.shape[1]:
                    continue  # too close to a corner to reflect-pad safely; leave NaN
                patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="reflect")

            if patch.shape != (window, window):
                continue

            # Rotate the window so the local ridge direction (theta_field)
            # becomes vertical, i.e. ridges run top-to-bottom.
            theta_deg = np.degrees(theta_field[bi, bj])
            rot_mat = cv2.getRotationMatrix2D((half, half), 90 - theta_deg, 1.0)
            rotated = cv2.warpAffine(patch, rot_mat, (window, window),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

            x_signature = rotated.sum(axis=0)  # column-wise sum -> 1D signal across ridges

            peaks, _ = find_peaks(x_signature)
            if len(peaks) >= 2:
                spacing = np.diff(peaks).mean()
                if spacing > 0:
                    freq_field[bi, bj] = 1.0 / spacing
            # else: ambiguous/background block — leave NaN, filled below

    freq_field = _fill_nan_nearest(freq_field)
    freq_field = np.clip(freq_field, min_freq, max_freq)
    return freq_field


def _build_log_gabor_kernel(theta_ridge, f0, kernel_size, bandwidth, angular_sigma):
    """Builds a real-valued 2D Log-Gabor (Field, 1987) convolution kernel for
    one specific orientation/frequency, by designing the filter in the
    frequency domain (as a radial log-frequency Gaussian x an angular
    Gaussian, with the radial term explicitly zeroed at rho==0 so the filter
    can never respond to DC/illumination) and taking its inverse FFT.

    theta_ridge is the SPATIAL ridge orientation (theta_field's convention:
    0=horizontal ridges, 90deg=vertical ridges). A set of parallel ridges at
    spatial angle theta_ridge has its Fourier-domain energy concentrated at
    frequency-domain angle theta_ridge + 90deg, not at theta_ridge itself —
    same 90-degree spatial-vs-gradient distinction as the orientation_field()
    fix in common.py, just showing up here as spatial-vs-frequency-domain
    instead. Confirmed directly: a kernel built with the angular Gaussian
    centered at theta_ridge (no correction) had ~1.5e-8x response to ridges
    actually at that spatial angle — matching the Gaussian's far-tail weight
    exactly 90 degrees off — while a kernel built at theta_ridge+90deg gives
    full-strength response, as it should.
    """
    theta0 = theta_ridge + np.pi / 2  # rotate ridge angle to its frequency-domain angle
    freqs = np.fft.fftshift(np.fft.fftfreq(kernel_size))
    U, V = np.meshgrid(freqs, freqs)  # U: x-frequency, V: y-frequency
    rho = np.sqrt(U ** 2 + V ** 2)
    phi = np.arctan2(V, U)
    dc_mask = rho == 0

    with np.errstate(divide="ignore", invalid="ignore"):
        radial = np.exp(-(np.log(rho / f0)) ** 2 / (2 * np.log(bandwidth) ** 2))
    radial[dc_mask] = 0.0  # guarantee zero DC / illumination response

    diff = phi - theta0
    diff = np.mod(diff + np.pi / 2, np.pi) - np.pi / 2  # ridge orientation is mod pi
    angular = np.exp(-(diff ** 2) / (2 * angular_sigma ** 2))

    mask = radial * angular
    kernel = np.real(np.fft.ifft2(np.fft.ifftshift(mask)))
    return np.fft.fftshift(kernel)


def _log_gabor_filter(image, theta_field, freq_field, block=16,
                       bandwidth=0.5, angular_sigma=np.pi / 12):
    """
    2D Log-Gabor filtering (Field, 1987), applied as a per-block real-space
    convolution: for each orientation/frequency block, build its own Log-Gabor
    kernel (see _build_log_gabor_kernel) sized to its local ridge spacing,
    convolve a local neighborhood of the image with it (using real
    neighboring pixels + reflect border handling, not a zero/Hann-tapered
    window), and blend overlapping blocks together with a smooth (Hanning)
    spatial weight to avoid hard block-boundary seams.

    An earlier version of this filter tiled the image into small (32x32)
    windows and did the whole filter in the frequency domain per window
    (fft2 -> multiply by mask -> ifft2), matching how STFT-based filtering
    is often described. That version had two compounding problems, both
    confirmed by testing on synthetic ridge images against the known-true
    pattern: (1) at only ~3 periods per window, the narrow Log-Gabor passband
    doesn't fit; its real-space impulse response is wider than the window,
    so the implicit circular-FFT convolution wraps around and rings, and (2)
    even after fixing that with weighted overlap-add, the filter's angular
    tolerance (was 30 degrees) let through enough off-orientation content
    that overlapping blocks' independent reconstructions didn't agree,
    producing a fine checkerboard/moire texture instead of clean ridges, and
    LOWERING measured orientation coherence rather than raising it. Building
    a real per-block kernel and convolving normally (no circular wraparound)
    plus a tighter angular_sigma (15 degrees default) fixed both: verified on
    synthetic noisy ridge images to raise post-filter orientation coherence
    back above the noisy input's in most tested cases.
    """
    h, w = image.shape
    n_block_rows, n_block_cols = theta_field.shape

    win = block * 3     # blending footprint per block (block itself + margin)
    stride = block
    max_kernel = 41
    pad = win + max_kernel // 2  # room for both the blend window and the widest kernel's support
    padded = cv2.copyMakeBorder(image.astype(np.float64), pad, pad, pad, pad, cv2.BORDER_REFLECT)

    blend = np.outer(np.hanning(win), np.hanning(win))
    out_accum = np.zeros_like(padded)
    weight_accum = np.zeros_like(padded)

    for bi in range(n_block_rows):
        for bj in range(n_block_cols):
            theta0 = float(theta_field[bi, bj])
            f0 = float(freq_field[bi, bj])

            # kernel wide enough to cover ~4 ridge periods, odd-sized, capped
            kernel_size = int(np.clip(round(4.0 / f0), 15, max_kernel))
            if kernel_size % 2 == 0:
                kernel_size += 1
            kpad = kernel_size // 2
            kernel = _build_log_gabor_kernel(theta0, f0, kernel_size, bandwidth, angular_sigma)

            cy = pad + bi * block + block // 2
            cx = pad + bj * block + block // 2
            y0, x0 = cy - win // 2, cx - win // 2
            y1, x1 = y0 + win, x0 + win

            patch = padded[y0 - kpad:y1 + kpad, x0 - kpad:x1 + kpad]
            filtered = cv2.filter2D(patch, -1, kernel, borderType=cv2.BORDER_REFLECT)
            filtered = filtered[kpad:kpad + win, kpad:kpad + win]

            out_accum[y0:y1, x0:x1] += filtered * blend
            weight_accum[y0:y1, x0:x1] += blend

    weight_accum[weight_accum == 0] = 1e-8
    result = out_accum / weight_accum
    result = result[pad:pad + h, pad:pad + w]

    # Rescale to 0-255 using the 1st/99th percentile, not the raw min/max.
    # On noisier images (DB3_B) a handful of outlier pixels can swing the
    # true min/max wide, which then crushes the other ~98% of real ridge
    # contrast into a narrow band (observed on real data: p1-p99 spanning
    # only ~40 of 255 levels, std~8, vs ~100 levels/std~18 on a clean image)
    # — low enough that NFIQ2 couldn't find a large-enough fingerprint area.
    lo, hi = np.percentile(result, [1.0, 99.0])
    if hi > lo:
        result = (result - lo) / (hi - lo) * 255.0
    else:
        result = result - result.min()
        if result.max() > 0:
            result = result / result.max() * 255.0
    return np.clip(result, 0, 255).astype(np.uint8)


def _clean_fg_mask(fg_mask_blocks):
    """Cleans up segment()'s raw per-block foreground mask before it's used
    to mask Step 5's output. Visualizing fg_mask_blocks over real images
    (overlaying it on the raw scan) showed segment()'s per-block Otsu
    variance classification failing in two ways, not just the enclosed-hole
    case this function originally only handled:

      1. Jagged notches biting into the real fingerprint area from the
         edges — a moderately-pressed ridge region right at the print's own
         boundary often has just-below-threshold variance, so segment()
         crops a chunk of real ridge detail off as "background" (confirmed
         visually on DB1_B/DB4_B: the enhanced output had big gray bites
         taken out of visibly-real ridge texture, not just a coarse but
         otherwise-reasonable outline).
      2. Small isolated false-foreground or false-background specks
         scattered away from the real print (sensor dust, paper grain).

    Fixed with a standard sequence: morphological closing (bridges small-
    to-medium notches/gaps), fill any now-enclosed holes (the over-inked-
    core case from before), keep only the largest connected component
    (drops stray specks disconnected from the real print), then a small
    dilation (recovers the marginal blocks along the print's true edge that
    Otsu was systematically too conservative about). This is post-
    processing on segment()'s OUTPUT only — segment() itself, shared by
    every pipeline, is untouched.

    Not a full fix: a segment() failure caused by an image-wide contrast
    gradient (one whole side of the print reading as lower-variance than
    the Otsu cutoff) can leave a large contiguous region misclassified that
    no reasonably-sized closing/dilation fully recovers — confirmed on one
    DB2_B image where roughly a third of the print stayed excluded even
    after this cleanup. That's a genuine segment() limitation to flag, not
    something this function claims to fully solve.
    """
    struct = np.ones((5, 5))
    closed = binary_closing(fg_mask_blocks, structure=struct, iterations=1)
    filled = binary_fill_holes(closed)

    labeled, n_components = label(filled)
    if n_components > 0:
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0  # background label never counts
        filled = labeled == sizes.argmax()

    return binary_dilation(filled, structure=np.ones((3, 3)), iterations=1)


def _mask_background(enhanced, image, fg_mask_blocks, block=16, fill="gray"):
    """Restores non-fingerprint (background) blocks to either a flat
    mid-gray (fill="gray", default) or the original raw pixels
    (fill="original"), so Step 5's Log-Gabor output — which assumes real
    ridge structure to filter — is never shown over background, only over
    the segmented fingerprint area from Step 1 (cleaned up by
    _clean_fg_mask first — see its docstring for why segment()'s raw
    per-block output isn't used directly).

    Tested both fill options on real images: leaving the raw background in
    actually made NFIQ2 score WORSE than doing no masking at all (e.g.
    53->19 on one DB3_B image) — NFIQ2 seems to read the real (noisy,
    textured) background as low-quality fingerprint-like content and lets
    it drag the whole score down. A flat gray background reads
    unambiguously as "not fingerprint" and consistently scored as high or
    higher than the unmasked baseline in the same tests, so it's the default.
    """
    fg_mask_blocks = _clean_fg_mask(fg_mask_blocks)

    h, w = enhanced.shape
    n_block_rows, n_block_cols = fg_mask_blocks.shape
    h2, w2 = n_block_rows * block, n_block_cols * block

    # upsample the per-block mask to pixel resolution; any leftover border
    # thinner than one block (past h2/w2) was never covered by segmentation
    # in the first place, so it's left as background (False) by default.
    fg_px = np.zeros((h, w), dtype=bool)
    fg_px[:h2, :w2] = np.repeat(np.repeat(fg_mask_blocks, block, axis=0), block, axis=1)

    background = image if fill == "original" else np.full_like(enhanced, 128)
    return np.where(fg_px, enhanced, background).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 1: segmentation (shared)
    fg_mask_blocks, block_var = segment(image)

    # Step 2: ridge orientation field estimation (shared; Hong et al., 1998)
    theta_field, coherence_field = orientation_field(image)

    # Step 3: coherence-enhancing anisotropic diffusion (Perona & Malik, 1990),
    # steered by theta_field — smooths more along the ridge direction and
    # less across it, which preserves ridge structure while reducing noise.
    diffusion_iterations = params.get("diffusion_iterations", 15)
    diffusion_dt = params.get("diffusion_dt", 0.2)
    diffusion_kappa = params.get("diffusion_kappa", 15.0)
    diffused = _coherence_diffusion(
        image,
        theta_field,
        coherence_field,
        iterations=diffusion_iterations,
        dt=diffusion_dt,
        kappa=diffusion_kappa,
    )

    # Step 4: local ridge-frequency estimation (supporting step for step 5)
    freq_field = _estimate_ridge_frequency(diffused, theta_field)

    # Step 5: 2D Log-Gabor filtering (Field, 1987; Shams et al., 2023)
    log_gabor_bandwidth = params.get("log_gabor_bandwidth", 0.5)
    log_gabor_angular_sigma = params.get("log_gabor_angular_sigma", np.pi / 12)
    enhanced = _log_gabor_filter(
        diffused,
        theta_field,
        freq_field,
        bandwidth=log_gabor_bandwidth,
        angular_sigma=log_gabor_angular_sigma,
    )

    # Mask background back in: fg_mask_blocks (Step 1) marks which blocks are
    # actual fingerprint vs background — Log-Gabor filtering only makes sense
    # over real ridge structure, so background blocks are restored to the
    # original pixels (or a flat gray) instead of showing filtered noise.
    background_fill = params.get("background_fill", "gray")
    enhanced = _mask_background(enhanced, image, fg_mask_blocks, fill=background_fill)

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
