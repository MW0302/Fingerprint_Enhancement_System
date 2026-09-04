"""
Read-only, per-stage adapters for the four pipelines -- give the dashboard
Stage 1 / Stage 2 / Stage 3 intermediate images (plus per-stage wall-clock
timing) without modifying any pipeline_*.py file. Every adapter below only
CALLS each pipeline's own already-existing functions, in the same order and
with the same parameters enhance() itself uses (confirmed by reading each
enhance() end to end, not assumed) -- none of the actual enhancement
algorithms are reimplemented here.

Per pipeline:
    Pipeline A -- src/pipeline_a/pipeline_a.py already exposes
        _clahe_contrast(), _bilateral_denoise(), _oriented_gabor_filter()
        as separately-callable module functions. This adapter just calls
        them in enhance()'s own order.
    Pipeline B -- Stage 1 exposes _wavelet_contrast(). Stages 2 and 3 are
        still TODO placeholders, so the adapter shows the real Stage 1 only
        and reports that the later cumulative stages are not yet available.
    Pipeline C -- its internals are intentionally NOT exposed (validated/
        tuned, not to be touched -- see src/pipeline_c/pipeline_c.py's own
        module docstring). scripts/pipeline_c_ablation.py already solves
        exactly this problem as a read-only external wrapper; this adapter
        imports and reuses its ablate() function rather than duplicating
        that logic here.
    Pipeline D -- src/pipeline_d/pipeline_d.py already exposes
        _fft_high_frequency_emphasis(), _frequency_domain_wiener_filter(),
        _stft_orientation_frequency_reconstruct() the same way Pipeline A
        does. No orientation_field() step (D's own docstring: STFT
        estimates orientation/frequency jointly on its own).

Every get_stages() below returns a dict:
    {
        "available": bool,
        "message": str,            # explanation when available=False
        "stages": [                # in order, only present when available
            {"name": ..., "image": np.ndarray uint8, "time_ms": float},
            ...
        ],
    }
"""

import os
import sys
import time

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_a"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_b"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_d"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))

from common import normalize_image, segment, orientation_field  # noqa: E402
from common import DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR  # noqa: E402

import pipeline_a  # noqa: E402
import pipeline_b  # noqa: E402
import pipeline_d  # noqa: E402


def _timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def get_stages_pipeline_a(image, params=None):
    params = params or {}
    stages = []

    normalized, t0 = _timed(
        normalize_image, image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    (fg_mask_blocks, _block_var), t_seg = _timed(segment, normalized)

    stage1, t1 = _timed(
        pipeline_a._clahe_contrast, normalized,
        clip_limit=params.get("clahe_clip", 2.0),
        grid_size=params.get("clahe_grid", 8),
    )
    stages.append({"name": "Stage 1 (P1 — CLAHE contrast)", "image": stage1, "time_ms": t0 + t_seg + t1})

    stage2, t2 = _timed(
        pipeline_a._bilateral_denoise, stage1,
        diameter=params.get("bilateral_diameter", 5),
        sigma_color=params.get("bilateral_sigma_color", 35.0),
        sigma_space=params.get("bilateral_sigma_space", 5.0),
    )
    stages.append({"name": "Stage 2 (P1+P2 — + bilateral denoise)", "image": stage2, "time_ms": t2})

    (theta_field, coherence_field), t_orient = _timed(orientation_field, stage2)
    stage3, t3 = _timed(
        pipeline_a._oriented_gabor_filter, stage2, theta_field, coherence_field, fg_mask_blocks,
        kernel_size=params.get("gabor_kernel_size", 17),
        sigma=params.get("gabor_sigma", 4.0),
        wavelength=params.get("gabor_wavelength", 8.0),
        gamma=params.get("gabor_gamma", 0.5),
        strength=params.get("gabor_strength", 0.7),
        orientation_bins=params.get("gabor_orientation_bins", 16),
        coherence_floor=params.get("gabor_coherence_floor", 0.2),
    )
    stages.append({"name": "Stage 3 (P1+P2+P6 — + oriented Gabor)", "image": stage3, "time_ms": t_orient + t3})

    return {"available": True, "message": "", "stages": stages}


def get_stages_pipeline_b(image, params=None):
    params = params or {}
    normalized, t0 = _timed(
        normalize_image,
        image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    (fg_mask_blocks, _block_var), t_seg = _timed(segment, normalized)
    stage1, t1 = _timed(
        pipeline_b._wavelet_contrast,
        normalized,
        fg_mask_blocks,
        wavelet=params.get("wavelet", "db4"),
        level=params.get("wavelet_level", 3),
        coarse_gain=params.get("wavelet_coarse_gain", 1.60),
        fine_gain=params.get("wavelet_fine_gain", 1.00),
        coefficient_floor_percentile=params.get(
            "wavelet_coefficient_floor_percentile", 25.0
        ),
        blend=params.get("wavelet_contrast_blend", 1.0),
    )
    return {
        "available": True,
        "message": (
            "Pipeline B Stage 1 is available. Stage 2 wavelet denoising and "
            "Stage 3 orientation-steered morphology remain TODO."
        ),
        "stages": [
            {
                "name": "Stage 1 (P1 — wavelet detail contrast)",
                "image": stage1,
                "time_ms": t0 + t_seg + t1,
            }
        ],
    }


def get_stages_pipeline_c(image, params=None):
    # scripts/pipeline_c_ablation.py's ablate() does not currently accept a
    # params override (it always uses pipeline_c's own alpha-adaptive
    # defaults) -- flagged rather than silently ignoring any params passed
    # in, since Pipeline C's Advanced Parameters section (if enabled) would
    # otherwise appear to do nothing for the stage view specifically.
    import pipeline_c_ablation  # local import: only needed for this pipeline

    (result, t_total) = _timed(pipeline_c_ablation.ablate, image)
    stages = [
        {"name": "Stage 1 (P1 — homomorphic contrast)", "image": result["stage1"], "time_ms": None},
        {"name": "Stage 2 (P1+P2 — + coherence diffusion)", "image": result["stage2"], "time_ms": None},
        {"name": "Stage 3 (P1+P2+P6 — + Log-Gabor)", "image": result["stage3"], "time_ms": None},
    ]
    # ablate() doesn't expose intermediate timing, only total wall-clock for
    # all three stages combined -- reported once rather than guessing a
    # split that wasn't actually measured.
    return {
        "available": True,
        "message": f"Pipeline C's ablate() doesn't expose per-stage timing; total (all 3 stages) = {t_total:.0f} ms.",
        "stages": stages,
        "note_alpha": result.get("alpha"),
    }


def get_stages_pipeline_d(image, params=None):
    params = params or {}
    stages = []

    normalized, t0 = _timed(
        normalize_image, image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )
    (fg_mask_blocks, _block_var), t_seg = _timed(segment, normalized)

    stage1, t1 = _timed(
        pipeline_d._fft_high_frequency_emphasis, normalized,
        cutoff_ratio=params.get("fft_cutoff_ratio", 0.06),
        low_gain=params.get("fft_low_gain", 0.95),
        high_boost=params.get("fft_high_boost", 0.75),
        percentile_low=params.get("fft_percentile_low", 1.0),
        percentile_high=params.get("fft_percentile_high", 99.0),
        blend=params.get("fft_blend", 0.55),
    )
    stages.append({"name": "Stage 1 (P1 — FFT high-frequency emphasis)", "image": stage1, "time_ms": t0 + t_seg + t1})

    stage2, t2 = _timed(
        pipeline_d._frequency_domain_wiener_filter, stage1, fg_mask_blocks,
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
    stages.append({"name": "Stage 2 (P1+P2 — + Wiener denoise)", "image": stage2, "time_ms": t2})

    stage3, t3 = _timed(
        pipeline_d._stft_orientation_frequency_reconstruct, stage2, fg_mask_blocks,
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
    stages.append({"name": "Stage 3 (P1+P2+P6 — + STFT reconstruction)", "image": stage3, "time_ms": t3})

    return {"available": True, "message": "", "stages": stages}


_ADAPTERS = {
    "pipeline_a": get_stages_pipeline_a,
    "pipeline_b": get_stages_pipeline_b,
    "pipeline_c": get_stages_pipeline_c,
    "pipeline_d": get_stages_pipeline_d,
}


def get_stages(pipeline_key, image, params=None):
    """pipeline_key: one of 'pipeline_a'/'pipeline_b'/'pipeline_c'/'pipeline_d'."""
    adapter = _ADAPTERS.get(pipeline_key)
    if adapter is None:
        return {"available": False, "message": f"Unknown pipeline key: {pipeline_key}", "stages": []}
    try:
        return adapter(image, params=params)
    except Exception as exc:  # noqa: BLE001 -- surfaced to the UI, not swallowed
        return {
            "available": False,
            "message": f"Stage breakdown failed: {type(exc).__name__}: {exc}",
            "stages": [],
        }
