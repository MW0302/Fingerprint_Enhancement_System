"""
"Advanced parameters" widgets for the Single Image tab. Every default value
here is copied from the pipeline's own enhance()/internal-function
signatures (read directly, not guessed) — moving a slider away from its
default is the only way any of these differ from calling enhance() with no
params at all, so a user who never opens this section gets exactly the
pipeline's own real, validated defaults.
"""

import streamlit as st

from common import DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR


def _shared_normalize_controls(key_prefix):
    st.caption("Shared Step 0 preprocessing (same default across all four pipelines)")
    target_var = st.slider(
        "normalize_target_var", min_value=25.0, max_value=400.0,
        value=float(DEFAULT_NORMALIZE_TARGET_VAR), step=25.0,
        key=f"{key_prefix}_target_var",
        help=f"Shared default: {DEFAULT_NORMALIZE_TARGET_VAR} (from common.DEFAULT_NORMALIZE_TARGET_VAR)",
    )
    return {
        "normalize_target_mean": DEFAULT_NORMALIZE_TARGET_MEAN,
        "normalize_target_var": target_var,
    }


def render_pipeline_a_controls():
    params = _shared_normalize_controls("a")
    st.caption("Stage 1 — CLAHE")
    params["clahe_clip"] = st.slider("clahe_clip", 0.5, 8.0, 2.0, 0.1, key="a_clahe_clip")
    params["clahe_grid"] = st.slider("clahe_grid", 2, 16, 8, 1, key="a_clahe_grid")
    st.caption("Stage 2 — bilateral denoise")
    params["bilateral_diameter"] = st.slider("bilateral_diameter", 3, 15, 5, 2, key="a_bil_d")
    params["bilateral_sigma_color"] = st.slider("bilateral_sigma_color", 5.0, 100.0, 35.0, 5.0, key="a_bil_sc")
    params["bilateral_sigma_space"] = st.slider("bilateral_sigma_space", 1.0, 20.0, 5.0, 1.0, key="a_bil_ss")
    st.caption("Stage 3 — oriented Gabor")
    params["gabor_strength"] = st.slider("gabor_strength", 0.0, 2.0, 0.7, 0.05, key="a_gabor_strength")
    return params


def render_pipeline_b_controls():
    st.info(
        "Pipeline B has no tunable parameters yet — Steps 1-3 are still "
        "TODO placeholders that don't read any params (only the shared "
        "Step 0 normalize_target_mean/var apply, and they're not exposed "
        "here since there's no technique yet for them to visibly affect)."
    )
    return {}


def render_pipeline_c_controls():
    params = _shared_normalize_controls("c")
    st.caption(
        "Pipeline C's gamma_high / diffusion iterations+kappa / add_gain are "
        "quality-adaptive by default (scaled per image from its own "
        "orientation coherence — see pipeline_c.py's _aggressiveness_alpha). "
        "The overrides below replace that adaptive behaviour with a fixed "
        "value for every image, which departs from the validated default — "
        "off by default."
    )
    if st.checkbox("Override log_gabor_add_gain (adaptive by default)", key="c_override_gain"):
        params["log_gabor_add_gain"] = st.slider(
            "log_gabor_add_gain", 0.5, 4.0, 1.5, 0.1, key="c_add_gain"
        )
    if st.checkbox("Override diffusion_kappa (adaptive by default)", key="c_override_kappa"):
        params["diffusion_kappa"] = st.slider(
            "diffusion_kappa", 5.0, 30.0, 17.5, 0.5, key="c_kappa"
        )
    return params


def render_pipeline_d_controls():
    params = _shared_normalize_controls("d")
    st.caption("Stage 1 — FFT high-frequency emphasis")
    params["fft_high_boost"] = st.slider("fft_high_boost", 0.0, 3.0, 0.75, 0.05, key="d_fft_boost")
    params["fft_blend"] = st.slider("fft_blend", 0.0, 1.0, 0.55, 0.05, key="d_fft_blend")
    st.caption("Stage 2 — Wiener denoise")
    params["wiener_blend"] = st.slider("wiener_blend", 0.0, 1.0, 0.20, 0.05, key="d_wiener_blend")
    st.caption("Stage 3 — STFT reconstruction")
    params["stft_window_size"] = st.select_slider(
        "stft_window_size", options=[8, 16, 24, 32, 40, 48, 56, 64], value=32, key="d_stft_window"
    )
    params["stft_reconstruction_blend"] = st.slider(
        "stft_reconstruction_blend", 0.0, 1.0, 0.25, 0.05, key="d_stft_blend"
    )
    return params


_RENDERERS = {
    "pipeline_a": render_pipeline_a_controls,
    "pipeline_b": render_pipeline_b_controls,
    "pipeline_c": render_pipeline_c_controls,
    "pipeline_d": render_pipeline_d_controls,
}


def render_advanced_params(pipeline_key):
    """Renders the widgets for one pipeline inside the caller's own
    st.expander, returns the resulting params dict (pipeline's real
    defaults unless the user moved a widget)."""
    renderer = _RENDERERS.get(pipeline_key)
    if renderer is None:
        st.info("No advanced parameters available for this pipeline.")
        return {}
    return renderer()
