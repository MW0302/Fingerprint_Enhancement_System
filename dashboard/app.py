"""
Fingerprint Enhancement Dashboard

This wires together the four pipeline modules and NFIQ2 scoring into one
Streamlit app. Run it with:

    streamlit run app.py

Tabs:
    "Single Image"          — pick one image + one pipeline, see before/after,
                               both NFIQ2 scores, per-stage breakdown, timing,
                               and optional advanced parameter overrides.
    "Batch"                 — run every image in one DB through one pipeline,
                               save outputs + batch_results.csv, show mean
                               scores. This is also the fallback for filling
                               in a pipeline/DB the Overview tab doesn't have
                               existing results for yet (see that tab).
    "Overview / Comparison" — master per-image table (Raw/A/B/C/D/Hybrid),
                               summary table, %improved/degraded/unchanged,
                               and charts — built from already-computed
                               result files on disk wherever they exist,
                               falling back to "not yet available" rather
                               than a slow live 320-image re-run.

Reads images from data/raw/ and writes enhanced output to data/processed/,
both relative to the repo root (see src/utils/config.py).
"""

import os
import sys
import glob
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_a"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_b"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_c"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_d"))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, PROCESSED_DIR, DBS  # noqa: E402
import pipeline_a  # noqa: E402
import pipeline_b  # noqa: E402
import pipeline_c  # noqa: E402
import pipeline_d  # noqa: E402

import results_loader  # noqa: E402
import stage_adapters  # noqa: E402
import param_controls  # noqa: E402
import charts  # noqa: E402

PIPELINES = {
    "Pipeline A — CLAHE + Median/Bilateral + Gabor": ("pipeline_a", pipeline_a),
    "Pipeline B — Wavelet Contrast + Wavelet Denoising + Morphology": ("pipeline_b", pipeline_b),
    "Pipeline C — Homomorphic + Coherence Diffusion + Log-Gabor": ("pipeline_c", pipeline_c),
    "Pipeline D — FFT Emphasis + Wiener + STFT": ("pipeline_d", pipeline_d),
}


def run_pipeline(name, image, params=None):
    """Every pipeline's enhance() should return a single greyscale image,
    except Pipeline B which also returns a binary intermediate — this
    normalises both cases to (enhanced_image, extra_or_None). Does NOT
    catch exceptions itself — callers decide how to surface a failure."""
    _, module = PIPELINES[name]
    result = module.enhance(image, params=params) if params else module.enhance(image)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def safe_imread(path):
    """cv2.imread returns None (not an exception) on a failed/corrupt read
    — turn that into a clean st.error instead of an obscure downstream
    crash (e.g. .shape on None)."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        st.error(f"Could not read image file: {path} (corrupt file, unsupported format, or missing).")
        return None
    return image


st.set_page_config(page_title="Fingerprint Enhancement Dashboard", layout="wide")
st.title("Fingerprint Enhancement Dashboard")

tab_single, tab_batch, tab_overview = st.tabs(["Single Image", "Batch", "Overview / Comparison"])

# ---------------------------------------------------------------------------
# Single image tab
# ---------------------------------------------------------------------------
with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        db = st.selectbox("Database", DBS)
    with col2:
        pipeline_name = st.selectbox("Pipeline", list(PIPELINES.keys()))
    pipeline_key, _ = PIPELINES[pipeline_name]

    db_dir = os.path.join(RAW_DIR, db)
    files = sorted(glob.glob(os.path.join(db_dir, "*.tif"))) if os.path.isdir(db_dir) else []

    if not files:
        st.warning(f"No .tif files found in {db_dir} — copy your dataset into data/raw/ (see README).")
    else:
        filename = st.selectbox("Image", [os.path.basename(f) for f in files])
        image_path = os.path.join(db_dir, filename)

        with st.expander("Advanced parameters (optional — defaults match this pipeline's own enhance())"):
            params = param_controls.render_advanced_params(pipeline_key)

        show_stages = st.checkbox("Show intermediate stages (Stage 1 / 2 / 3)", value=False)

        if st.button("Run"):
            image = safe_imread(image_path)
            if image is not None:
                try:
                    start = time.perf_counter()
                    enhanced, extra = run_pipeline(pipeline_name, image, params)
                    enhance_ms = (time.perf_counter() - start) * 1000.0
                except Exception as exc:  # noqa: BLE001 — keep the rest of the page alive
                    st.error(f"{pipeline_name} raised an error while running: {type(exc).__name__}: {exc}")
                    enhanced, extra, enhance_ms = None, None, None

                if enhanced is not None:
                    col_before, col_after = st.columns(2)
                    with col_before:
                        st.subheader("Original")
                        st.image(image, use_container_width=True, clamp=True)
                    with col_after:
                        st.subheader("Enhanced")
                        st.image(enhanced, use_container_width=True, clamp=True)
                    st.caption(f"enhance() wall-clock time: {enhance_ms:.0f} ms")

                    if extra is not None:
                        st.caption("Binary intermediate (structural aid only — not sent to NFIQ2)")
                        st.image(extra, use_container_width=True, clamp=True)

                    with st.spinner("Scoring with NFIQ2..."):
                        raw_score, raw_err = run_nfiq2_single(image_path)

                        tmp_path = os.path.join(os.path.dirname(__file__), "_tmp_enhanced.png")
                        cv2.imwrite(tmp_path, enhanced)
                        enh_score, enh_err = run_nfiq2_single(tmp_path)
                        os.remove(tmp_path)

                    score_col1, score_col2 = st.columns(2)
                    score_col1.metric("Raw NFIQ2", raw_score if raw_score is not None else "failed")
                    score_col2.metric(
                        "Enhanced NFIQ2",
                        enh_score if enh_score is not None else "failed",
                        delta=(None if raw_score is None or enh_score is None else enh_score - raw_score),
                    )
                    if raw_err:
                        st.caption(f"Raw image NFIQ2 note: {raw_err}")
                    if enh_err:
                        st.caption(f"Enhanced image NFIQ2 note: {enh_err}")

                    # -----------------------------------------------------
                    # Intermediate stages
                    # -----------------------------------------------------
                    if show_stages:
                        st.divider()
                        st.subheader("Intermediate stages")
                        stage_result = stage_adapters.get_stages(pipeline_key, image, params)
                        if not stage_result["available"]:
                            st.info(stage_result["message"])
                        else:
                            if stage_result["message"]:
                                st.caption(stage_result["message"])
                            stage_cols = st.columns(len(stage_result["stages"]))
                            prev_score = raw_score
                            for col, stage in zip(stage_cols, stage_result["stages"]):
                                with col:
                                    st.image(stage["image"], use_container_width=True, clamp=True)
                                    st.caption(stage["name"])
                                    tmp_stage_path = os.path.join(
                                        os.path.dirname(__file__), "_tmp_stage.png"
                                    )
                                    cv2.imwrite(tmp_stage_path, stage["image"])
                                    stage_score, stage_err = run_nfiq2_single(tmp_stage_path)
                                    os.remove(tmp_stage_path)
                                    if stage["time_ms"] is not None:
                                        st.caption(f"{stage['time_ms']:.0f} ms")
                                    if stage_score is not None:
                                        delta_txt = (
                                            f" (Δ {stage_score - prev_score:+.1f})"
                                            if prev_score is not None else ""
                                        )
                                        st.metric("NFIQ2", stage_score, delta=None)
                                        st.caption(f"marginal delta vs previous stage{delta_txt}")
                                        prev_score = stage_score
                                    elif stage_err:
                                        st.caption(f"NFIQ2 note: {stage_err}")

# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Runs every image in a chosen database through one pipeline, saves the "
        "enhanced images, and scores every output with NFIQ2. Use this only "
        "after a pipeline has been validated on a handful of images in the "
        "Single Image tab above — see the analysis document's recommendation "
        "to pilot on 10-20 images before the full 320-image batch. This is "
        "also how to fill in a pipeline/DB combination the Overview tab "
        "reports as \"not yet available\" — its results are picked up "
        "automatically from data/processed/<pipeline>/<db>/batch_results.csv "
        "the next time that tab loads."
    )
    batch_db = st.selectbox("Database ", DBS, key="batch_db")
    batch_pipeline_name = st.selectbox("Pipeline ", list(PIPELINES.keys()), key="batch_pipeline")

    if st.button("Run batch"):
        pipeline_key, _ = PIPELINES[batch_pipeline_name]
        db_dir = os.path.join(RAW_DIR, batch_db)
        out_dir = os.path.join(PROCESSED_DIR, pipeline_key, batch_db)
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(glob.glob(os.path.join(db_dir, "*.tif")))
        progress = st.progress(0.0)
        rows = []

        for i, f in enumerate(files):
            image = safe_imread(f)
            if image is None:
                progress.progress((i + 1) / len(files))
                continue

            try:
                enhanced, _extra = run_pipeline(batch_pipeline_name, image)
            except Exception as exc:  # noqa: BLE001 — one bad image shouldn't kill the batch
                st.warning(f"{os.path.basename(f)}: {batch_pipeline_name} raised "
                           f"{type(exc).__name__}: {exc} — skipped")
                progress.progress((i + 1) / len(files))
                continue

            out_path = os.path.join(out_dir, os.path.basename(f))
            cv2.imwrite(out_path, enhanced)

            raw_score, raw_err = run_nfiq2_single(f)
            enh_score, enh_err = run_nfiq2_single(out_path)

            rows.append(
                dict(
                    file=os.path.basename(f),
                    db=batch_db,
                    pipeline=pipeline_key,
                    raw_nfiq2=raw_score,
                    enhanced_nfiq2=enh_score,
                    raw_error=raw_err,
                    enhanced_error=enh_err,
                )
            )
            progress.progress((i + 1) / len(files))

        results = pd.DataFrame(rows)
        st.dataframe(results)

        results_path = os.path.join(out_dir, "batch_results.csv")
        results.to_csv(results_path, index=False)
        st.success(f"Saved {results_path}")

        valid = results.dropna(subset=["raw_nfiq2", "enhanced_nfiq2"])
        if len(valid) > 0:
            st.metric("Mean raw NFIQ2", round(valid["raw_nfiq2"].mean(), 2))
            st.metric("Mean enhanced NFIQ2", round(valid["enhanced_nfiq2"].mean(), 2))
            improved = (valid["enhanced_nfiq2"] > valid["raw_nfiq2"]).sum()
            st.caption(f"{improved} of {len(valid)} images improved")

# ---------------------------------------------------------------------------
# Overview / Comparison tab
# ---------------------------------------------------------------------------
with tab_overview:
    st.write(
        "Loads already-computed NFIQ2 results per pipeline from disk (see "
        "dashboard/results_loader.py for exactly where each one is read "
        "from) instead of re-running the full 320-image batch here — that "
        "would take a long time and duplicate work already done. A "
        "pipeline/DB with no existing results shows as \"not yet "
        "available\"; use the Batch tab to generate and save them."
    )

    if st.button("Refresh results from disk"):
        st.cache_data.clear()

    @st.cache_data
    def _load_all_cached():
        return results_loader.load_all()

    loaded = _load_all_cached()
    availability = results_loader.availability_summary(loaded)
    st.dataframe(availability, use_container_width=True, hide_index=True)

    master, inconsistency_notes = results_loader.build_master_table(loaded)
    for note in inconsistency_notes:
        st.warning(note)

    if master.empty:
        st.info("No results found anywhere on disk yet — run the Batch tab for at least one pipeline/DB first.")
    else:
        st.subheader("Master per-image results")
        db_filter = st.multiselect("Filter by DB", DBS, default=DBS)
        st.dataframe(
            master[master["DB"].isin(db_filter)] if db_filter else master,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Summary: mean NFIQ2 per method")
        summary = results_loader.build_summary_table(master)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        charts.mean_nfiq2_bar_chart(summary)

        st.subheader("% improved / degraded / unchanged per DB")
        improvement = results_loader.build_improvement_stats(master)
        st.dataframe(improvement, use_container_width=True, hide_index=True)

        st.subheader("Before / after distribution")
        available_labels = [
            label for label in ("Pipeline A", "Pipeline B", "Pipeline C", "Pipeline D", "Hybrid")
            if label in master.columns and master[label].notna().any()
        ]
        if available_labels:
            dist_pipeline = st.selectbox("Pipeline", available_labels, key="dist_pipeline")
            charts.before_after_distribution_chart(master, dist_pipeline)
        else:
            st.info("No pipeline has scored results yet — nothing to plot.")
