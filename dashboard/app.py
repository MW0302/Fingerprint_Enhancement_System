"""
Fingerprint Enhancement Dashboard — skeleton

This wires together the four pipeline modules and NFIQ2 scoring into one
Streamlit app, so pipeline code has somewhere to plug in from day one instead
of being integrated at the last minute. Run it with:

    streamlit run app.py

Two tabs:
    "Single Image"  — pick one image + one pipeline, see before/after and
                       both NFIQ2 scores immediately. Use this while
                       developing/debugging a pipeline.
    "Batch"         — run every image in a folder through one pipeline, save
                       the enhanced images, and build the master results CSV
                       described in the Handover Notes (Section 15).

Reads images from data/raw/ and writes enhanced output to data/processed/,
both relative to the repo root (see src/utils/config.py) — every teammate's
own local dataset copy just needs to live at data/raw/DB1_B, DB2_B, etc.
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd
import streamlit as st

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

PIPELINES = {
    "Pipeline A — CLAHE + Gabor": ("pipeline_a", pipeline_a),
    "Pipeline B — Wavelet + Threshold + Morphology": ("pipeline_b", pipeline_b),
    "Pipeline C — Coherence Diffusion + Log-Gabor": ("pipeline_c", pipeline_c),
    "Pipeline D — STFT": ("pipeline_d", pipeline_d),
}


def run_pipeline(name, image):
    """Every pipeline's enhance() should return a single greyscale image,
    except Pipeline B which also returns a binary intermediate — this
    normalises both cases to (enhanced_image, extra_or_None)."""
    _, module = PIPELINES[name]
    result = module.enhance(image)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


st.set_page_config(page_title="Fingerprint Enhancement Dashboard", layout="wide")
st.title("Fingerprint Enhancement Dashboard")

tab_single, tab_batch = st.tabs(["Single Image", "Batch"])

# ---------------------------------------------------------------------------
# Single image tab
# ---------------------------------------------------------------------------
with tab_single:
    col1, col2 = st.columns(2)
    with col1:
        db = st.selectbox("Database", DBS)
    with col2:
        pipeline_name = st.selectbox("Pipeline", list(PIPELINES.keys()))

    db_dir = os.path.join(RAW_DIR, db)
    files = sorted(glob.glob(os.path.join(db_dir, "*.tif"))) if os.path.isdir(db_dir) else []

    if not files:
        st.warning(f"No .tif files found in {db_dir} — copy your dataset into data/raw/ (see README).")
    else:
        filename = st.selectbox("Image", [os.path.basename(f) for f in files])
        image_path = os.path.join(db_dir, filename)

        if st.button("Run"):
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            enhanced, extra = run_pipeline(pipeline_name, image)

            col_before, col_after = st.columns(2)
            with col_before:
                st.subheader("Original")
                st.image(image, use_container_width=True, clamp=True)
            with col_after:
                st.subheader("Enhanced")
                st.image(enhanced, use_container_width=True, clamp=True)

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

# ---------------------------------------------------------------------------
# Batch tab
# ---------------------------------------------------------------------------
with tab_batch:
    st.write(
        "Runs every image in a chosen database through one pipeline, saves the "
        "enhanced images, and scores every output with NFIQ2. Use this only "
        "after a pipeline has been validated on a handful of images in the "
        "Single Image tab above — see the analysis document's recommendation "
        "to pilot on 10-20 images before the full 320-image batch."
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
            image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            enhanced, _extra = run_pipeline(batch_pipeline_name, image)

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
