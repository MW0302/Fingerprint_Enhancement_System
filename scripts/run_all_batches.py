"""
Standalone batch runner — extracted from dashboard/app.py's Batch tab so the
full DB1_B..DB4_B sweep can run without going through Streamlit.

The per-image logic (calling pipeline.enhance(), scoring with
run_nfiq2_single(), the row/CSV schema) is copied as-is from the Batch tab
in dashboard/app.py, not reimplemented, so results line up with what the
dashboard itself would produce. Pipeline is fixed to pipeline_c.

Run with:
    python scripts/run_all_batches.py
"""

import os
import sys
import glob

import cv2
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_c"))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, PROCESSED_DIR, DBS  # noqa: E402
import pipeline_c  # noqa: E402

PIPELINE_KEY = "pipeline_c"


def run_pipeline(image):
    """Same normalisation as dashboard/app.py's run_pipeline(): pipeline_c
    returns a single image (not a tuple), but this mirrors the dashboard's
    handling exactly in case that ever changes."""
    result = pipeline_c.enhance(image)
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, None


def run_batch_for_db(db):
    """Same body as the "Run batch" button handler in dashboard/app.py's
    Batch tab, minus the Streamlit widgets (progress bar / st.dataframe)."""
    db_dir = os.path.join(RAW_DIR, db)
    out_dir = os.path.join(PROCESSED_DIR, PIPELINE_KEY, db)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(db_dir, "*.tif")))
    rows = []

    for i, f in enumerate(files):
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        enhanced, _extra = run_pipeline(image)

        out_path = os.path.join(out_dir, os.path.basename(f))
        cv2.imwrite(out_path, enhanced)

        raw_score, raw_err = run_nfiq2_single(f)
        enh_score, enh_err = run_nfiq2_single(out_path)

        rows.append(
            dict(
                file=os.path.basename(f),
                db=db,
                pipeline=PIPELINE_KEY,
                raw_nfiq2=raw_score,
                enhanced_nfiq2=enh_score,
                raw_error=raw_err,
                enhanced_error=enh_err,
            )
        )
        print(f"[{db}] {i + 1}/{len(files)} {os.path.basename(f)}: "
              f"raw={raw_score} enhanced={enh_score}", flush=True)

    results = pd.DataFrame(rows)
    results_path = os.path.join(out_dir, "batch_results.csv")
    results.to_csv(results_path, index=False)
    print(f"[{db}] saved {results_path}", flush=True)
    return results_path


def summarize(csv_paths):
    print("\n=== Summary ===")
    for db, path in csv_paths.items():
        results = pd.read_csv(path)
        valid = results.dropna(subset=["raw_nfiq2", "enhanced_nfiq2"])
        n = len(valid)
        if n == 0:
            print(f"{db}: 0 valid samples (all NFIQ2 scoring failed)")
            continue
        delta = valid["enhanced_nfiq2"] - valid["raw_nfiq2"]
        improved = (delta > 0).sum()
        regressed = (delta < 0).sum()
        print(
            f"{db}: n={n}  raw_mean={valid['raw_nfiq2'].mean():.2f}  "
            f"enhanced_mean={valid['enhanced_nfiq2'].mean():.2f}  "
            f"mean_delta={delta.mean():+.2f}  "
            f"improved={improved}/{n} ({improved / n * 100:.0f}%)  "
            f"regressed={regressed}/{n} ({regressed / n * 100:.0f}%)"
        )


if __name__ == "__main__":
    csv_paths = {}
    for db in DBS:
        csv_paths[db] = run_batch_for_db(db)

    print("\nCSV files:")
    for db, path in csv_paths.items():
        print(f"  {db}: {path}")

    summarize(csv_paths)
