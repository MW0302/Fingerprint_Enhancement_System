"""
Batch runner for the pipeline_c experiments in this directory. Reuses the
same per-image logic as scripts/run_all_batches.py (which itself mirrors
dashboard/app.py's Batch tab): enhance() the image, save it, score raw and
enhanced with common.run_nfiq2_single(), build a results CSV with the same
column schema (file, db, pipeline, raw_nfiq2, enhanced_nfiq2, raw_error,
enhanced_error).

Writes to data/processed/<output_key>/<DB>/batch_results.csv — a SEPARATE
tree from data/processed/pipeline_c/ (the verified pipeline's own batch
output), so this never overwrites the already-verified results.

Usage:
    python run_experiment_batch.py exp1
    python run_experiment_batch.py exp2
"""

import os
import sys
import glob
import json

import cv2
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src", "utils"))
sys.path.append(os.path.dirname(__file__))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, PROCESSED_DIR, DBS  # noqa: E402

EXPERIMENTS = {
    "exp1": dict(
        module="pipeline_c_exp1_homomorphic_confidence",
        output_key="pipeline_c_exp1_homomorphic_confidence",
        params=None,
    ),
    "exp2": dict(
        module="pipeline_c_exp2_loggabor_replace",
        output_key="pipeline_c_exp2_loggabor_replace",
        params={"log_gabor_mode": "replace_confidence_gated"},
    ),
}


def run_batch_for_db(db, module, output_key, params):
    db_dir = os.path.join(RAW_DIR, db)
    out_dir = os.path.join(PROCESSED_DIR, output_key, db)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(db_dir, "*.tif")))
    rows = []

    for i, f in enumerate(files):
        image = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        enhanced = module.enhance(image, params=params)

        out_path = os.path.join(out_dir, os.path.basename(f))
        cv2.imwrite(out_path, enhanced)

        raw_score, raw_err = run_nfiq2_single(f)
        enh_score, enh_err = run_nfiq2_single(out_path)

        rows.append(
            dict(
                file=os.path.basename(f),
                db=db,
                pipeline=output_key,
                raw_nfiq2=raw_score,
                enhanced_nfiq2=enh_score,
                raw_error=raw_err,
                enhanced_error=enh_err,
            )
        )
        print(f"[{output_key}][{db}] {i + 1}/{len(files)} {os.path.basename(f)}: "
              f"raw={raw_score} enhanced={enh_score}", flush=True)

    results = pd.DataFrame(rows)
    results_path = os.path.join(out_dir, "batch_results.csv")
    results.to_csv(results_path, index=False)
    print(f"[{output_key}][{db}] saved {results_path}", flush=True)
    return results_path


def summarize(csv_paths, label):
    print(f"\n=== Summary: {label} ===")
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
    exp_name = sys.argv[1]
    exp = EXPERIMENTS[exp_name]
    module = __import__(exp["module"])

    csv_paths = {}
    for db in DBS:
        csv_paths[db] = run_batch_for_db(db, module, exp["output_key"], exp["params"])

    print("\nCSV files:")
    for db, path in csv_paths.items():
        print(f"  {db}: {path}")

    summarize(csv_paths, exp_name)
