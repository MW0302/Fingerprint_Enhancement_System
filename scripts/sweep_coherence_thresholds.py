"""
Sweep _COHERENCE_FULL_GENTLE / _COHERENCE_FULL_AGGRESSIVE in pipeline_c.py.

Does NOT edit pipeline_c.py -- monkey-patches the two module-level constants
at runtime before calling the REAL enhance() (which reads them as globals
inside _aggressiveness_alpha()), so every candidate runs the actual,
unmodified pipeline end to end, not a reimplementation. Metric is Δtotal
(stage3/final enhanced NFIQ2 minus raw NFIQ2) -- these two thresholds shape
all three adaptive parameters (gamma_high, iterations/kappa, add_gain)
simultaneously, so only the final output is a fair readout of their effect.

raw_nfiq2 is loaded from the current data/processed/pipeline_c/<DB>/
batch_results.csv (already validated, candidate-independent) rather than
rescored, to roughly halve the NFIQ2 subprocess-call cost.

Run with:
    python scripts/sweep_coherence_thresholds.py --phase aggressive
    python scripts/sweep_coherence_thresholds.py --phase gentle --aggressive 0.55
(exact CLI shaped around the two-phase design in the task.)
"""

import os
import sys
import glob
import argparse

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_c"))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, PROCESSED_DIR, DBS  # noqa: E402
import pipeline_c  # noqa: E402

import tempfile


def load_raw_nfiq2():
    rows = []
    for db in DBS:
        path = os.path.join(PROCESSED_DIR, "pipeline_c", db, "batch_results.csv")
        df = pd.read_csv(path)
        rows.append(df[["file", "db", "raw_nfiq2"]])
    return pd.concat(rows, ignore_index=True).set_index(["file", "db"])["raw_nfiq2"]


def score_enhanced(img):
    tmp_path = os.path.join(tempfile.gettempdir(), f"coherence_sweep_tmp_{os.getpid()}.tif")
    cv2.imwrite(tmp_path, img)
    score, err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return score, err


def run_candidate(gentle, aggressive, raw_nfiq2):
    pipeline_c._COHERENCE_FULL_GENTLE = gentle
    pipeline_c._COHERENCE_FULL_AGGRESSIVE = aggressive

    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            enhanced = pipeline_c.enhance(img)
            score, _err = score_enhanced(enhanced)
            raw = raw_nfiq2.get((fname, db))
            rows.append(dict(file=fname, db=db, gentle=gentle, aggressive=aggressive,
                              raw_nfiq2=raw, enhanced_nfiq2=score))
        print(f"  [{gentle=} {aggressive=}] {db}: {len(paths)} images done", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    df = df.copy()
    df["delta_total"] = df["enhanced_nfiq2"] - df["raw_nfiq2"]
    valid = df.dropna(subset=["raw_nfiq2", "enhanced_nfiq2"])
    pivot = valid.pivot_table(index=["gentle", "aggressive"], columns="db", values="delta_total", aggfunc="mean")
    pivot["OVERALL"] = valid.groupby(["gentle", "aggressive"])["delta_total"].mean()
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["aggressive", "gentle"], required=True)
    parser.add_argument("--aggressive", type=float, default=None,
                         help="fixed aggressive value for the gentle-sweep phase")
    parser.add_argument("--gentle-candidates", type=str, default=None,
                         help="comma-separated gentle candidates for the gentle-sweep phase")
    args = parser.parse_args()

    raw_nfiq2 = load_raw_nfiq2()
    print(f"Loaded raw_nfiq2 for {len(raw_nfiq2)} images from current batch_results.csv\n")

    all_dfs = []

    if args.phase == "aggressive":
        gentle_fixed = 0.68
        candidates = [0.45, 0.50, 0.55, 0.60, 0.65]
        print(f"=== PHASE A: sweeping FULL_AGGRESSIVE, FULL_GENTLE fixed at {gentle_fixed} ===")
        for aggressive in candidates:
            print(f"\n--- aggressive={aggressive} ---")
            df = run_candidate(gentle_fixed, aggressive, raw_nfiq2)
            all_dfs.append(df)
        out_path = os.path.join(RESULTS_DIR, "coherence_sweep_aggressive.csv")

    else:  # gentle
        if args.aggressive is None:
            raise SystemExit("--aggressive is required for --phase gentle")
        aggressive_fixed = args.aggressive
        if args.gentle_candidates:
            candidates = [float(x) for x in args.gentle_candidates.split(",")]
        else:
            candidates = [round(aggressive_fixed + 0.05 * k, 2) for k in range(1, 6)]
        print(f"=== PHASE B: sweeping FULL_GENTLE, FULL_AGGRESSIVE fixed at {aggressive_fixed} ===")
        print(f"gentle candidates: {candidates}")
        for gentle in candidates:
            print(f"\n--- gentle={gentle} ---")
            df = run_candidate(gentle, aggressive_fixed, raw_nfiq2)
            all_dfs.append(df)
        out_path = os.path.join(RESULTS_DIR, "coherence_sweep_gentle.csv")

    full_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(full_df)} rows to {out_path}")

    print("\n=== Mean delta_total (enhanced - raw NFIQ2) per candidate, per DB ===")
    pivot = summarize(full_df)
    print(pivot.round(3).to_string())

    best_overall = pivot["OVERALL"].idxmax()
    print(f"\nBest candidate by OVERALL mean delta_total: {best_overall} "
          f"(delta_total={pivot.loc[best_overall, 'OVERALL']:.3f})")


if __name__ == "__main__":
    main()
