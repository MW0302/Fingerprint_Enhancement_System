"""
Sweep _LOG_GABOR_ADD_GAIN_RANGE (gentle, aggressive) in pipeline_c.py.

Does NOT edit pipeline_c.py -- reuses scripts/pipeline_c_ablation.py's
ablate() wrapper unmodified for every candidate, so the actual Log-Gabor
step (_log_gabor_enhance) is never reimplemented here, only its add_gain
input varies. Metric is delta_p6 = stage3 (Log-Gabor output) NFIQ2 minus
stage2 (post-diffusion) NFIQ2 -- the marginal contribution of the Log-Gabor
step alone, not delta_total, since add_gain only affects this one stage and
Δtotal would dilute the signal with P1/P2's contributions.

IMPORTANT monkey-patch target: pipeline_c_ablation.py did
`from pipeline_c import _LOG_GABOR_ADD_GAIN_RANGE`, which copies the name
into pipeline_c_ablation's OWN module namespace at import time. ablate()
reads that bare name, resolved against pipeline_c_ablation's own globals --
patching pipeline_c._LOG_GABOR_ADD_GAIN_RANGE (as the coherence-threshold
sweep did, correctly, for _COHERENCE_FULL_GENTLE/_AGGRESSIVE, which
ablate() never imports and always reads live off the pipeline_c module via
_aggressiveness_alpha() defined there) would silently have NO EFFECT here.
This script patches pipeline_c_ablation's own copy instead.

Run with:
    python scripts/sweep_log_gabor_add_gain.py --phase aggressive
    python scripts/sweep_log_gabor_add_gain.py --phase gentle --aggressive 2.5
"""

import os
import sys
import glob
import argparse

import cv2
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_c"))
sys.path.append(os.path.dirname(__file__))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402
import pipeline_c_ablation as abl  # noqa: E402

import tempfile


def score(img):
    tmp_path = os.path.join(tempfile.gettempdir(), f"loggabor_sweep_tmp_{os.getpid()}.tif")
    cv2.imwrite(tmp_path, img)
    s, _err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return s


def run_candidate(gentle, aggressive):
    abl._LOG_GABOR_ADD_GAIN_RANGE = (gentle, aggressive)

    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            result = abl.ablate(img)  # real, unmodified wrapper -> real internal functions
            stage2_score = score(result["stage2"])
            stage3_score = score(result["stage3"])
            rows.append(dict(file=fname, db=db, gentle=gentle, aggressive=aggressive,
                              stage2_noise_nfiq2=stage2_score, stage3_orientation_nfiq2=stage3_score))
        print(f"  [{gentle=} {aggressive=}] {db}: {len(paths)} images done", flush=True)
    return pd.DataFrame(rows)


def summarize(df):
    df = df.copy()
    df["delta_p6"] = df["stage3_orientation_nfiq2"] - df["stage2_noise_nfiq2"]
    valid = df.dropna(subset=["stage2_noise_nfiq2", "stage3_orientation_nfiq2"])
    pivot = valid.pivot_table(index=["gentle", "aggressive"], columns="db", values="delta_p6", aggfunc="mean")
    pivot["OVERALL"] = valid.groupby(["gentle", "aggressive"])["delta_p6"].mean()
    return pivot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["aggressive", "gentle"], required=True)
    parser.add_argument("--aggressive", type=float, default=None,
                         help="fixed aggressive value for the gentle-sweep phase")
    parser.add_argument("--gentle", type=float, default=None,
                         help="fixed gentle value for the aggressive-sweep phase (default 0.8)")
    parser.add_argument("--aggressive-candidates", type=str, default=None)
    parser.add_argument("--gentle-candidates", type=str, default=None)
    parser.add_argument("--out-suffix", type=str, default="",
                         help="appended to the output CSV filename, to avoid overwriting an earlier phase's file")
    args = parser.parse_args()

    all_dfs = []

    if args.phase == "aggressive":
        gentle_fixed = args.gentle if args.gentle is not None else 0.8
        candidates = ([float(x) for x in args.aggressive_candidates.split(",")]
                      if args.aggressive_candidates else [1.5, 2.0, 2.5, 3.0, 3.5])
        print(f"=== PHASE A: sweeping add_gain AGGRESSIVE, gentle fixed at {gentle_fixed} ===")
        for aggressive in candidates:
            print(f"\n--- aggressive={aggressive} ---")
            all_dfs.append(run_candidate(gentle_fixed, aggressive))
        out_path = os.path.join(RESULTS_DIR, f"log_gabor_add_gain_sweep_aggressive{args.out_suffix}.csv")

    else:  # gentle
        if args.aggressive is None:
            raise SystemExit("--aggressive is required for --phase gentle")
        aggressive_fixed = args.aggressive
        candidates = ([float(x) for x in args.gentle_candidates.split(",")]
                      if args.gentle_candidates else [0.4, 0.6, 0.8, 1.0, 1.2])
        print(f"=== PHASE B: sweeping add_gain GENTLE, aggressive fixed at {aggressive_fixed} ===")
        print(f"gentle candidates: {candidates}")
        for gentle in candidates:
            print(f"\n--- gentle={gentle} ---")
            all_dfs.append(run_candidate(gentle, aggressive_fixed))
        out_path = os.path.join(RESULTS_DIR, f"log_gabor_add_gain_sweep_gentle{args.out_suffix}.csv")

    full_df = pd.concat(all_dfs, ignore_index=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(full_df)} rows to {out_path}")

    print("\n=== Mean delta_p6 (stage3 - stage2 NFIQ2) per candidate, per DB ===")
    pivot = summarize(full_df)
    print(pivot.round(3).to_string())

    best = pivot["OVERALL"].idxmax()
    print(f"\nBest candidate by OVERALL mean delta_p6: {best} (delta_p6={pivot.loc[best, 'OVERALL']:.3f})")


if __name__ == "__main__":
    main()
