"""
Re-validation, NOT a change to pipeline_a.py or common.py.

Pipeline A's enhance() is currently: normalize_image() -> segment()
(unused for output) -> CLAHE -> [Steps 2-4 are TODO passthroughs] -> return.
So enhance(image) == CLAHE(normalize_image(image)) exactly as things stand
today -- confirmed by reading src/pipeline_a/pipeline_a.py directly, not
assumed.

Why this needs re-checking: common.py's own docstring records that the
normalize_image() target_var default (std=10 -> std=40) was chosen using
CLAHE as the test case -- but that original check was a SINGLE test image
(DB3_B/108_6.tif) compared by CLAHE's own OUTPUT STD (49 raw-CLAHE vs 17
after std=10 normalisation), not NFIQ2, and not a real sweep across
candidate values -- std=40 was picked directly as "close to the dataset's
own typical raw std", never swept against other candidates. That original
check used exactly the kind of intermediate-statistic-only methodology this
project's own Section 2.1/4.0 notes have separately flagged as a trap (see
Evaluation_Algorithm_ML_Technique_Selection.md) -- it can miss a residual
downstream NFIQ2 effect that isn't visible in an output std number.

normalize_image() has since changed (FIFTH revision, headroom-capped
recentring on each image's own mean instead of a fixed target_mean=100) --
so CLAHE's input distribution is different now, and the original std=10 vs
std=40 comparison (itself never a real sweep) doesn't necessarily still
hold.

This script re-runs that comparison properly: multiple target_std
candidates, the REAL post-fix normalize_image() and the REAL CLAHE call
pipeline_a.py uses (clip_limit=2.0, grid=8, its current defaults --
untouched, this script does not tune those), evaluated with delta_p1 =
stage1_nfiq2 - raw_nfiq2 (the same metric the pipeline_c_ablation.py
framework already established) across all 320 images / all 4 DBs, not a
single test image.

Run with:
    python scripts/revalidate_pipeline_a_clahe_normalize_std.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from common import normalize_image, run_nfiq2_single  # noqa: E402  -- REAL, post-fix function
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402

# Candidate target_std values. 10 and 40 are the two values that appear in
# common.py's own revision history (10 = Hong et al.'s textbook default, 40
# = the empirically-picked current default). No other candidates were ever
# actually swept in that original decision, so this adds a proper spread
# (20/30/50/60/70) spanning below and above the dataset's documented raw std
# range (roughly 40-70 across subsets) to actually characterise the curve,
# not just re-confirm the same two points.
CANDIDATE_STDS = [10, 20, 30, 40, 50, 60, 70]

# Pipeline A's own current CLAHE defaults (src/pipeline_a/pipeline_a.py) --
# NOT being tuned here, only normalize_image()'s target_std feeding into it.
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID = 8


def clahe_stage1(normalized_img):
    """Reproduces pipeline_a.py's Step 1 exactly (same library call, same
    defaults) -- not a separate importable function there, so replicated
    verbatim rather than reimplemented differently."""
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID, CLAHE_GRID))
    return clahe.apply(normalized_img)


def main():
    # Score raw NFIQ2 once per image (candidate-independent).
    raw_scores = {}
    rows_raw = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            score, _err = run_nfiq2_single(path)
            raw_scores[(db, fname)] = score
        print(f"{db}: {len(paths)} raw images scored")

    all_rows = []
    for target_std in CANDIDATE_STDS:
        target_var = float(target_std) ** 2
        print(f"\n--- target_std={target_std} (target_var={target_var:.0f}) ---")
        for db in DBS:
            paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
            for path in paths:
                fname = os.path.basename(path)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                normalized = normalize_image(img, target_var=target_var)
                stage1 = clahe_stage1(normalized)
                stage1_score, _err = run_nfiq2_single_from_array(stage1)

                raw_score = raw_scores.get((db, fname))
                all_rows.append(dict(
                    file=fname, db=db, target_std=target_std,
                    raw_nfiq2=raw_score, stage1_nfiq2=stage1_score,
                ))
            print(f"  {db}: {len(paths)} images done")

    df = pd.DataFrame(all_rows)
    df["delta_p1"] = df["stage1_nfiq2"] - df["raw_nfiq2"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pipeline_a_clahe_normalize_std_sweep.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    print("\n=== Mean delta_p1 (stage1 - raw NFIQ2) per candidate std, per DB ===")
    valid = df.dropna(subset=["raw_nfiq2", "stage1_nfiq2"])
    pivot = valid.pivot_table(index="target_std", columns="db", values="delta_p1", aggfunc="mean")
    pivot["OVERALL"] = valid.groupby("target_std")["delta_p1"].mean()
    print(pivot.round(3).to_string())

    print("\n=== %improved per candidate std, per DB ===")
    def pct_improved(g):
        return (g["delta_p1"] > 0).mean() * 100
    improved_pivot = valid.groupby(["target_std", "db"]).apply(pct_improved).unstack("db")
    improved_pivot["OVERALL"] = valid.groupby("target_std").apply(pct_improved)
    print(improved_pivot.round(1).to_string())

    print("\n=== Best candidate per DB (highest mean delta_p1) ===")
    for db in DBS:
        db_pivot = pivot[db]
        best_std = db_pivot.idxmax()
        print(f"  {db}: best std={best_std}  (delta_p1={db_pivot[best_std]:.3f})  "
              f"vs std=40: {pivot.loc[40, db]:.3f}  vs std=10: {pivot.loc[10, db]:.3f}")

    best_overall = pivot["OVERALL"].idxmax()
    print(f"\n  OVERALL: best std={best_overall}  (delta_p1={pivot.loc[best_overall, 'OVERALL']:.3f})  "
          f"vs std=40: {pivot.loc[40, 'OVERALL']:.3f}  vs std=10: {pivot.loc[10, 'OVERALL']:.3f}")


_NFIQ2_TMP_COUNTER = [0]


def run_nfiq2_single_from_array(img):
    """Writes img to a temp file and scores it with run_nfiq2_single(), same
    pattern used elsewhere in this project's scripts (e.g.
    pipeline_c_ablation.py's score_nfiq2()). Uses a counter-suffixed temp
    filename to avoid the shared-temp-file collision found and documented in
    run_nfiq2_single() itself while re-validating the normalize_image() fix
    (see the FIFTH-revision commit) -- this script only ever runs one NFIQ2
    call at a time itself, but a unique name costs nothing and avoids ever
    reintroducing that class of bug here."""
    import tempfile
    _NFIQ2_TMP_COUNTER[0] += 1
    tmp_path = os.path.join(
        tempfile.gettempdir(), f"pipeline_a_clahe_sweep_tmp_{os.getpid()}_{_NFIQ2_TMP_COUNTER[0]}.tif"
    )
    cv2.imwrite(tmp_path, img)
    score, err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return score, err


if __name__ == "__main__":
    main()
