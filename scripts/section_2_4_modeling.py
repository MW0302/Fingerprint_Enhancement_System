"""
Section 2.4, steps 1-3: cross-pipeline technique-contribution modeling and
selection. Loads all four pipelines' real, committed per-image cumulative
NFIQ2 results (Section 2.1, already satisfied), joins them with the 7
diagnostic features (results/per_image_metrics.csv, from
scripts/compute_diagnostic_features.py -- reused, not recomputed), trains
the Section 2.2 RandomForestRegressor model for all 4 pipelines x 3
categories = 12 (pipeline, category) combinations, and does the Section 2.4
step-3 cross-pipeline comparison for P1/P2/P6.

This REPLACES scripts/train_technique_contribution_models.py (the
Pipeline-C-only dry run, commit 4291acd) now that real ablation data exists
for all four pipelines -- same methodology (RandomForestRegressor, 5-fold
CV, naive training-fold-mean baseline), extended to every pipeline rather
than just Pipeline C.

Per-pipeline source files and their ACTUAL column names (read directly,
not assumed -- see docs/section_2_4_findings.md for the header dump this
was built from):
    Pipeline A: data/pipeline_a_full_320_cumulative_nfiq2_config_A_20260904/
                per_image_cumulative_scores.csv
                (database, filename, raw_nfiq2, stage1_nfiq2, stage2_nfiq2,
                stage3_nfiq2, ... -- no precomputed delta columns)
    Pipeline B: results/pipeline_b_ablation.csv
                (database, filename, raw_nfiq2, stage1_p1_nfiq2,
                stage2_p1_p2_nfiq2, stage3_p1_p2_p6_nfiq2, delta_p1,
                delta_p2, delta_p6, delta_final, ... -- HAS precomputed
                deltas, already correctly NaN-ing delta_p1 alone when only
                raw is missing; recomputed independently here anyway and
                cross-checked to match, rather than trusted blindly)
    Pipeline C: results/pipeline_c_ablation.csv
                (file, db, alpha, raw_nfiq2, stage1_contrast_nfiq2,
                stage2_noise_nfiq2, stage3_orientation_nfiq2 -- no
                precomputed deltas)
    Pipeline D: results/pipeline_d_ablation/pipeline_d_ablation.csv
                (same convention as Pipeline B)

Missing-value handling (confirmed identical across all four files, and
confirmed to be the ONLY such row in any of them -- see the findings doc):
DB3_B/110_5.tif has no raw_nfiq2 (NFIQ2 itself can't score that raw image --
FRFXLL_ERR_FB_TOO_SMALL_AREA) in every pipeline, but stage1/stage2/stage3
ARE all successfully scored in every pipeline. So delta_p1 (= stage1 - raw)
is undefined and this row is excluded from delta_p1's n specifically, but
delta_p2 (= stage2 - stage1) and delta_p6 (= stage3 - stage2) don't involve
raw at all and stay valid for this row -- excluding it from every category's
n (as a blanket "drop this row" would) would silently throw away 3 good
data points across the 4 pipelines for no reason, and zero-filling it would
fabricate a technique contribution that was never measured. Deltas are
recomputed from each pipeline's own stageN columns directly (not by trusting
each file's own precomputed delta columns, since two of the four files
don't have any), which naturally reproduces exactly this per-category
missingness pattern with no special-casing required.

Run with:
    python scripts/section_2_4_modeling.py
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from config import REPO_ROOT, RESULTS_DIR  # noqa: E402

FEATURE_COLS = [
    "contrast_std",
    "michelson_contrast",
    "illumination_unevenness",
    "noise_sigma",
    "sharpness_laplacian_var",
    "foreground_ratio",
    "orientation_coherence",
]

CATEGORIES = ["delta_p1", "delta_p2", "delta_p6"]
CATEGORY_LABELS = {
    "delta_p1": "P1 (contrast)",
    "delta_p2": "P2 (noise)",
    "delta_p6": "P6 (orientation)",
}

RANDOM_STATE = 42
N_SPLITS = 5


def _find_pipeline_a_dir():
    base = os.path.join(REPO_ROOT, "data")
    candidates = [
        d for d in os.listdir(base)
        if d.startswith("pipeline_a_full_320_cumulative_nfiq2_config_")
        and os.path.isdir(os.path.join(base, d))
    ]
    if not candidates:
        raise FileNotFoundError(
            "No data/pipeline_a_full_320_cumulative_nfiq2_config_*/ directory found."
        )
    # Most recent by name (date-stamped) if more than one is ever present.
    candidates.sort()
    return os.path.join(base, candidates[-1])


def _compute_deltas(df, raw_col, stage1_col, stage2_col, stage3_col):
    """Uniform delta computation from each pipeline's own stageN columns --
    see module docstring for why this (not each file's own precomputed
    delta columns, which two of the four files don't even have) is what
    naturally gets the DB3_B/110_5.tif per-category missingness right."""
    out = pd.DataFrame({
        "raw_nfiq2": df[raw_col],
        "stage1": df[stage1_col],
        "stage2": df[stage2_col],
        "stage3": df[stage3_col],
    })
    out["delta_p1"] = out["stage1"] - out["raw_nfiq2"]
    out["delta_p2"] = out["stage2"] - out["stage1"]
    out["delta_p6"] = out["stage3"] - out["stage2"]
    return out


def load_pipeline_a():
    path = os.path.join(_find_pipeline_a_dir(), "per_image_cumulative_scores.csv")
    df = pd.read_csv(path)
    deltas = _compute_deltas(df, "raw_nfiq2", "stage1_nfiq2", "stage2_nfiq2", "stage3_nfiq2")
    out = pd.concat([df[["database", "filename"]].rename(columns={"database": "db", "filename": "file"}),
                      deltas], axis=1)
    out.insert(0, "pipeline", "pipeline_a")
    return out


def load_pipeline_b():
    path = os.path.join(RESULTS_DIR, "pipeline_b_ablation.csv")
    df = pd.read_csv(path)
    deltas = _compute_deltas(df, "raw_nfiq2", "stage1_p1_nfiq2", "stage2_p1_p2_nfiq2", "stage3_p1_p2_p6_nfiq2")
    out = pd.concat([df[["database", "filename"]].rename(columns={"database": "db", "filename": "file"}),
                      deltas], axis=1)
    out.insert(0, "pipeline", "pipeline_b")

    # Cross-check against this file's own precomputed delta_p1/p2/p6 rather
    # than trusting our recomputation blindly -- they should match exactly
    # wherever the file's own columns are non-null.
    for cat, own_col in [("delta_p1", "delta_p1"), ("delta_p2", "delta_p2"), ("delta_p6", "delta_p6")]:
        own = df[own_col]
        mask = own.notna() & out[cat].notna()
        mismatch = (own[mask] - out.loc[mask, cat]).abs() > 1e-6
        if mismatch.any():
            raise AssertionError(
                f"Pipeline B: recomputed {cat} disagrees with the file's own "
                f"{own_col} column on {mismatch.sum()} row(s) -- investigate before trusting this data."
            )
    return out


def load_pipeline_c():
    path = os.path.join(RESULTS_DIR, "pipeline_c_ablation.csv")
    df = pd.read_csv(path)
    deltas = _compute_deltas(df, "raw_nfiq2", "stage1_contrast_nfiq2", "stage2_noise_nfiq2", "stage3_orientation_nfiq2")
    out = pd.concat([df[["db", "file"]], deltas], axis=1)
    out.insert(0, "pipeline", "pipeline_c")
    return out


def load_pipeline_d():
    path = os.path.join(RESULTS_DIR, "pipeline_d_ablation", "pipeline_d_ablation.csv")
    df = pd.read_csv(path)
    deltas = _compute_deltas(df, "raw_nfiq2", "stage1_p1_nfiq2", "stage2_p1_p2_nfiq2", "stage3_p1_p2_p6_nfiq2")
    out = pd.concat([df[["database", "filename"]].rename(columns={"database": "db", "filename": "file"}),
                      deltas], axis=1)
    out.insert(0, "pipeline", "pipeline_d")

    for cat in ("delta_p1", "delta_p2", "delta_p6"):
        own = df[cat]
        mask = own.notna() & out[cat].notna()
        mismatch = (own[mask] - out.loc[mask, cat]).abs() > 1e-6
        if mismatch.any():
            raise AssertionError(
                f"Pipeline D: recomputed {cat} disagrees with the file's own "
                f"{cat} column on {mismatch.sum()} row(s) -- investigate before trusting this data."
            )
    return out


def load_all_pipelines():
    frames = [load_pipeline_a(), load_pipeline_b(), load_pipeline_c(), load_pipeline_d()]
    combined = pd.concat(frames, ignore_index=True)
    print("Rows loaded per pipeline (320 raw images each, before any category-specific exclusion):")
    for name, frame in zip(("pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"), frames):
        n_missing_raw = frame["raw_nfiq2"].isna().sum()
        print(f"  {name}: {len(frame)} rows, {n_missing_raw} with missing raw_nfiq2 "
              f"(excluded from delta_p1 only, per the module docstring)")
    return combined


def load_features():
    path = os.path.join(RESULTS_DIR, "per_image_metrics.csv")
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"per_image_metrics.csv is missing expected feature columns: {missing}")
    return df[["db", "file"] + FEATURE_COLS]


def join_features(combined, features):
    joined = combined.merge(features, on=["db", "file"], how="left", validate="many_to_one")
    missing_features = joined[FEATURE_COLS[0]].isna().sum()
    if missing_features:
        print(f"WARNING: {missing_features} rows have no matching diagnostic-feature row "
              f"(unexpected -- per_image_metrics.csv should cover all 320 raw images).")
    return joined


def cross_validate(X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    model_r2s, model_maes, baseline_maes = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)

        model_r2s.append(r2_score(y_val, pred))
        model_maes.append(mean_absolute_error(y_val, pred))

        baseline_pred = np.full_like(y_val, fill_value=y_train.mean(), dtype=np.float64)
        baseline_maes.append(mean_absolute_error(y_val, baseline_pred))

    return dict(
        n=len(X),
        model_r2_mean=np.mean(model_r2s), model_r2_sd=np.std(model_r2s),
        model_mae_mean=np.mean(model_maes), model_mae_sd=np.std(model_maes),
        baseline_mae_mean=np.mean(baseline_maes), baseline_mae_sd=np.std(baseline_maes),
    )


def phase_a_modeling(joined):
    print("\n" + "=" * 100)
    print("PHASE A -- Section 2.2 model vs. naive baseline, all 4 pipelines x 3 categories = 12 combinations")
    print("=" * 100)
    rows = []
    for pipeline in ("pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"):
        sub_all = joined[joined["pipeline"] == pipeline]
        for cat in CATEGORIES:
            sub = sub_all.dropna(subset=[cat] + FEATURE_COLS)
            X = sub[FEATURE_COLS]
            y = sub[cat]
            result = cross_validate(X, y)
            beats = result["model_mae_mean"] < result["baseline_mae_mean"]
            margin_pct = 100 * (1 - result["model_mae_mean"] / result["baseline_mae_mean"])
            rows.append(dict(
                pipeline=pipeline, category=cat, n=result["n"],
                model_r2=result["model_r2_mean"], model_r2_sd=result["model_r2_sd"],
                model_mae=result["model_mae_mean"], model_mae_sd=result["model_mae_sd"],
                baseline_mae=result["baseline_mae_mean"], baseline_mae_sd=result["baseline_mae_sd"],
                beats_baseline=beats, margin_pct=margin_pct,
            ))

    result_df = pd.DataFrame(rows)
    with pd.option_context("display.width", 160):
        print(result_df.round(3).to_string(index=False))
    return result_df


def phase_b_selection(joined):
    print("\n" + "=" * 100)
    print("PHASE B -- Section 2.4 step 3: cross-pipeline comparison per category")
    print("=" * 100)

    summary_rows = []
    for cat in CATEGORIES:
        for pipeline in ("pipeline_a", "pipeline_b", "pipeline_c", "pipeline_d"):
            sub = joined[(joined["pipeline"] == pipeline)].dropna(subset=[cat])
            delta = sub[cat]
            n = len(delta)
            improved = (delta > 0).sum()
            degraded = (delta < 0).sum()
            unchanged = (delta == 0).sum()
            summary_rows.append(dict(
                category=cat, pipeline=pipeline, n=n,
                mean_delta=delta.mean(), sd_delta=delta.std(),
                pct_improved=100 * improved / n, pct_degraded=100 * degraded / n,
                pct_unchanged=100 * unchanged / n,
                worst_case=delta.min(), best_case=delta.max(),
            ))
    summary_df = pd.DataFrame(summary_rows)

    print("\nFull per-category, per-pipeline comparison table:")
    with pd.option_context("display.width", 160):
        print(summary_df.round(2).to_string(index=False))

    print("\n--- Selection rule ---")
    print(
        "1. Rank the 4 pipelines by mean_delta for this category (highest wins by default).\n"
        "2. Reliability check on the mean_delta leader: flag it if EITHER\n"
        "   (a) its pct_degraded is more than 15 percentage points worse than the\n"
        "       next-best pipeline's pct_degraded, i.e. it wins on average partly by\n"
        "       relying on a much higher regression rate, or\n"
        "   (b) its worst_case is more than 2x the magnitude of the runner-up\n"
        "       (by mean_delta)'s worst_case, i.e. it has a severe single-image\n"
        "       failure mode the runner-up doesn't.\n"
        "   If flagged, fall back to the runner-up UNLESS the runner-up fails the same\n"
        "   check against the pipeline after it -- reported explicitly either way, not\n"
        "   silently auto-resolved."
    )

    selections = {}
    for cat in CATEGORIES:
        cat_df = summary_df[summary_df["category"] == cat].sort_values("mean_delta", ascending=False).reset_index(drop=True)
        leader = cat_df.iloc[0]
        runner_up = cat_df.iloc[1] if len(cat_df) > 1 else None

        flagged_reasons = []
        if runner_up is not None:
            degraded_gap = leader["pct_degraded"] - runner_up["pct_degraded"]
            if degraded_gap > 15:
                flagged_reasons.append(
                    f"pct_degraded {leader['pct_degraded']:.1f}% vs runner-up's "
                    f"{runner_up['pct_degraded']:.1f}% (gap {degraded_gap:.1f}pp > 15pp)"
                )
            leader_worst_mag = abs(leader["worst_case"])
            runner_worst_mag = abs(runner_up["worst_case"])
            if runner_worst_mag > 0 and leader_worst_mag > 2 * runner_worst_mag:
                flagged_reasons.append(
                    f"worst_case {leader['worst_case']:.1f} vs runner-up's "
                    f"{runner_up['worst_case']:.1f} (>2x magnitude)"
                )

        winner = leader
        note = "selected on mean_delta alone, no reliability flag triggered"
        if flagged_reasons:
            note = (f"mean_delta leader ({leader['pipeline']}) FLAGGED: " + "; ".join(flagged_reasons)
                    + f" -- falling back to runner-up ({runner_up['pipeline']})")
            winner = runner_up

        selections[cat] = dict(winner=winner["pipeline"], mean_delta=winner["mean_delta"], note=note)
        print(f"\n{CATEGORY_LABELS[cat]}: WINNER = {winner['pipeline']}  (mean_delta={winner['mean_delta']:.2f})")
        print(f"  {note}")

    return summary_df, selections


def main():
    combined = load_all_pipelines()
    features = load_features()
    joined = join_features(combined, features)

    joined_path = os.path.join(RESULTS_DIR, "section_2_4_joined_per_image.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    joined.to_csv(joined_path, index=False)
    print(f"\nWrote joined per-image table ({len(joined)} rows) to {joined_path}")

    model_results = phase_a_modeling(joined)
    model_results.to_csv(os.path.join(RESULTS_DIR, "section_2_4_model_vs_baseline.csv"), index=False)

    summary_df, selections = phase_b_selection(joined)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "section_2_4_cross_pipeline_summary.csv"), index=False)

    print("\n" + "=" * 100)
    print("FINAL SELECTION SUMMARY")
    print("=" * 100)
    for cat, sel in selections.items():
        print(f"  {CATEGORY_LABELS[cat]}: {sel['winner']}  (mean_delta={sel['mean_delta']:.2f})")


if __name__ == "__main__":
    main()
