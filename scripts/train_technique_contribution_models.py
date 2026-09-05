"""
SUPERSEDED (real data for all 4 pipelines now exists) -- see
scripts/section_2_4_modeling.py and docs/section_2_4_findings.md for the
real, 4-pipeline x 3-category version of this analysis. Kept as-is
(unmodified) for history; not deleted, not re-run.

Methodology dry-run: can the 7 diagnostic degradation features
(results/per_image_metrics.csv) predict how much NFIQ2 each of Pipeline C's
three techniques contributes per image (results/pipeline_c_ablation.csv's
delta_p1/p2/p6)? This is a signal-check on Pipeline C alone, meant to be
re-applied unchanged once Pipelines A/B/D have equivalent ablation data --
not the final 4-pipeline analysis.

For each of the 3 targets (delta_p1, delta_p2, delta_p6):
  - 5-fold CV RandomForestRegressor: mean R^2 and mean MAE across folds.
  - A naive per-fold "always predict the training-fold mean" baseline MAE,
    reported alongside the model MAE -- the model must clearly beat this to
    count as finding real signal, not just fitting noise in a 319-row set.
  - Feature importances from a model fit on the full 319 rows (only
    reported/discussed for targets that beat the baseline).
  - Pearson r between each of the 7 features and the target, as a simpler
    cross-check against the RF importances.

Run with:
    python scripts/train_technique_contribution_models.py
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from config import RESULTS_DIR  # noqa: E402

FEATURE_COLS = [
    "contrast_std",
    "michelson_contrast",
    "illumination_unevenness",
    "noise_sigma",
    "sharpness_laplacian_var",
    "foreground_ratio",
    "orientation_coherence",
]

TARGETS = {
    "delta_p1": "homomorphic filtering (P1 / contrast)",
    "delta_p2": "coherence diffusion (P2 / noise)",
    "delta_p6": "Log-Gabor filtering (P6 / orientation)",
}

RANDOM_STATE = 42
N_SPLITS = 5


def load_joined():
    metrics = pd.read_csv(os.path.join(RESULTS_DIR, "per_image_metrics.csv"))
    ablation = pd.read_csv(os.path.join(RESULTS_DIR, "pipeline_c_ablation.csv"))

    df = metrics.merge(ablation, on=["file", "db"], how="inner", validate="one_to_one")
    before = len(df)
    df = df.dropna(subset=["raw_nfiq2"]).copy()
    print(f"Joined {len(metrics)} x {len(ablation)} rows -> {before} matched, "
          f"{len(df)} usable after dropping rows with no raw_nfiq2 "
          f"({before - len(df)} excluded, e.g. DB3_B/110_5.tif).")

    df["delta_p1"] = df["stage1_contrast_nfiq2"] - df["raw_nfiq2"]
    df["delta_p2"] = df["stage2_noise_nfiq2"] - df["stage1_contrast_nfiq2"]
    df["delta_p6"] = df["stage3_orientation_nfiq2"] - df["stage2_noise_nfiq2"]
    return df


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

    return {
        "model_r2_mean": np.mean(model_r2s),
        "model_r2_sd": np.std(model_r2s),
        "model_mae_mean": np.mean(model_maes),
        "model_mae_sd": np.std(model_maes),
        "baseline_mae_mean": np.mean(baseline_maes),
        "baseline_mae_sd": np.std(baseline_maes),
    }


def main():
    df = load_joined()
    X = df[FEATURE_COLS]

    print(f"\n{'target':<10} {'label':<32} {'model R2':>10} {'model MAE':>11} {'baseline MAE':>13}  beats_baseline")
    results = {}
    for target, label in TARGETS.items():
        y = df[target]
        cv = cross_validate(X, y)
        results[target] = cv
        beats = cv["model_mae_mean"] < cv["baseline_mae_mean"]
        margin_pct = 100 * (1 - cv["model_mae_mean"] / cv["baseline_mae_mean"])
        print(f"{target:<10} {label:<32} {cv['model_r2_mean']:>10.3f} "
              f"{cv['model_mae_mean']:>11.3f} {cv['baseline_mae_mean']:>13.3f}  "
              f"{'YES' if beats else 'NO'} ({margin_pct:+.1f}% vs baseline)")

    print("\n(model_r2/mae shown as mean across 5 folds; SD below)")
    for target, cv in results.items():
        print(f"  {target}: R2 sd={cv['model_r2_sd']:.3f}  model_mae sd={cv['model_mae_sd']:.3f}  "
              f"baseline_mae sd={cv['baseline_mae_sd']:.3f}")

    # Pearson correlations (all targets, all features) -- simpler cross-check
    print("\n=== Pearson r: each feature vs. each target ===")
    corr_rows = []
    for target in TARGETS:
        row = {"target": target}
        for feat in FEATURE_COLS:
            r, p = pearsonr(df[feat], df[target])
            row[feat] = r
        corr_rows.append(row)
    corr_df = pd.DataFrame(corr_rows).set_index("target")
    print(corr_df.round(3).to_string())

    # Feature importances -- full-data fit, only meaningfully interpreted for
    # targets that clearly beat the naive baseline (see printed verdicts above).
    print("\n=== Feature importances (RF fit on all 319 rows, one model per target) ===")
    importance_rows = []
    for target in TARGETS:
        y = df[target]
        model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE)
        model.fit(X, y)
        row = {"target": target}
        for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
            row[feat] = imp
        importance_rows.append(row)
    importance_df = pd.DataFrame(importance_rows).set_index("target")
    print(importance_df.round(3).to_string())


if __name__ == "__main__":
    main()
