"""Derive reproducible report statistics from Pipeline B's final CSV."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
STAGES = ("delta_p1", "delta_p2", "delta_p6", "delta_final")


def _describe(values):
    values = pd.Series(values).dropna().astype(float)
    margin = 1.96 * float(values.std(ddof=1)) / np.sqrt(len(values))
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "improved": int((values > 0).sum()),
        "regressed": int((values < 0).sum()),
        "unchanged": int((values == 0).sum()),
        "improved_fraction": float((values > 0).mean()),
        "approximate_95_ci_mean": [float(values.mean() - margin), float(values.mean() + margin)],
    }


def build_statistics(data):
    output = {"scopes": {}}
    for scope in (*sorted(data["database"].unique()), "OVERALL"):
        subset = data if scope == "OVERALL" else data[data["database"] == scope]
        paired = subset.dropna(subset=["raw_nfiq2", "stage3_p1_p2_p6_nfiq2"])
        output["scopes"][scope] = {
            "paired_n": int(len(paired)),
            "paired_raw_mean": float(paired["raw_nfiq2"].mean()),
            "paired_final_mean": float(paired["stage3_p1_p2_p6_nfiq2"].mean()),
            "deltas": {field: _describe(subset[field]) for field in STAGES},
        }
        print(f"Computed {scope}", flush=True)

    paired = data.dropna(subset=["raw_nfiq2", "delta_final"]).copy()
    print("Computing raw-quality relationship", flush=True)
    raw_values = paired["raw_nfiq2"].to_numpy(dtype=np.float64)
    delta_values = paired["delta_final"].to_numpy(dtype=np.float64)

    def correlation(left, right):
        left = left - left.mean()
        right = right - right.mean()
        return float(np.sum(left * right) / np.sqrt(np.sum(left * left) * np.sum(right * right)))

    output["raw_quality_relationship"] = {
        "pearson_raw_vs_final_delta": correlation(raw_values, delta_values),
        # Spearman correlation is Pearson correlation of average ranks. This
        # explicit form avoids an optional SciPy dependency in the report tool.
        "spearman_raw_vs_final_delta": correlation(
            paired["raw_nfiq2"].rank(method="average").to_numpy(dtype=np.float64),
            paired["delta_final"].rank(method="average").to_numpy(dtype=np.float64),
        ),
    }
    paired["raw_quality_quartile"] = pd.qcut(
        paired["raw_nfiq2"], 4, duplicates="drop"
    ).astype(str)
    quartiles = []
    for label, group in paired.groupby("raw_quality_quartile", sort=False):
        quartiles.append({
            "raw_score_interval": label,
            "n": int(len(group)),
            "mean_raw": float(group["raw_nfiq2"].mean()),
            "mean_final_delta": float(group["delta_final"].mean()),
            "improved_fraction": float((group["delta_final"] > 0).mean()),
        })
    output["raw_quality_relationship"]["quartiles"] = quartiles
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "pipeline_b_ablation.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "pipeline_b_report_statistics.json")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    statistics = build_statistics(data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
