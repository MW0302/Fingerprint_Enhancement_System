"""
READ-ONLY diagnostic: quantifies how widespread the normalize_image() edge
case Member A found on DB1_B/101_4.tif is -- an image whose raw std sits
just under the pass-through threshold (target_std=40) still goes through the
full mean-shift-and-rescale path, which can crush a naturally high raw mean
toward target_mean=100 even though its contrast was already close to fine.

Does NOT modify common.py or any pipeline file -- imports and calls
normalize_image() directly, does not reimplement its logic.

Confirmed logic (src/utils/common.py, normalize_image()):
    target_mean=100.0, target_var=1600.0 (i.e. target_std=40.0)
    if img.var() >= target_var (i.e. raw_std >= 40): PASS THROUGH, untouched
    else: full rescale -- every pixel's deviation from the image's own mean
          is rescaled to match target_var, then recentred to target_mean=100

Run with:
    python scripts/check_normalization_edge_cases.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from common import normalize_image  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402

TARGET_STD = 40.0  # sqrt(target_var=1600.0), normalize_image()'s own default

# Danger-zone definition (stated explicitly, not just applied silently):
# raw_std BELOW the pass-through threshold but within 15 of it (i.e. the
# image was close enough to "already fine" that full-strength rescaling is
# most likely overkill), combined with a large resulting mean shift.
DANGER_STD_LOW = TARGET_STD - 15.0   # 25.0
DANGER_STD_HIGH = TARGET_STD          # 40.0 (exclusive -- pass-through starts here)
DANGER_DELTA_MEAN_ABS = 40.0


def main():
    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"WARNING: could not read {path}, skipping")
                continue

            raw_mean = float(img.mean())
            raw_std = float(img.std())

            normalized = normalize_image(img)  # actual function, defaults as used by every pipeline
            norm_mean = float(normalized.mean())
            norm_std = float(normalized.std())

            pass_through = raw_std >= TARGET_STD
            delta_mean = raw_mean - norm_mean

            rows.append(dict(
                file=os.path.basename(path),
                db=db,
                raw_mean=raw_mean,
                raw_std=raw_std,
                norm_mean=norm_mean,
                norm_std=norm_std,
                delta_mean=delta_mean,
                pass_through=pass_through,
            ))
        print(f"{db}: {len(paths)} images processed")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "normalization_check.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    # --- cross-check raw_std against per_image_metrics.csv's contrast_std ---
    metrics_path = os.path.join(RESULTS_DIR, "per_image_metrics.csv")
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        merged = df.merge(metrics[["file", "db", "contrast_std"]], on=["file", "db"], how="inner")
        merged["cross_check_diff"] = (merged["raw_std"] - merged["contrast_std"]).abs()
        print(f"\n=== Cross-check: raw_std (this script) vs. contrast_std (per_image_metrics.csv) ===")
        print(f"n compared: {len(merged)}  max abs diff: {merged['cross_check_diff'].max():.6f}  "
              f"mean abs diff: {merged['cross_check_diff'].mean():.6f}")
        sample = merged.sample(n=min(5, len(merged)), random_state=1)[
            ["file", "db", "raw_std", "contrast_std", "cross_check_diff"]
        ]
        print(sample.to_string(index=False))

    # --- distribution of delta_mean among NON-pass-through images, so the
    # danger-zone cutoff (|delta_mean| > 40) can be judged, not just applied ---
    not_passed = df[~df["pass_through"]]
    print(f"\n=== delta_mean distribution among {len(not_passed)} non-pass-through images "
          f"(all 4 DBs) ===")
    print(not_passed["delta_mean"].describe().to_string())
    print("\nPercentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"  p{p}: {np.percentile(not_passed['delta_mean'].abs(), p):.2f}")
    print("\nHistogram of |delta_mean| among non-pass-through images (10 bins):")
    counts, edges = np.histogram(not_passed["delta_mean"].abs(), bins=10)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        print(f"  [{lo:6.2f}, {hi:6.2f}): {c}")

    # --- danger zone: raw_std in [25, 40) AND |delta_mean| > 40 ---
    danger = df[
        (df["raw_std"] >= DANGER_STD_LOW) & (df["raw_std"] < DANGER_STD_HIGH)
        & (df["delta_mean"].abs() > DANGER_DELTA_MEAN_ABS)
    ]
    print(f"\n=== Danger zone: raw_std in [{DANGER_STD_LOW:.0f}, {DANGER_STD_HIGH:.0f}) "
          f"AND |delta_mean| > {DANGER_DELTA_MEAN_ABS:.0f} ===")
    print(f"Total: {len(danger)} / {len(df)} images ({100 * len(danger) / len(df):.1f}%)")
    print("\nPer-DB breakdown:")
    for db in DBS:
        db_total = (df["db"] == db).sum()
        db_danger = ((danger["db"] == db)).sum()
        print(f"  {db}: {db_danger} / {db_total} ({100 * db_danger / db_total:.1f}%)")

    if len(danger) > 0:
        print("\nDanger-zone images:")
        print(danger[["file", "db", "raw_mean", "raw_std", "norm_mean", "norm_std", "delta_mean"]]
              .sort_values("delta_mean", key=lambda s: s.abs(), ascending=False)
              .to_string(index=False))

    # --- worst 5 cases dataset-wide by |delta_mean| among non-pass-through images ---
    worst5 = not_passed.reindex(
        not_passed["delta_mean"].abs().sort_values(ascending=False).index
    ).head(5)
    print(f"\n=== Worst 5 cases dataset-wide by |delta_mean| (non-pass-through only) ===")
    print(worst5[["file", "db", "raw_mean", "raw_std", "norm_mean", "norm_std", "delta_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
