"""
VALIDATION ONLY -- does not modify common.py or any pipeline file.

Round 2 of the normalize_image() fix. Round 1 (normalize_image_v2, see
scripts/validate_normalize_image_v2.py) correctly killed the mean-crush bug
by recentring on the image's own mean instead of a fixed target_mean=100,
but that introduced catastrophic clipping (80-86% of pixels) on DB1's
near-ceiling-brightness images, because it applied the full std=40 boost
regardless of how much headroom to 0/255 was actually available around that
image's own (near-white) mean.

This round (normalize_image_v3) adds a headroom-aware cap: still recentres
on the image's own mean, but the scale factor is capped at whatever the
image's own 1st/99th percentile spread can safely support before hitting
the 0/255 boundary (with a small 2/253 safety margin, not a hard wall) --
same robust-percentile idea already used in pipeline_c.py's
_homomorphic_filter to avoid a few outlier pixels distorting a rescale.

Run with:
    python scripts/validate_normalize_image_v3.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
from common import normalize_image  # noqa: E402  -- the REAL, current function
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402

TARGET_VAR = 1600.0
TARGET_STD = 40.0
WORST5 = ["109_6.tif", "109_8.tif", "101_5.tif", "101_4.tif", "109_3.tif"]  # all DB1_B
# The 9 DB1 images that broke under round-1's v2 (60-85% clipping)
V2_BROKEN_DB1 = [
    "109_6.tif", "109_8.tif", "101_5.tif", "101_4.tif",
    "110_1.tif", "110_3.tif", "110_4.tif", "110_6.tif", "110_8.tif",
]


def normalize_image_v2(img, target_var=1600.0):
    """Round-1 candidate, kept here only for a direct old/v2/v3 comparison."""
    img = img.astype(np.float64)
    mean = img.mean()
    var = img.var() + 1e-8
    if var >= target_var:
        return np.clip(img, 0, 255).astype(np.uint8)
    normalized = mean + np.sign(img - mean) * np.sqrt(
        target_var * (img - mean) ** 2 / var
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


def normalize_image_v3(img, target_var=1600.0, target_std=40.0):
    """
    Round-2 candidate: headroom-aware version of v2. Same recentre-on-own-
    mean idea, but the boost factor is capped so it never pushes the image's
    own robust 1st/99th percentile spread past [2, 253].
    """
    img = img.astype(np.float64)
    mu = img.mean()
    var = img.var() + 1e-8
    if var >= target_var:
        return np.clip(img, 0, 255).astype(np.uint8)

    raw_std = np.sqrt(var)
    p1, p99 = np.percentile(img, [1, 99])

    k_bright = (253.0 - mu) / (p99 - mu) if p99 > mu else np.inf
    k_dark = (mu - 2.0) / (mu - p1) if p1 < mu else np.inf
    k_headroom = min(k_bright, k_dark)

    k_target = target_std / raw_std
    k_final = min(k_target, k_headroom)

    normalized = mu + k_final * (img - mu)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def clip_fraction(out_img):
    return float(((out_img == 0) | (out_img == 255)).mean())


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

            old = normalize_image(img)
            v2 = normalize_image_v2(img)
            v3 = normalize_image_v3(img)

            rows.append(dict(
                file=os.path.basename(path),
                db=db,
                raw_mean=raw_mean, raw_std=raw_std,
                old_mean=float(old.mean()), old_std=float(old.std()), old_clip_frac=clip_fraction(old),
                v2_mean=float(v2.mean()), v2_std=float(v2.std()), v2_clip_frac=clip_fraction(v2),
                v3_mean=float(v3.mean()), v3_std=float(v3.std()), v3_clip_frac=clip_fraction(v3),
            ))
        print(f"{db}: {len(paths)} images processed")

    df = pd.DataFrame(rows)
    df["pass_through"] = df["raw_std"] >= TARGET_STD
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "normalize_v3_validation.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    # --- worst-5 old vs v2 vs v3 ---
    print("\n=== Worst-5: OLD vs. v2 (round 1) vs. v3 (round 2) ===")
    worst5_df = df[df["file"].isin(WORST5) & (df["db"] == "DB1_B")]
    cols = ["file", "raw_mean", "raw_std",
            "old_mean", "old_clip_frac",
            "v2_mean", "v2_clip_frac",
            "v3_mean", "v3_clip_frac"]
    print(worst5_df[cols].sort_values("raw_mean", ascending=False).to_string(index=False))

    # --- DB3 check: v3 vs OLD (should still be barely affected) ---
    print("\n=== DB3_B: OLD vs. v3 ===")
    db3 = df[df["db"] == "DB3_B"]
    mean_diff = (db3["old_mean"] - db3["v3_mean"]).abs()
    std_diff = (db3["old_std"] - db3["v3_std"]).abs()
    print(f"n={len(db3)}")
    print(f"  |old_mean - v3_mean|: mean={mean_diff.mean():.3f}  max={mean_diff.max():.3f}")
    print(f"  |old_std  - v3_std |: mean={std_diff.mean():.3f}  max={std_diff.max():.3f}")
    print(f"  old_clip_frac: mean={db3['old_clip_frac'].mean():.4f}  max={db3['old_clip_frac'].max():.4f}")
    print(f"  v3_clip_frac:  mean={db3['v3_clip_frac'].mean():.4f}  max={db3['v3_clip_frac'].max():.4f}")

    # --- full-dataset clipping-fraction summary, OLD vs v3 ---
    print("\n=== Full-dataset (all 320 images) clipping-fraction summary: OLD vs. v3 ===")
    print(df.groupby("db")[["old_clip_frac", "v3_clip_frac"]].agg(["mean", "max"]).to_string())
    print(f"\nOverall max old_clip_frac: {df['old_clip_frac'].max():.4f}  "
          f"(worst: {df.loc[df['old_clip_frac'].idxmax(), 'file']}, {df.loc[df['old_clip_frac'].idxmax(), 'db']})")
    print(f"Overall max v3_clip_frac:  {df['v3_clip_frac'].max():.4f}  "
          f"(worst: {df.loc[df['v3_clip_frac'].idxmax(), 'file']}, {df.loc[df['v3_clip_frac'].idxmax(), 'db']})")
    increased = df[df["v3_clip_frac"] > df["old_clip_frac"] + 1e-6]
    print(f"\nImages where v3_clip_frac > old_clip_frac: {len(increased)} / {len(df)}")
    if len(increased) > 0:
        print(increased[["file", "db", "old_clip_frac", "v3_clip_frac"]]
              .sort_values("v3_clip_frac", ascending=False)
              .to_string(index=False))

    # --- the 9 DB1 images that broke under v2: what happens under v3? ---
    print("\n=== The 9 DB1 images that broke under v2 (60-85% clip) -- v3 outcome ===")
    broken = df[df["file"].isin(V2_BROKEN_DB1) & (df["db"] == "DB1_B")]
    bcols = ["file", "raw_mean", "raw_std",
             "old_mean", "old_clip_frac",
             "v2_mean", "v2_clip_frac",
             "v3_mean", "v3_clip_frac"]
    bdf = broken[bcols].sort_values("raw_mean", ascending=False).copy()
    bdf["v3_delta_mean_vs_raw"] = broken["raw_mean"] - broken["v3_mean"]
    bdf["v3_stayed_near_old"] = (broken["v3_mean"] - broken["old_mean"]).abs() < 5.0
    print(bdf.to_string(index=False))


if __name__ == "__main__":
    main()
