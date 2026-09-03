"""
VALIDATION ONLY -- does not modify common.py or any pipeline file.

Tests a candidate fix for normalize_image()'s mean-crushing bug found in the
previous scan (results/normalization_check.csv): 89/320 images (up to 75% of
DB4_B) had their brightness forcibly recentred toward a fixed
target_mean=100.0 whenever their variance was below target_var, regardless
of how far their own natural brightness already was from 100 -- even though
only their CONTRAST was actually deficient.

Candidate (normalize_image_v2 below): identical variance-boosting math to
the real normalize_image() (imported, not reimplemented, for the baseline
comparison) -- the ONLY change is that rescaled deviations are re-added onto
the image's OWN raw mean instead of the fixed target_mean=100.0. The
pass-through condition (var >= target_var -> untouched) is unchanged.

Run with:
    python scripts/validate_normalize_image_v2.py
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
WORST5 = ["109_6.tif", "109_8.tif", "101_5.tif", "101_4.tif", "109_3.tif"]  # all DB1_B


def normalize_image_v2(img, target_var=1600.0):
    """
    CANDIDATE fix, NOT applied to common.py -- validation only.
    Same variance-boost math as normalize_image(); deviations are re-added
    onto the image's own raw mean rather than a fixed target_mean=100.0.
    """
    img = img.astype(np.float64)
    mean = img.mean()
    var = img.var() + 1e-8
    if var >= target_var:
        return np.clip(img, 0, 255).astype(np.uint8)
    normalized = mean + np.sign(img - mean) * np.sqrt(
        target_var * (img - mean) ** 2 / var
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


def clip_fraction(out_img):
    """Fraction of output pixels sitting exactly at 0 or 255 -- the same
    boundary-hit proxy for 'got clipped' already used elsewhere in this
    project's own normalize_image() debugging history (see common.py's
    docstring: 'checking clip-fraction ... 0% of pixels clipped ... vs
    8-17% under the first attempt')."""
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

            old = normalize_image(img)       # real, current common.py function
            new = normalize_image_v2(img)    # candidate

            rows.append(dict(
                file=os.path.basename(path),
                db=db,
                raw_mean=raw_mean,
                raw_std=raw_std,
                old_mean=float(old.mean()), old_std=float(old.std()), old_clip_frac=clip_fraction(old),
                new_mean=float(new.mean()), new_std=float(new.std()), new_clip_frac=clip_fraction(new),
            ))
        print(f"{db}: {len(paths)} images processed")

    df = pd.DataFrame(rows)
    df["pass_through"] = df["raw_std"] >= 40.0
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "normalize_v2_validation.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    # --- worst-5 before/after ---
    print("\n=== Worst-5 (from the previous scan) before/after ===")
    worst5_df = df[df["file"].isin(WORST5) & (df["db"] == "DB1_B")]
    cols = ["file", "db", "raw_mean", "raw_std", "old_mean", "old_std", "old_clip_frac",
            "new_mean", "new_std", "new_clip_frac"]
    print(worst5_df[cols].sort_values("raw_mean", ascending=False).to_string(index=False))

    # --- DB3 unaffected check ---
    print("\n=== DB3_B: old vs. new should be nearly identical (raw_mean already close to 100) ===")
    db3 = df[df["db"] == "DB3_B"]
    mean_diff = (db3["old_mean"] - db3["new_mean"]).abs()
    std_diff = (db3["old_std"] - db3["new_std"]).abs()
    print(f"n={len(db3)}")
    print(f"  |old_mean - new_mean|: mean={mean_diff.mean():.3f}  max={mean_diff.max():.3f}")
    print(f"  |old_std  - new_std |: mean={std_diff.mean():.3f}  max={std_diff.max():.3f}")
    worst_db3 = db3.loc[mean_diff.sort_values(ascending=False).index[:3]]
    print("\n  3 largest DB3 old-vs-new mean differences (should still be small):")
    print(worst_db3[cols].to_string(index=False))

    # --- danger-zone resolution check (89 flagged images from the prior scan) ---
    danger = df[
        (df["raw_std"] >= 25.0) & (df["raw_std"] < 40.0)
        & ((df["raw_mean"] - df["old_mean"]).abs() > 40.0)
    ]
    print(f"\n=== Previously-flagged danger-zone images (n={len(danger)}): "
          f"old |raw_mean-mean| vs. new |raw_mean-mean| ===")
    danger = danger.copy()
    danger["old_abs_delta_mean"] = (danger["raw_mean"] - danger["old_mean"]).abs()
    danger["new_abs_delta_mean"] = (danger["raw_mean"] - danger["new_mean"]).abs()
    print(f"  old: mean={danger['old_abs_delta_mean'].mean():.2f}  max={danger['old_abs_delta_mean'].max():.2f}")
    print(f"  new: mean={danger['new_abs_delta_mean'].mean():.2f}  max={danger['new_abs_delta_mean'].max():.2f}")
    print("\n  Per-DB breakdown of danger-zone images, old vs new |delta_mean|:")
    print(danger.groupby("db")[["old_abs_delta_mean", "new_abs_delta_mean"]].mean().to_string())

    # --- full-dataset clipping-fraction summary (make sure v2 doesn't introduce NEW clipping) ---
    print("\n=== Full-dataset (all 320 images) clipping-fraction summary ===")
    print(df.groupby("db")[["old_clip_frac", "new_clip_frac"]].agg(["mean", "max"]).to_string())
    print(f"\nOverall max old_clip_frac: {df['old_clip_frac'].max():.4f}  "
          f"(worst file: {df.loc[df['old_clip_frac'].idxmax(), 'file']}, "
          f"{df.loc[df['old_clip_frac'].idxmax(), 'db']})")
    print(f"Overall max new_clip_frac: {df['new_clip_frac'].max():.4f}  "
          f"(worst file: {df.loc[df['new_clip_frac'].idxmax(), 'file']}, "
          f"{df.loc[df['new_clip_frac'].idxmax(), 'db']})")
    increased = df[df["new_clip_frac"] > df["old_clip_frac"] + 1e-6]
    print(f"\nImages where new_clip_frac > old_clip_frac (candidate introduced MORE clipping): "
          f"{len(increased)} / {len(df)}")
    if len(increased) > 0:
        print(increased[["file", "db", "old_clip_frac", "new_clip_frac"]].to_string(index=False))


if __name__ == "__main__":
    main()
