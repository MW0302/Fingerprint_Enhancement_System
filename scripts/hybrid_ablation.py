"""
Full-320-image validation for the fixed Hybrid pipeline (Section 2.4 step
5 / Handover Notes Section 17): the complete hybrid's cumulative ablation
(raw -> +P1 -> +P1+P2 -> +P1+P2+P6), plus the three hybrid-minus-one-
component variants, all scored with real NFIQ2 runs.

Does NOT modify src/hybrid/hybrid.py or any pipeline_*.py -- only calls
hybrid.py's own separately-callable stage functions, composed differently
per variant:
    full        : stage1(normalized) -> stage2(stage1) -> stage3(stage2)
    minus_p1    : stage2(normalized) -> stage3(that)      [P1 skipped]
    minus_p2    : stage3(stage1)                          [P2 skipped]
    minus_p6    : stage2(stage1) is itself the final output [P6 skipped]

Computation is reused across variants wherever the inputs are identical,
rather than recomputed 4 separate times per image:
    - Step 0 (normalize+segment): shared by all 4 variants.
    - Stage 1 output: identical for full/minus_p2 (both start from real
      Stage 1); ALSO reused as minus_p6's final Stage 2 input.
    - Stage 2 (full): identical to minus_p6's final output -- computed
      once, its NFIQ2 score is reused directly as minus_p6's final score.
    - Stage 2 (minus_p1): a genuinely separate diffusion run (different
      input image), computed once.
    - Stage 3 is computed 3 times (full, minus_p1, minus_p2) since each
      one's input differs.
Missing-value handling: DB3_B/110_5.tif (the one image NFIQ2 can't score
raw) gets a blank raw_nfiq2 and the real NFIQ2 error text in raw_error,
matching every other pipeline's own ablation file -- never zero-filled,
never dropped from stage scores that don't depend on raw.

Outputs:
    results/hybrid_ablation.csv          full hybrid, per-image, all 4 stages
    results/hybrid_ablation_summary.csv  full hybrid, per-DB + overall
    results/hybrid_ablation_variants.csv the 3 minus-one-component variants,
                                          per-image, one `variant` column
    results/hybrid_ablation_variants_summary.csv  same, per-DB + overall,
                                          plus the full hybrid's own final
                                          score for direct side-by-side comparison

Run with:
    python scripts/hybrid_ablation.py
"""

import os
import sys
import glob
import tempfile

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "hybrid"))

from common import run_nfiq2_single  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402
import hybrid  # noqa: E402


def score(img):
    tmp_path = os.path.join(tempfile.gettempdir(), f"hybrid_ablation_tmp_{os.getpid()}.tif")
    cv2.imwrite(tmp_path, img)
    s, err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return s, err


def process_image(path, db, fname):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None

    raw_score, raw_err = score(image)

    normalized, fg_mask_blocks = hybrid.stage0_preprocess(image)

    # --- full hybrid ---
    stage1 = hybrid.stage1_contrast(normalized, fg_mask_blocks)
    stage1_score, stage1_err = score(stage1)

    stage2_full, _alpha_full = hybrid.stage2_noise(normalized, stage1, fg_mask_blocks)
    stage2_full_score, stage2_full_err = score(stage2_full)

    stage3_full = hybrid.stage3_orientation(stage2_full, fg_mask_blocks)
    stage3_full_score, stage3_full_err = score(stage3_full)

    full_row = dict(
        db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        stage1_nfiq2=stage1_score, stage1_error=stage1_err,
        stage2_nfiq2=stage2_full_score, stage2_error=stage2_full_err,
        stage3_nfiq2=stage3_full_score, stage3_error=stage3_full_err,
    )

    # --- minus_p6: final IS stage2_full, already computed above ---
    minus_p6_row = dict(
        variant="minus_p6", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage2_full_score, final_error=stage2_full_err,
    )

    # --- minus_p2: stage3 applied directly to stage1 (P2 skipped) ---
    stage3_minus_p2 = hybrid.stage3_orientation(stage1, fg_mask_blocks)
    stage3_minus_p2_score, stage3_minus_p2_err = score(stage3_minus_p2)
    minus_p2_row = dict(
        variant="minus_p2", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_minus_p2_score, final_error=stage3_minus_p2_err,
    )

    # --- minus_p1: stage2 applied directly to normalized (P1 skipped),
    # then stage3 on that ---
    stage2_minus_p1, _alpha_minus_p1 = hybrid.stage2_noise(normalized, normalized, fg_mask_blocks)
    stage3_minus_p1 = hybrid.stage3_orientation(stage2_minus_p1, fg_mask_blocks)
    stage3_minus_p1_score, stage3_minus_p1_err = score(stage3_minus_p1)
    minus_p1_row = dict(
        variant="minus_p1", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_minus_p1_score, final_error=stage3_minus_p1_err,
    )

    return full_row, [minus_p1_row, minus_p2_row, minus_p6_row]


def main():
    full_rows = []
    variant_rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            full_row, variants = process_image(path, db, fname)
            if full_row is None:
                print(f"WARNING: could not read {path}, skipping")
                continue
            full_rows.append(full_row)
            variant_rows.extend(variants)
        print(f"{db}: {len(paths)} images done", flush=True)

    full_df = pd.DataFrame(full_rows)
    full_df["delta_p1"] = full_df["stage1_nfiq2"] - full_df["raw_nfiq2"]
    full_df["delta_p2"] = full_df["stage2_nfiq2"] - full_df["stage1_nfiq2"]
    full_df["delta_p6"] = full_df["stage3_nfiq2"] - full_df["stage2_nfiq2"]
    full_df["delta_final"] = full_df["stage3_nfiq2"] - full_df["raw_nfiq2"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_path = os.path.join(RESULTS_DIR, "hybrid_ablation.csv")
    full_df.to_csv(full_path, index=False)
    print(f"\nWrote {len(full_df)} rows to {full_path}")

    valid_final = full_df.dropna(subset=["raw_nfiq2", "stage3_nfiq2"])
    summary_rows = []
    for db in DBS:
        sub = valid_final[valid_final["db"] == db]
        summary_rows.append(dict(
            db=db, n=len(sub),
            raw_mean=sub["raw_nfiq2"].mean(),
            final_mean=sub["stage3_nfiq2"].mean(),
            delta_final_mean=sub["delta_final"].mean(),
            improved=(sub["delta_final"] > 0).sum(),
            regressed=(sub["delta_final"] < 0).sum(),
            unchanged=(sub["delta_final"] == 0).sum(),
        ))
    overall = dict(
        db="OVERALL", n=len(valid_final),
        raw_mean=valid_final["raw_nfiq2"].mean(),
        final_mean=valid_final["stage3_nfiq2"].mean(),
        delta_final_mean=valid_final["delta_final"].mean(),
        improved=(valid_final["delta_final"] > 0).sum(),
        regressed=(valid_final["delta_final"] < 0).sum(),
        unchanged=(valid_final["delta_final"] == 0).sum(),
    )
    summary_rows.append(overall)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "hybrid_ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n=== Full hybrid summary ===")
    print(summary_df.round(2).to_string(index=False))

    variant_df = pd.DataFrame(variant_rows)
    variant_df["delta_final"] = variant_df["final_nfiq2"] - variant_df["raw_nfiq2"]
    variant_path = os.path.join(RESULTS_DIR, "hybrid_ablation_variants.csv")
    variant_df.to_csv(variant_path, index=False)
    print(f"\nWrote {len(variant_df)} rows to {variant_path}")

    variant_summary_rows = []
    for variant in ("minus_p1", "minus_p2", "minus_p6"):
        for db in DBS:
            sub = variant_df[(variant_df["variant"] == variant) & (variant_df["db"] == db)]
            sub = sub.dropna(subset=["raw_nfiq2", "final_nfiq2"])
            variant_summary_rows.append(dict(
                variant=variant, db=db, n=len(sub),
                final_mean=sub["final_nfiq2"].mean(),
                delta_final_mean=sub["delta_final"].mean(),
            ))
        sub_all = variant_df[variant_df["variant"] == variant].dropna(subset=["raw_nfiq2", "final_nfiq2"])
        variant_summary_rows.append(dict(
            variant=variant, db="OVERALL", n=len(sub_all),
            final_mean=sub_all["final_nfiq2"].mean(),
            delta_final_mean=sub_all["delta_final"].mean(),
        ))
    # Fold the full hybrid's own final score in for direct side-by-side comparison.
    for db in DBS:
        sub = valid_final[valid_final["db"] == db]
        variant_summary_rows.append(dict(
            variant="full", db=db, n=len(sub),
            final_mean=sub["stage3_nfiq2"].mean(),
            delta_final_mean=sub["delta_final"].mean(),
        ))
    variant_summary_rows.append(dict(
        variant="full", db="OVERALL", n=len(valid_final),
        final_mean=valid_final["stage3_nfiq2"].mean(),
        delta_final_mean=valid_final["delta_final"].mean(),
    ))
    variant_summary_df = pd.DataFrame(variant_summary_rows)
    variant_summary_path = os.path.join(RESULTS_DIR, "hybrid_ablation_variants_summary.csv")
    variant_summary_df.to_csv(variant_summary_path, index=False)
    print(f"Wrote {variant_summary_path}")
    print("\n=== Full hybrid vs. each minus-one-component variant (final NFIQ2, per DB + overall) ===")
    pivot = variant_summary_df.pivot_table(index="db", columns="variant", values="final_mean")
    pivot = pivot[["full", "minus_p1", "minus_p2", "minus_p6"]]
    pivot = pivot.reindex(DBS + ["OVERALL"])
    print(pivot.round(2).to_string())


if __name__ == "__main__":
    main()
