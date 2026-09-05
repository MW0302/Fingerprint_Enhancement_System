"""
Full-320-image validation for the two alternative Hybrid combinations
(docs/hybrid_alternative_combinations.md): variant 2a (src/hybrid/
hybrid_v2a.py -- Pipeline C's own homomorphic+feathered P1, paired back
with Pipeline C's diffusion) and variant 2b (src/hybrid/hybrid_v2b.py --
Pipeline B's wavelet contrast P1, paired with Pipeline B's OWN wavelet
shrinkage denoise instead of Pipeline C's diffusion). Both variants' full
cumulative ablation (raw -> +P1 -> +P1+P2 -> +P1+P2+P6) AND their own
minus-one-component variants, all scored with real NFIQ2 runs -- same
method as scripts/hybrid_ablation.py (which validated the current, already-
assembled Hybrid in src/hybrid/hybrid.py).

Does NOT modify src/hybrid/hybrid.py, hybrid_v2a.py, hybrid_v2b.py, or any
pipeline_*.py -- only calls their own separately-callable stage functions,
composed differently per variant, exactly like hybrid_ablation.py does for
the original Hybrid:
    v2a          : stage1_contrast_c(normalized) -> stage2_noise(stage1) -> stage3_orientation(stage2)
    v2a_minus_p1 : stage2_noise(normalized, normalized) -> stage3_orientation(that)  [P1 skipped]
    v2a_minus_p2 : stage3_orientation(stage1)                                        [P2 skipped]
    v2a_minus_p6 : stage2_noise(normalized, stage1) is itself the final output        [P6 skipped]
    v2b          : stage1_contrast(normalized) -> stage2_noise_b(stage1) -> stage3_orientation(stage2)
    v2b_minus_p1 : stage2_noise_b(normalized) -> stage3_orientation(that)             [P1 skipped]
    v2b_minus_p2 : stage3_orientation(stage1)                                         [P2 skipped]
    v2b_minus_p6 : stage2_noise_b(stage1) is itself the final output                  [P6 skipped]

Both variants share Step 0 (normalize+segment, hybrid.stage0_preprocess)
and the same raw NFIQ2 score, computed once per image and reused across
both. Everything else is scored fresh within this one script run for
self-consistency (not read back from results/hybrid_ablation.csv), since
NFIQ2 is invoked via subprocess per image and mixing scores across two
separate script runs is an unnecessary risk for no real time saving.

Missing-value handling: DB3_B/110_5.tif (the one image NFIQ2 can't score
raw) gets a blank raw_nfiq2 and the real NFIQ2 error text in raw_error,
matching every other pipeline's own ablation file -- never zero-filled,
never dropped from stage scores that don't depend on raw.

Outputs:
    results/hybrid_v2_ablation.csv          v2a+v2b, per-image, all 4 stages, one `variant` column ("v2a"/"v2b")
    results/hybrid_v2_ablation_summary.csv   same, per-DB + overall, per variant
    results/hybrid_v2_ablation_variants.csv  the 6 minus-one-component rows (3 per variant), per-image
    results/hybrid_v2_ablation_variants_summary.csv  same, per-DB + overall, plus each
                                              variant's own full final score for direct comparison

Run with:
    python scripts/hybrid_v2_ablation.py
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
import hybrid_v2a  # noqa: E402
import hybrid_v2b  # noqa: E402


def score(img):
    tmp_path = os.path.join(tempfile.gettempdir(), f"hybrid_v2_ablation_tmp_{os.getpid()}.tif")
    cv2.imwrite(tmp_path, img)
    s, err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return s, err


def process_image(path, db, fname):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    raw_score, raw_err = score(image)

    normalized, fg_mask_blocks = hybrid.stage0_preprocess(image)

    # ================= Variant 2a =================
    stage1_a = hybrid_v2a.stage1_contrast_c(normalized, fg_mask_blocks)
    stage1_a_score, stage1_a_err = score(stage1_a)

    stage2_a, _alpha_a = hybrid_v2a.stage2_noise(normalized, stage1_a, fg_mask_blocks)
    stage2_a_score, stage2_a_err = score(stage2_a)

    stage3_a = hybrid_v2a.stage3_orientation(stage2_a, fg_mask_blocks)
    stage3_a_score, stage3_a_err = score(stage3_a)

    full_a_row = dict(
        variant="v2a", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        stage1_nfiq2=stage1_a_score, stage1_error=stage1_a_err,
        stage2_nfiq2=stage2_a_score, stage2_error=stage2_a_err,
        stage3_nfiq2=stage3_a_score, stage3_error=stage3_a_err,
    )

    # v2a_minus_p6: final IS stage2_a, already computed above
    v2a_minus_p6_row = dict(
        variant="v2a_minus_p6", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage2_a_score, final_error=stage2_a_err,
    )

    # v2a_minus_p2: stage3 applied directly to stage1_a (P2 skipped)
    stage3_a_minus_p2 = hybrid_v2a.stage3_orientation(stage1_a, fg_mask_blocks)
    stage3_a_minus_p2_score, stage3_a_minus_p2_err = score(stage3_a_minus_p2)
    v2a_minus_p2_row = dict(
        variant="v2a_minus_p2", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_a_minus_p2_score, final_error=stage3_a_minus_p2_err,
    )

    # v2a_minus_p1: stage2 applied directly to normalized (P1 skipped), then stage3
    stage2_a_minus_p1, _alpha_a_minus_p1 = hybrid_v2a.stage2_noise(normalized, normalized, fg_mask_blocks)
    stage3_a_minus_p1 = hybrid_v2a.stage3_orientation(stage2_a_minus_p1, fg_mask_blocks)
    stage3_a_minus_p1_score, stage3_a_minus_p1_err = score(stage3_a_minus_p1)
    v2a_minus_p1_row = dict(
        variant="v2a_minus_p1", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_a_minus_p1_score, final_error=stage3_a_minus_p1_err,
    )

    # ================= Variant 2b =================
    stage1_b = hybrid_v2b.stage1_contrast(normalized, fg_mask_blocks)
    stage1_b_score, stage1_b_err = score(stage1_b)

    stage2_b = hybrid_v2b.stage2_noise_b(stage1_b, fg_mask_blocks)
    stage2_b_score, stage2_b_err = score(stage2_b)

    stage3_b = hybrid_v2b.stage3_orientation(stage2_b, fg_mask_blocks)
    stage3_b_score, stage3_b_err = score(stage3_b)

    full_b_row = dict(
        variant="v2b", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        stage1_nfiq2=stage1_b_score, stage1_error=stage1_b_err,
        stage2_nfiq2=stage2_b_score, stage2_error=stage2_b_err,
        stage3_nfiq2=stage3_b_score, stage3_error=stage3_b_err,
    )

    # v2b_minus_p6: final IS stage2_b, already computed above
    v2b_minus_p6_row = dict(
        variant="v2b_minus_p6", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage2_b_score, final_error=stage2_b_err,
    )

    # v2b_minus_p2: stage3 applied directly to stage1_b (P2 skipped)
    stage3_b_minus_p2 = hybrid_v2b.stage3_orientation(stage1_b, fg_mask_blocks)
    stage3_b_minus_p2_score, stage3_b_minus_p2_err = score(stage3_b_minus_p2)
    v2b_minus_p2_row = dict(
        variant="v2b_minus_p2", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_b_minus_p2_score, final_error=stage3_b_minus_p2_err,
    )

    # v2b_minus_p1: stage2_noise_b applied directly to normalized (P1 skipped), then stage3
    stage2_b_minus_p1 = hybrid_v2b.stage2_noise_b(normalized, fg_mask_blocks)
    stage3_b_minus_p1 = hybrid_v2b.stage3_orientation(stage2_b_minus_p1, fg_mask_blocks)
    stage3_b_minus_p1_score, stage3_b_minus_p1_err = score(stage3_b_minus_p1)
    v2b_minus_p1_row = dict(
        variant="v2b_minus_p1", db=db, file=fname,
        raw_nfiq2=raw_score, raw_error=raw_err,
        final_nfiq2=stage3_b_minus_p1_score, final_error=stage3_b_minus_p1_err,
    )

    full_rows = [full_a_row, full_b_row]
    variant_rows = [
        v2a_minus_p1_row, v2a_minus_p2_row, v2a_minus_p6_row,
        v2b_minus_p1_row, v2b_minus_p2_row, v2b_minus_p6_row,
    ]
    return full_rows, variant_rows


def main():
    full_rows = []
    variant_rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            result = process_image(path, db, fname)
            if result is None:
                print(f"WARNING: could not read {path}, skipping")
                continue
            db_full_rows, db_variant_rows = result
            full_rows.extend(db_full_rows)
            variant_rows.extend(db_variant_rows)
        print(f"{db}: {len(paths)} images done", flush=True)

    full_df = pd.DataFrame(full_rows)
    full_df["delta_p1"] = full_df["stage1_nfiq2"] - full_df["raw_nfiq2"]
    full_df["delta_p2"] = full_df["stage2_nfiq2"] - full_df["stage1_nfiq2"]
    full_df["delta_p6"] = full_df["stage3_nfiq2"] - full_df["stage2_nfiq2"]
    full_df["delta_final"] = full_df["stage3_nfiq2"] - full_df["raw_nfiq2"]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    full_path = os.path.join(RESULTS_DIR, "hybrid_v2_ablation.csv")
    full_df.to_csv(full_path, index=False)
    print(f"\nWrote {len(full_df)} rows to {full_path}")

    summary_rows = []
    for variant in ("v2a", "v2b"):
        variant_df = full_df[full_df["variant"] == variant]
        valid_final = variant_df.dropna(subset=["raw_nfiq2", "stage3_nfiq2"])
        for db in DBS:
            sub = valid_final[valid_final["db"] == db]
            summary_rows.append(dict(
                variant=variant, db=db, n=len(sub),
                raw_mean=sub["raw_nfiq2"].mean(),
                final_mean=sub["stage3_nfiq2"].mean(),
                delta_final_mean=sub["delta_final"].mean(),
                improved=(sub["delta_final"] > 0).sum(),
                regressed=(sub["delta_final"] < 0).sum(),
                unchanged=(sub["delta_final"] == 0).sum(),
            ))
        summary_rows.append(dict(
            variant=variant, db="OVERALL", n=len(valid_final),
            raw_mean=valid_final["raw_nfiq2"].mean(),
            final_mean=valid_final["stage3_nfiq2"].mean(),
            delta_final_mean=valid_final["delta_final"].mean(),
            improved=(valid_final["delta_final"] > 0).sum(),
            regressed=(valid_final["delta_final"] < 0).sum(),
            unchanged=(valid_final["delta_final"] == 0).sum(),
        ))
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "hybrid_v2_ablation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\n=== v2a / v2b cumulative summary ===")
    print(summary_df.round(2).to_string(index=False))

    variant_df_all = pd.DataFrame(variant_rows)
    variant_df_all["delta_final"] = variant_df_all["final_nfiq2"] - variant_df_all["raw_nfiq2"]
    variant_path = os.path.join(RESULTS_DIR, "hybrid_v2_ablation_variants.csv")
    variant_df_all.to_csv(variant_path, index=False)
    print(f"\nWrote {len(variant_df_all)} rows to {variant_path}")

    variant_summary_rows = []
    for variant in ("v2a_minus_p1", "v2a_minus_p2", "v2a_minus_p6",
                    "v2b_minus_p1", "v2b_minus_p2", "v2b_minus_p6"):
        for db in DBS:
            sub = variant_df_all[(variant_df_all["variant"] == variant) & (variant_df_all["db"] == db)]
            sub = sub.dropna(subset=["raw_nfiq2", "final_nfiq2"])
            variant_summary_rows.append(dict(
                variant=variant, db=db, n=len(sub),
                final_mean=sub["final_nfiq2"].mean(),
                delta_final_mean=sub["delta_final"].mean(),
            ))
        sub_all = variant_df_all[variant_df_all["variant"] == variant].dropna(subset=["raw_nfiq2", "final_nfiq2"])
        variant_summary_rows.append(dict(
            variant=variant, db="OVERALL", n=len(sub_all),
            final_mean=sub_all["final_nfiq2"].mean(),
            delta_final_mean=sub_all["delta_final"].mean(),
        ))
    # Fold each variant's own full final score in for direct side-by-side comparison.
    for variant in ("v2a", "v2b"):
        variant_full_df = full_df[full_df["variant"] == variant]
        valid_final = variant_full_df.dropna(subset=["raw_nfiq2", "stage3_nfiq2"])
        for db in DBS:
            sub = valid_final[valid_final["db"] == db]
            variant_summary_rows.append(dict(
                variant=variant, db=db, n=len(sub),
                final_mean=sub["stage3_nfiq2"].mean(),
                delta_final_mean=sub["delta_final"].mean(),
            ))
        variant_summary_rows.append(dict(
            variant=variant, db="OVERALL", n=len(valid_final),
            final_mean=valid_final["stage3_nfiq2"].mean(),
            delta_final_mean=valid_final["delta_final"].mean(),
        ))
    variant_summary_df = pd.DataFrame(variant_summary_rows)
    variant_summary_path = os.path.join(RESULTS_DIR, "hybrid_v2_ablation_variants_summary.csv")
    variant_summary_df.to_csv(variant_summary_path, index=False)
    print(f"Wrote {variant_summary_path}")

    print("\n=== v2a: full vs. each minus-one-component variant (final NFIQ2, per DB + overall) ===")
    pivot_a = variant_summary_df.pivot_table(index="db", columns="variant", values="final_mean")
    pivot_a = pivot_a[["v2a", "v2a_minus_p1", "v2a_minus_p2", "v2a_minus_p6"]]
    pivot_a = pivot_a.reindex(DBS + ["OVERALL"])
    print(pivot_a.round(2).to_string())

    print("\n=== v2b: full vs. each minus-one-component variant (final NFIQ2, per DB + overall) ===")
    pivot_b = variant_summary_df.pivot_table(index="db", columns="variant", values="final_mean")
    pivot_b = pivot_b[["v2b", "v2b_minus_p1", "v2b_minus_p2", "v2b_minus_p6"]]
    pivot_b = pivot_b.reindex(DBS + ["OVERALL"])
    print(pivot_b.round(2).to_string())


if __name__ == "__main__":
    main()
