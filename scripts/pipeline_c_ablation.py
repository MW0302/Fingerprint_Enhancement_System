"""
Cumulative (step-wise) ablation wrapper around Pipeline C's enhance(), for
per-technique NFIQ2 contribution analysis. Does NOT modify pipeline_c.py —
imports and calls its actual internal functions directly, in the exact same
order and with the exact same alpha-threading enhance() itself uses, so
Stage 3's output is expected to reproduce enhance()'s own output (same
inputs -> same deterministic pipeline).

Chain reproduced from src/pipeline_c/pipeline_c.py's enhance() (confirmed by
reading that file directly, not assumed):
    0a. normalize_image()              (common.py)
    0b. segment()                      (common.py)                -> fg_mask_blocks
    0c. orientation_field(normalized)  (common.py, PROBE ONLY, on
        the normalised-but-unfiltered image)                      -> coherence_probe
        alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)
        (alpha is computed ONCE here and reused for gamma_high,
        diffusion iterations+kappa, AND add_gain below -- enhance()
        does not recompute it per step, so neither does this script)
    1.  _homomorphic_filter(normalized, gamma_high=lerp(alpha))    -> contrast_enhanced_raw
    1b. fg_alpha-weighted blend of contrast_enhanced_raw back
        toward `normalized` over background regions (NOT a separate
        function in pipeline_c.py -- it's inline in enhance() itself,
        reproduced here verbatim)                                 -> contrast_enhanced = STAGE 1
    2.  orientation_field(contrast_enhanced) -- a SEPARATE call from
        the Step 0c probe above, run on Stage 1's output           -> theta_field, coherence_field
    3.  _coherence_diffusion(contrast_enhanced, theta_field,
        coherence_field, iterations=lerp(alpha), kappa=lerp(alpha))-> diffused = STAGE 2
    4a. _estimate_ridge_frequency(diffused, theta_field)            -> freq_field
    4b. _log_gabor_enhance(diffused, theta_field, freq_field,
        fg_mask_blocks, add_gain=lerp(alpha),
        coherence_field=coherence_field)                           -> STAGE 3 (== enhance()'s output)

Run with:
    python scripts/pipeline_c_ablation.py --pilot   16-image pilot (4 per DB),
                                                      validates Stage 3 against
                                                      data/processed/pipeline_c/<DB>/batch_results.csv
    python scripts/pipeline_c_ablation.py --full     all 320 images -> results/pipeline_c_ablation.csv
                                                      (only run this after --pilot passes)
"""

import os
import sys
import glob
import random
import argparse
import tempfile

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "pipeline_c"))

from common import normalize_image, segment, orientation_field, run_nfiq2_single  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, PROCESSED_DIR, DBS  # noqa: E402
from pipeline_c import (  # noqa: E402
    _aggressiveness_alpha,
    _lerp,
    _homomorphic_filter,
    _coherence_diffusion,
    _estimate_ridge_frequency,
    _log_gabor_enhance,
    _HOMOMORPHIC_GAMMA_HIGH_RANGE,
    _DIFFUSION_ITERATIONS_RANGE,
    _DIFFUSION_KAPPA_RANGE,
    _LOG_GABOR_ADD_GAIN_RANGE,
)


def score_nfiq2(img, ext=".tif"):
    """Writes img to a temp file and scores it with the same run_nfiq2_single
    helper the real pipeline/batch scripts use, then removes the temp file."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"pipeline_c_ablation_tmp{ext}")
    cv2.imwrite(tmp_path, img)
    score, err = run_nfiq2_single(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return score, err


def ablate(raw_img):
    """Reproduces pipeline_c.enhance()'s exact step chain, returning each
    stage's output image plus the alpha value used throughout."""
    # Step 0a/0b (shared preprocessing, same as enhance())
    normalized = normalize_image(raw_img)
    fg_mask_blocks, _block_var = segment(normalized)

    # Step 0c: alpha probe -- SEPARATE orientation_field() call on the
    # normalised-but-unfiltered image, used ONLY to set alpha (see
    # enhance()'s own comment: "This probe orientation_field() call is
    # separate from step 2's (below)").
    _theta_probe, coherence_probe = orientation_field(normalized)
    alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)

    # Step 1: homomorphic filtering (P1), gamma_high alpha-scaled.
    contrast_enhanced_raw = _homomorphic_filter(
        normalized,
        gamma_high=_lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha),
    )

    # Step 1b: fg_alpha-weighted background feathering blend, reproduced
    # verbatim from enhance() (this logic lives inline in enhance() itself,
    # not in a separate _-prefixed function, so it's copied here rather than
    # imported).
    h, w = normalized.shape
    fg_alpha = cv2.resize(fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), 8.0)
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
    contrast_enhanced = (
        fg_alpha * contrast_enhanced_raw.astype(np.float64)
        + (1 - fg_alpha) * normalized.astype(np.float64)
    )
    stage1 = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)

    # Step 2: orientation field on STAGE 1's output (separate from the Step
    # 0c probe above) -- steers steps 3 and 4, exactly as in enhance().
    theta_field, coherence_field = orientation_field(stage1)

    # Step 3: coherence-enhancing anisotropic diffusion (P2), iterations/kappa
    # alpha-scaled.
    diffusion_iterations = int(round(_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha)))
    diffusion_kappa = _lerp(_DIFFUSION_KAPPA_RANGE, alpha)
    stage2 = _coherence_diffusion(
        stage1, theta_field, coherence_field,
        iterations=diffusion_iterations, dt=0.2, kappa=diffusion_kappa,
        confidence_ceiling=0.45,
    )

    # Step 4a/4b: ridge-frequency estimation + 2D Log-Gabor filtering (P6),
    # add_gain alpha-scaled.
    freq_field = _estimate_ridge_frequency(stage2, theta_field)
    stage3 = _log_gabor_enhance(
        stage2, theta_field, freq_field, fg_mask_blocks,
        add_gain=_lerp(_LOG_GABOR_ADD_GAIN_RANGE, alpha),
        coherence_field=coherence_field,
        confidence_ceiling=0.45,
    )

    return dict(alpha=alpha, stage1=stage1, stage2=stage2, stage3=stage3)


def run_one(db, fname):
    path = os.path.join(RAW_DIR, db, fname)
    raw_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if raw_img is None:
        return None

    raw_score, _ = run_nfiq2_single(path)
    result = ablate(raw_img)
    stage1_score, _ = score_nfiq2(result["stage1"])
    stage2_score, _ = score_nfiq2(result["stage2"])
    stage3_score, _ = score_nfiq2(result["stage3"])

    return dict(
        file=fname,
        db=db,
        alpha=result["alpha"],
        raw_nfiq2=raw_score,
        stage1_contrast_nfiq2=stage1_score,
        stage2_noise_nfiq2=stage2_score,
        stage3_orientation_nfiq2=stage3_score,
    )


def pilot():
    random.seed(42)
    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        fnames = [os.path.basename(p) for p in paths]
        sample = random.sample(fnames, min(4, len(fnames)))
        for fname in sorted(sample):
            print(f"Running ablation on {db}/{fname}...")
            row = run_one(db, fname)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)

    # Compare stage3 against the already-validated recorded enhanced_nfiq2
    # in data/processed/pipeline_c/<DB>/batch_results.csv for the same file.
    print("\n=== Pilot validation: Stage 3 vs. recorded enhanced_nfiq2 ===")
    all_ok = True
    for db in DBS:
        recorded_path = os.path.join(PROCESSED_DIR, "pipeline_c", db, "batch_results.csv")
        recorded = pd.read_csv(recorded_path).set_index("file")["enhanced_nfiq2"]
        sub = df[df["db"] == db]
        for _, row in sub.iterrows():
            recorded_score = recorded.get(row["file"])
            stage3_score = row["stage3_orientation_nfiq2"]
            if recorded_score is None or pd.isna(recorded_score):
                print(f"  {db}/{row['file']}: no recorded score to compare against -- skipping")
                continue
            diff = abs(stage3_score - recorded_score) if stage3_score is not None else float("inf")
            status = "OK" if diff <= 1.0 else "MISMATCH"
            if status == "MISMATCH":
                all_ok = False
            print(f"  {db}/{row['file']}: stage3={stage3_score}  recorded={recorded_score}  "
                  f"diff={diff:.3f}  alpha={row['alpha']:.3f}  [{status}]")

    print(f"\nPilot result: {'PASS' if all_ok else 'FAIL'}")
    if not all_ok:
        print("STOPPING -- Stage 3 does not reproduce the recorded pipeline output. "
              "Do not proceed to a full batch run. Check alpha-threading and the "
              "Step 1b blend first.")
    return all_ok, df


def full_run():
    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            fname = os.path.basename(path)
            print(f"Running ablation on {db}/{fname}...")
            row = run_one(db, fname)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "pipeline_c_ablation.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    print("\n=== Per-DB marginal contribution summary ===")
    summary_rows = []
    for db in DBS:
        sub = df[df["db"] == db]
        valid = sub.dropna(subset=["raw_nfiq2", "stage1_contrast_nfiq2",
                                    "stage2_noise_nfiq2", "stage3_orientation_nfiq2"])
        delta_p1 = (valid["stage1_contrast_nfiq2"] - valid["raw_nfiq2"]).mean()
        delta_p2 = (valid["stage2_noise_nfiq2"] - valid["stage1_contrast_nfiq2"]).mean()
        delta_p6 = (valid["stage3_orientation_nfiq2"] - valid["stage2_noise_nfiq2"]).mean()
        delta_total = (valid["stage3_orientation_nfiq2"] - valid["raw_nfiq2"]).mean()
        summary_rows.append(dict(
            db=db, n=len(valid),
            delta_p1_homomorphic=delta_p1,
            delta_p2_diffusion=delta_p2,
            delta_p6_log_gabor=delta_p6,
            delta_total=delta_total,
        ))
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_out = os.path.join(RESULTS_DIR, "pipeline_c_ablation_summary.csv")
    summary_df.to_csv(summary_out, index=False)
    print(f"\nWrote per-DB summary to {summary_out}")
    return df, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    if args.pilot:
        pilot()
    elif args.full:
        full_run()
    else:
        print("Pass --pilot (16-image validation) or --full (all 320 images).")
