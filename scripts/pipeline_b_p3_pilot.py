"""Run Pipeline B's fixed 16-image cumulative Stage-3 morphology pilot.

Shared Step 0 and locked P1/P2 defaults are always applied first. Only P6
parameters vary. Results are ignored tuning artefacts, not final 320-image
Pipeline B evidence.
"""

import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import (  # noqa: E402
    DEFAULT_NORMALIZE_TARGET_MEAN,
    DEFAULT_NORMALIZE_TARGET_STD,
    DEFAULT_NORMALIZE_TARGET_VAR,
    normalize_image,
    orientation_field,
    run_nfiq2_single,
    segment,
)
from config import NFIQ2_EXE, RAW_DIR  # noqa: E402
from pipeline_b import (  # noqa: E402
    _orientation_steered_morphology,
    _wavelet_contrast,
    _wavelet_shrinkage_denoise,
)
from pipeline_b_p2_pilot import (  # noqa: E402
    REFERENCE_SAMPLES,
    letterbox,
    load_samples,
    nfiq2_version,
    score_array,
)


OUTPUT_DIR = REPO_ROOT / "results" / "pipeline_b_p3_pilot"
SELECTED_DEFAULT = "L7_B12_S050_C16"

# Bounded first sweep around short ridge-aligned closing. IDENTITY validates
# the cumulative harness. Other candidates isolate kernel length, strength,
# angular resolution, and coherence gating without changing P1 or P2.
CANDIDATES = [
    dict(name="IDENTITY", length=1, bins=12, strength=0.0, floor=0.20, power=1.0, cap=16.0),
    dict(name="L3_B12_S025_C12", length=3, bins=12, strength=0.25, floor=0.20, power=1.0, cap=12.0),
    dict(name="L3_B12_S050_C12", length=3, bins=12, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L3_B12_S075_C12", length=3, bins=12, strength=0.75, floor=0.20, power=1.0, cap=12.0),
    dict(name="L5_B12_S025_C08", length=5, bins=12, strength=0.25, floor=0.20, power=1.0, cap=8.0),
    dict(name="L5_B12_S025_C12", length=5, bins=12, strength=0.25, floor=0.20, power=1.0, cap=12.0),
    dict(name="L5_B12_S025_C16", length=5, bins=12, strength=0.25, floor=0.20, power=1.0, cap=16.0),
    dict(name="L5_B12_S050_C08", length=5, bins=12, strength=0.50, floor=0.20, power=1.0, cap=8.0),
    dict(name="L5_B12_S050_C12", length=5, bins=12, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L5_B12_S050_C16", length=5, bins=12, strength=0.50, floor=0.20, power=1.0, cap=16.0),
    dict(name="L5_B08_S050_C12", length=5, bins=8, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L5_B16_S050_C12", length=5, bins=16, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S025_C12", length=7, bins=12, strength=0.25, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S050_C12", length=7, bins=12, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L5_F010_P1_S050", length=5, bins=12, strength=0.50, floor=0.10, power=1.0, cap=12.0),
    dict(name="L5_F030_P1_S050", length=5, bins=12, strength=0.50, floor=0.30, power=1.0, cap=12.0),
    dict(name="L5_F020_P05_S050", length=5, bins=12, strength=0.50, floor=0.20, power=0.5, cap=12.0),
    dict(name="L5_F020_P2_S050", length=5, bins=12, strength=0.50, floor=0.20, power=2.0, cap=12.0),
    # Focused refinement around the first sweep's L7/B12/S0.50 optimum.
    dict(name="L7_B12_S035_C12", length=7, bins=12, strength=0.35, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S040_C12", length=7, bins=12, strength=0.40, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S045_C12", length=7, bins=12, strength=0.45, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S055_C12", length=7, bins=12, strength=0.55, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S060_C12", length=7, bins=12, strength=0.60, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B12_S050_C08", length=7, bins=12, strength=0.50, floor=0.20, power=1.0, cap=8.0),
    dict(name="L7_B12_S050_C10", length=7, bins=12, strength=0.50, floor=0.20, power=1.0, cap=10.0),
    dict(name="L7_B12_S050_C16", length=7, bins=12, strength=0.50, floor=0.20, power=1.0, cap=16.0),
    dict(name="L7_B08_S050_C12", length=7, bins=8, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_B16_S050_C12", length=7, bins=16, strength=0.50, floor=0.20, power=1.0, cap=12.0),
    dict(name="L7_F030_P1_S050", length=7, bins=12, strength=0.50, floor=0.30, power=1.0, cap=12.0),
    dict(name="L7_F020_P2_S050", length=7, bins=12, strength=0.50, floor=0.20, power=2.0, cap=12.0),
]


def apply_candidate(record, candidate):
    return _orientation_steered_morphology(
        record["stage2"],
        record["theta"],
        record["coherence"],
        record["mask"],
        kernel_length=candidate["length"],
        orientation_bins=candidate["bins"],
        strength=candidate["strength"],
        coherence_floor=candidate["floor"],
        coherence_power=candidate["power"],
        max_darkening=candidate["cap"],
    )


def write_visual_panels(records, candidate_names):
    candidates = [next(c for c in CANDIDATES if c["name"] == name) for name in candidate_names]
    for db in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        rows = []
        for record in (item for item in records if item["database"] == db):
            images = [("RAW", record["raw"]), ("STAGE 2", record["stage2"])]
            images += [(candidate["name"], apply_candidate(record, candidate)) for candidate in candidates]
            tiles = []
            for index, (label, image) in enumerate(images):
                tile = cv2.cvtColor(letterbox(image), cv2.COLOR_GRAY2BGR)
                header = f"{record['filename']} {label}" if index == 0 else label
                cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (255, 255, 255), -1)
                cv2.putText(tile, header, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1, cv2.LINE_AA)
                tiles.append(tile)
            rows.append(cv2.hconcat(tiles))
        cv2.imwrite(str(OUTPUT_DIR / f"comparison_{db}.png"), cv2.vconcat(rows))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_samples()
    samples.to_csv(OUTPUT_DIR / "samples.csv", index=False)
    scores_path = OUTPUT_DIR / "per_image_configuration_scores.csv"
    existing = pd.read_csv(scores_path) if scores_path.is_file() else pd.DataFrame()

    records = []
    for row in samples.itertuples(index=False):
        path = Path(RAW_DIR) / row.database / row.filename
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        normalized = normalize_image(
            raw,
            target_mean=DEFAULT_NORMALIZE_TARGET_MEAN,
            target_var=DEFAULT_NORMALIZE_TARGET_VAR,
        )
        mask, _ = segment(normalized)
        stage1 = _wavelet_contrast(normalized, mask)
        stage2 = _wavelet_shrinkage_denoise(stage1, mask)
        theta, coherence = orientation_field(stage2)
        raw_score, raw_error = run_nfiq2_single(str(path))
        stage1_score, stage1_error = score_array(stage1, f"{row.database}_{row.filename}_stage1")
        stage2_score, stage2_error = score_array(stage2, f"{row.database}_{row.filename}_stage2")
        records.append(
            dict(
                database=row.database,
                filename=row.filename,
                raw=raw,
                stage1=stage1,
                stage2=stage2,
                mask=mask,
                theta=theta,
                coherence=coherence,
                raw_nfiq2=raw_score,
                raw_error=raw_error,
                stage1_nfiq2=stage1_score,
                stage1_error=stage1_error,
                stage2_nfiq2=stage2_score,
                stage2_error=stage2_error,
            )
        )
        print(
            f"Prepared {row.database}/{row.filename}: raw={raw_score}, "
            f"stage1={stage1_score}, stage2={stage2_score}",
            flush=True,
        )

    rows = existing.to_dict("records")
    completed = set(existing["configuration"]) if not existing.empty else set()
    if completed:
        print(f"Reusing {len(completed)} completed configurations.", flush=True)

    for candidate in CANDIDATES:
        if candidate["name"] in completed:
            continue
        print(f"\n--- {candidate['name']} ---", flush=True)
        for record in records:
            stage3 = apply_candidate(record, candidate)
            score, error = score_array(
                stage3,
                f"{candidate['name']}_{record['database']}_{record['filename']}",
            )
            rows.append(
                dict(
                    database=record["database"],
                    filename=record["filename"],
                    configuration=candidate["name"],
                    kernel_length=candidate["length"],
                    orientation_bins=candidate["bins"],
                    strength=candidate["strength"],
                    coherence_floor=candidate["floor"],
                    coherence_power=candidate["power"],
                    max_darkening=candidate["cap"],
                    raw_nfiq2=record["raw_nfiq2"],
                    stage1_nfiq2=record["stage1_nfiq2"],
                    stage2_nfiq2=record["stage2_nfiq2"],
                    stage3_nfiq2=score,
                    stage3_error=error,
                    stage3_minus_stage2=(
                        None if score is None or record["stage2_nfiq2"] is None
                        else score - record["stage2_nfiq2"]
                    ),
                    stage3_minus_raw=(
                        None if score is None or record["raw_nfiq2"] is None
                        else score - record["raw_nfiq2"]
                    ),
                    changed_pixels=int(np.count_nonzero(stage3 != record["stage2"])),
                    mean_absolute_change=float(
                        np.mean(np.abs(stage3.astype(np.float32) - record["stage2"].astype(np.float32)))
                    ),
                )
            )
            print(f"  {record['database']}/{record['filename']}: {score}", flush=True)
        pd.DataFrame(rows).to_csv(scores_path, index=False)

    scores = pd.DataFrame(rows)
    valid = scores.dropna(subset=["raw_nfiq2", "stage2_nfiq2", "stage3_nfiq2"]).copy()
    summaries = []
    grouped = [("ALL", name, group) for name, group in valid.groupby("configuration")]
    grouped += [(db, name, group) for (name, db), group in valid.groupby(["configuration", "database"])]
    for scope, name, group in grouped:
        delta = group["stage3_minus_stage2"]
        summaries.append(
            dict(
                scope=scope,
                configuration=name,
                samples=len(group),
                mean_stage3_minus_stage2=float(delta.mean()),
                median_stage3_minus_stage2=float(delta.median()),
                mean_stage3_minus_raw=float(group["stage3_minus_raw"].mean()),
                improved=int((delta > 0).sum()),
                regressed=int((delta < 0).sum()),
                unchanged=int((delta == 0).sum()),
                worst_delta=float(delta.min()),
                mean_absolute_change=float(group["mean_absolute_change"].mean()),
            )
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["scope", "mean_stage3_minus_stage2"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT_DIR / "configuration_summary.csv", index=False)
    all_scope = summary[(summary["scope"] == "ALL") & (summary["configuration"] != "IDENTITY")]
    automatic_best = all_scope.iloc[0]["configuration"]
    selected_name = SELECTED_DEFAULT
    write_visual_panels(
        records,
        ["L7_B12_S050_C12", automatic_best, selected_name],
    )

    stage_rows = []
    for record in records:
        for stage, score in (
            ("raw", record["raw_nfiq2"]),
            ("stage1", record["stage1_nfiq2"]),
            ("stage2", record["stage2_nfiq2"]),
        ):
            stage_rows.append(dict(database=record["database"], filename=record["filename"], configuration="SHARED", stage=stage, nfiq2=score))
    for row in valid.itertuples(index=False):
        stage_rows.append(dict(database=row.database, filename=row.filename, configuration=row.configuration, stage="stage3", nfiq2=row.stage3_nfiq2))
    pd.DataFrame(stage_rows).to_csv(OUTPUT_DIR / "stage_scores_long.csv", index=False)

    metadata = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        pywavelets=pywt.__version__,
        nfiq2_executable=NFIQ2_EXE,
        nfiq2_version=nfiq2_version(),
        shared_step0=dict(target_mean=DEFAULT_NORMALIZE_TARGET_MEAN, target_std=DEFAULT_NORMALIZE_TARGET_STD, target_var=DEFAULT_NORMALIZE_TARGET_VAR),
        locked_stage1=dict(wavelet="db4", level=3, coarse_gain=1.60, fine_gain=1.00, coefficient_floor_percentile=25.0, blend=1.0),
        locked_stage2=dict(wavelet="db4", level=3, threshold_scale=1.00, finest_levels=1, noise_adaptive=True, noise_reference_sigma=5.0, noise_adaptive_power=4.0, minimum_scale_factor=0.10),
        source_samples=str(REFERENCE_SAMPLES.relative_to(REPO_ROOT)),
        candidates=CANDIDATES,
        automatic_best_before_visual_review=automatic_best,
        selected_default_after_quantitative_and_visual_review=selected_name,
        selection_rationale=(
            "Balanced candidate: positive mean contribution in all four "
            "databases, 12/16 images improved, and worst delta -3; the "
            "highest-mean candidate had fewer improvements and worst delta -5. "
            "Visual review found no obvious cross-ridge merging, wrong-direction "
            "strokes, block seams, background spill, or grayscale clipping."
        ),
    )
    (OUTPUT_DIR / "environment_and_parameters.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    selected_rows = valid[valid["configuration"] == selected_name]
    selected_by_db = selected_rows.groupby("database").agg(
        mean_stage3_minus_stage2=("stage3_minus_stage2", "mean"),
        median_stage3_minus_stage2=("stage3_minus_stage2", "median"),
        mean_stage3_minus_raw=("stage3_minus_raw", "mean"),
    )
    report = [
        "# Pipeline B P6 orientation-steered morphology pilot",
        "",
        f"Selected balanced default: `{selected_name}`.",
        f"Highest-mean candidate retained for comparison: `{automatic_best}`.",
        "",
        "This is a fixed 16-image tuning pilot. P1/P2 are locked.",
        "",
        "## Selected default by database",
        "",
        "```",
        selected_by_db.round(3).to_string(),
        "```",
        "",
        "## Overall candidate ranking",
        "",
        "```",
        all_scope[["configuration", "mean_stage3_minus_stage2", "median_stage3_minus_stage2", "mean_stage3_minus_raw", "improved", "regressed", "worst_delta", "mean_absolute_change"]].round(3).to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "results_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote pilot outputs to {OUTPUT_DIR}")
    print(f"Automatic highest-mean candidate: {automatic_best}")
    print(f"Selected balanced default: {selected_name}")


if __name__ == "__main__":
    main()
