"""Run Pipeline B's fixed 16-image cumulative Stage-2 denoising pilot.

The experiment always applies shared Step 0 and Pipeline B's locked P1
defaults before varying only P2. P6 is deliberately not called. Outputs live
under an ignored results directory because they are tuning evidence rather
than the final 320-image CSV.
"""

import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pywt


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))

from common import (  # noqa: E402
    DEFAULT_NORMALIZE_TARGET_MEAN,
    DEFAULT_NORMALIZE_TARGET_STD,
    DEFAULT_NORMALIZE_TARGET_VAR,
    normalize_image,
    run_nfiq2_single,
    segment,
)
from config import NFIQ2_EXE, RAW_DIR  # noqa: E402
from pipeline_b import _wavelet_contrast, _wavelet_shrinkage_denoise  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "results" / "pipeline_b_p2_pilot"
SELECTED_DEFAULT = "ADAPT_B100_P4_M010"
REFERENCE_SAMPLES = (
    REPO_ROOT
    / "data"
    / "pipeline_a_16_sample_single_parameter_candidates_nfiq2_20260904"
    / "samples.csv"
)

# A bounded comparison around BayesShrink soft thresholding. IDENTITY checks
# the scoring harness; finest_levels compares ridge-preserving selective
# shrinkage against processing two scales; other wavelets test robustness.
CANDIDATES = [
    dict(name="IDENTITY", wavelet="db4", level=3, scale=0.00, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S010_F1", wavelet="db4", level=3, scale=0.10, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S025_F1", wavelet="db4", level=3, scale=0.25, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S050_F1", wavelet="db4", level=3, scale=0.50, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S075_F1", wavelet="db4", level=3, scale=0.75, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S100_F1", wavelet="db4", level=3, scale=1.00, finest_levels=1, blend=1.0),
    dict(name="DB4_L3_S010_F2", wavelet="db4", level=3, scale=0.10, finest_levels=2, blend=1.0),
    dict(name="DB4_L3_S025_F2", wavelet="db4", level=3, scale=0.25, finest_levels=2, blend=1.0),
    dict(name="DB4_L3_S050_F2", wavelet="db4", level=3, scale=0.50, finest_levels=2, blend=1.0),
    dict(name="DB4_L3_S100_F2", wavelet="db4", level=3, scale=1.00, finest_levels=2, blend=1.0),
    dict(name="DB4_L2_S025_F1", wavelet="db4", level=2, scale=0.25, finest_levels=1, blend=1.0),
    dict(name="DB4_L2_S050_F1", wavelet="db4", level=2, scale=0.50, finest_levels=1, blend=1.0),
    dict(name="DB2_L3_S025_F1", wavelet="db2", level=3, scale=0.25, finest_levels=1, blend=1.0),
    dict(name="DB2_L3_S050_F1", wavelet="db2", level=3, scale=0.50, finest_levels=1, blend=1.0),
    dict(name="SYM4_L3_S025_F1", wavelet="sym4", level=3, scale=0.25, finest_levels=1, blend=1.0),
    dict(name="SYM4_L3_S050_F1", wavelet="sym4", level=3, scale=0.50, finest_levels=1, blend=1.0),
    dict(name="COIF1_L3_S025_F1", wavelet="coif1", level=3, scale=0.25, finest_levels=1, blend=1.0),
    dict(name="COIF1_L3_S050_F1", wavelet="coif1", level=3, scale=0.50, finest_levels=1, blend=1.0),
    # Refinement: continuous noise-adaptive strength avoids treating strong
    # clean ridge detail as if it were random noise. All images retain a
    # non-zero minimum factor, so P2 is never conditionally bypassed.
    dict(name="ADAPT_B050_P1_M010", wavelet="db4", level=3, scale=0.50, finest_levels=1, blend=1.0, adaptive=True, power=1.0, minimum=0.10),
    dict(name="ADAPT_B050_P2_M010", wavelet="db4", level=3, scale=0.50, finest_levels=1, blend=1.0, adaptive=True, power=2.0, minimum=0.10),
    dict(name="ADAPT_B075_P1_M010", wavelet="db4", level=3, scale=0.75, finest_levels=1, blend=1.0, adaptive=True, power=1.0, minimum=0.10),
    dict(name="ADAPT_B075_P2_M010", wavelet="db4", level=3, scale=0.75, finest_levels=1, blend=1.0, adaptive=True, power=2.0, minimum=0.10),
    dict(name="ADAPT_B100_P2_M010", wavelet="db4", level=3, scale=1.00, finest_levels=1, blend=1.0, adaptive=True, power=2.0, minimum=0.10),
    dict(name="ADAPT_B050_P2_M020", wavelet="db4", level=3, scale=0.50, finest_levels=1, blend=1.0, adaptive=True, power=2.0, minimum=0.20),
    dict(name="ADAPT_B075_P3_M010", wavelet="db4", level=3, scale=0.75, finest_levels=1, blend=1.0, adaptive=True, power=3.0, minimum=0.10),
    dict(name="ADAPT_B075_P4_M010", wavelet="db4", level=3, scale=0.75, finest_levels=1, blend=1.0, adaptive=True, power=4.0, minimum=0.10),
    dict(name="ADAPT_B100_P3_M010", wavelet="db4", level=3, scale=1.00, finest_levels=1, blend=1.0, adaptive=True, power=3.0, minimum=0.10),
    dict(name="ADAPT_B100_P4_M010", wavelet="db4", level=3, scale=1.00, finest_levels=1, blend=1.0, adaptive=True, power=4.0, minimum=0.10),
]


def score_array(image, label):
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
    fd, path = tempfile.mkstemp(prefix=f"pipeline_b_p2_{safe_label}_", suffix=".tif")
    os.close(fd)
    try:
        if not cv2.imwrite(path, np.clip(image, 0, 255).astype(np.uint8)):
            return None, "OpenCV could not write temporary TIFF"
        return run_nfiq2_single(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def load_samples():
    samples = pd.read_csv(REFERENCE_SAMPLES)
    if set(samples.columns) != {"database", "filename"}:
        raise ValueError(f"Unexpected sample columns: {list(samples.columns)}")
    if len(samples) != 16 or not (samples.groupby("database").size() == 4).all():
        raise ValueError("Pilot manifest must contain four samples from each DB")
    for row in samples.itertuples(index=False):
        path = Path(RAW_DIR) / row.database / row.filename
        if not path.is_file():
            raise FileNotFoundError(path)
    return samples


def nfiq2_version():
    result = subprocess.run([NFIQ2_EXE], capture_output=True, text=True, check=False)
    for line in (result.stdout + "\n" + result.stderr).splitlines():
        if line.strip().startswith("NFIQ 2:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def letterbox(image, size=300):
    scale = min(size / image.shape[1], size / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size), 127, dtype=np.uint8)
    x0, y0 = (size - width) // 2, (size - height) // 2
    canvas[y0:y0 + height, x0:x0 + width] = resized
    return canvas


def write_visual_panels(records, candidate_names):
    candidates = [next(c for c in CANDIDATES if c["name"] == name) for name in candidate_names]
    for db in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        rows = []
        for record in (item for item in records if item["database"] == db):
            images = [("RAW", record["raw"]), ("STAGE 1", record["stage1"])]
            for candidate in candidates:
                stage2 = _wavelet_shrinkage_denoise(
                    record["stage1"],
                    record["mask"],
                    wavelet=candidate["wavelet"],
                    level=candidate["level"],
                    threshold_scale=candidate["scale"],
                    denoise_finest_levels=candidate["finest_levels"],
                    blend=candidate["blend"],
                    noise_adaptive=candidate.get("adaptive", False),
                    noise_adaptive_power=candidate.get("power", 2.0),
                    minimum_scale_factor=candidate.get("minimum", 0.10),
                )
                images.append((candidate["name"], stage2))
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
        raw_score, raw_error = run_nfiq2_single(str(path))
        stage1_score, stage1_error = score_array(stage1, f"{row.database}_{row.filename}_stage1")
        records.append(
            dict(
                database=row.database,
                filename=row.filename,
                raw=raw,
                stage1=stage1,
                mask=mask,
                raw_nfiq2=raw_score,
                raw_error=raw_error,
                stage1_nfiq2=stage1_score,
                stage1_error=stage1_error,
            )
        )
        print(f"Prepared {row.database}/{row.filename}: raw={raw_score}, stage1={stage1_score}", flush=True)

    rows = existing.to_dict("records")
    completed = set(existing["configuration"]) if not existing.empty else set()
    if completed:
        print(f"Reusing {len(completed)} completed configurations.", flush=True)

    for candidate in CANDIDATES:
        if candidate["name"] in completed:
            continue
        print(f"\n--- {candidate['name']} ---", flush=True)
        for record in records:
            stage2 = _wavelet_shrinkage_denoise(
                record["stage1"],
                record["mask"],
                wavelet=candidate["wavelet"],
                level=candidate["level"],
                threshold_scale=candidate["scale"],
                denoise_finest_levels=candidate["finest_levels"],
                blend=candidate["blend"],
                noise_adaptive=candidate.get("adaptive", False),
                noise_adaptive_power=candidate.get("power", 2.0),
                minimum_scale_factor=candidate.get("minimum", 0.10),
            )
            score, error = score_array(
                stage2,
                f"{candidate['name']}_{record['database']}_{record['filename']}",
            )
            rows.append(
                dict(
                    database=record["database"],
                    filename=record["filename"],
                    configuration=candidate["name"],
                    wavelet=candidate["wavelet"],
                    level=candidate["level"],
                    threshold_scale=candidate["scale"],
                    denoise_finest_levels=candidate["finest_levels"],
                    blend=candidate["blend"],
                    noise_adaptive=candidate.get("adaptive", False),
                    noise_adaptive_power=candidate.get("power", 2.0),
                    minimum_scale_factor=candidate.get("minimum", 0.10),
                    raw_nfiq2=record["raw_nfiq2"],
                    stage1_nfiq2=record["stage1_nfiq2"],
                    stage2_nfiq2=score,
                    stage2_error=error,
                    stage2_minus_stage1=(
                        None if score is None or record["stage1_nfiq2"] is None
                        else score - record["stage1_nfiq2"]
                    ),
                    stage2_minus_raw=(
                        None if score is None or record["raw_nfiq2"] is None
                        else score - record["raw_nfiq2"]
                    ),
                    changed_pixels=int(np.count_nonzero(stage2 != record["stage1"])),
                    mean_absolute_change=float(
                        np.mean(np.abs(stage2.astype(np.float32) - record["stage1"].astype(np.float32)))
                    ),
                )
            )
            print(f"  {record['database']}/{record['filename']}: {score}", flush=True)
        pd.DataFrame(rows).to_csv(scores_path, index=False)

    scores = pd.DataFrame(rows)
    valid = scores.dropna(subset=["raw_nfiq2", "stage1_nfiq2", "stage2_nfiq2"]).copy()
    summaries = []
    grouped = [("ALL", name, group) for name, group in valid.groupby("configuration")]
    grouped += [(db, name, group) for (name, db), group in valid.groupby(["configuration", "database"])]
    for scope, name, group in grouped:
        delta = group["stage2_minus_stage1"]
        summaries.append(
            dict(
                scope=scope,
                configuration=name,
                samples=len(group),
                mean_stage2_minus_stage1=float(delta.mean()),
                median_stage2_minus_stage1=float(delta.median()),
                mean_stage2_minus_raw=float(group["stage2_minus_raw"].mean()),
                improved=int((delta > 0).sum()),
                regressed=int((delta < 0).sum()),
                unchanged=int((delta == 0).sum()),
                worst_delta=float(delta.min()),
                mean_absolute_change=float(group["mean_absolute_change"].mean()),
            )
        )
    summary = pd.DataFrame(summaries).sort_values(
        ["scope", "mean_stage2_minus_stage1"], ascending=[True, False]
    )
    summary.to_csv(OUTPUT_DIR / "configuration_summary.csv", index=False)
    all_scope = summary[(summary["scope"] == "ALL") & (summary["configuration"] != "IDENTITY")]
    automatic_best = all_scope.iloc[0]["configuration"]
    selected_name = SELECTED_DEFAULT
    write_visual_panels(
        records,
        ["DB4_L3_S050_F1", automatic_best, selected_name],
    )

    stage_rows = []
    for record in records:
        stage_rows.extend(
            [
                dict(database=record["database"], filename=record["filename"], configuration="SHARED", stage="raw", nfiq2=record["raw_nfiq2"]),
                dict(database=record["database"], filename=record["filename"], configuration="SHARED", stage="stage1", nfiq2=record["stage1_nfiq2"]),
            ]
        )
    for row in valid.itertuples(index=False):
        stage_rows.append(dict(database=row.database, filename=row.filename, configuration=row.configuration, stage="stage2", nfiq2=row.stage2_nfiq2))
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
        source_samples=str(REFERENCE_SAMPLES.relative_to(REPO_ROOT)),
        candidates=CANDIDATES,
        automatic_best_before_visual_review=automatic_best,
        selected_default_after_quantitative_and_visual_review=selected_name,
        selection_rationale=(
            "All four database means are non-negative; compared with the "
            "highest-mean candidate this choice has fewer regressions and a "
            "better worst case while retaining a positive DB3 contribution. "
            "Visual review of all 16 images found no obvious blur, ringing, "
            "foreground-mask seams, or grayscale clipping."
        ),
    )
    (OUTPUT_DIR / "environment_and_parameters.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    selected_rows = valid[valid["configuration"] == selected_name]
    selected_by_db = selected_rows.groupby("database").agg(
        mean_stage2_minus_stage1=("stage2_minus_stage1", "mean"),
        median_stage2_minus_stage1=("stage2_minus_stage1", "median"),
        mean_stage2_minus_raw=("stage2_minus_raw", "mean"),
    )
    report = [
        "# Pipeline B P2 wavelet shrinkage pilot",
        "",
        f"Selected balanced default: `{selected_name}`.",
        f"Highest-mean candidate retained for comparison: `{automatic_best}`.",
        "",
        "This is a fixed 16-image tuning pilot. P1 is locked and P6 was not run.",
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
        all_scope[["configuration", "mean_stage2_minus_stage1", "median_stage2_minus_stage1", "mean_stage2_minus_raw", "improved", "regressed", "worst_delta", "mean_absolute_change"]].round(3).to_string(index=False),
        "```",
    ]
    (OUTPUT_DIR / "results_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote pilot outputs to {OUTPUT_DIR}")
    print(f"Automatic highest-mean candidate: {automatic_best}")
    print(f"Selected balanced default: {selected_name}")


if __name__ == "__main__":
    main()
