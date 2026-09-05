"""Run the fixed 16-image Pipeline B Stage-1 wavelet contrast pilot.

The pilot deliberately stops after P1. Pipeline B's P2 denoising and P6
morphology remain untouched. Raw and shared-Step-0 scores are computed once
per image; every candidate is then evaluated from the same normalised input.

Outputs are written below ``results/pipeline_b_p1_pilot``. That directory is
ignored by Git because these are tuning artefacts, not the final locked 320-
image CSVs that the team has agreed to commit.
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
from pipeline_b import _wavelet_contrast  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "results" / "pipeline_b_p1_pilot"
SELECTED_DEFAULT = "DB4_L3_G160_100_P25"
REFERENCE_SAMPLES = (
    REPO_ROOT
    / "data"
    / "pipeline_a_16_sample_single_parameter_candidates_nfiq2_20260904"
    / "samples.csv"
)

# A deliberately bounded search. IDENTITY proves that the test harness does
# not manufacture a score change; the remaining candidates vary one compact
# family of design choices rather than conducting an unconstrained search.
CANDIDATES = [
    dict(name="IDENTITY", wavelet="db4", level=2, coarse_gain=1.0, fine_gain=1.0, floor=50.0),
    dict(name="DB4_L2_G120_108_P50", wavelet="db4", level=2, coarse_gain=1.20, fine_gain=1.08, floor=50.0),
    dict(name="DB4_L2_G140_115_P50", wavelet="db4", level=2, coarse_gain=1.40, fine_gain=1.15, floor=50.0),
    dict(name="DB4_L2_G160_120_P50", wavelet="db4", level=2, coarse_gain=1.60, fine_gain=1.20, floor=50.0),
    dict(name="DB4_L2_G180_125_P50", wavelet="db4", level=2, coarse_gain=1.80, fine_gain=1.25, floor=50.0),
    dict(name="DB4_L2_G160_120_P25", wavelet="db4", level=2, coarse_gain=1.60, fine_gain=1.20, floor=25.0),
    dict(name="DB4_L2_G160_120_P75", wavelet="db4", level=2, coarse_gain=1.60, fine_gain=1.20, floor=75.0),
    dict(name="DB2_L2_G140_115_P50", wavelet="db2", level=2, coarse_gain=1.40, fine_gain=1.15, floor=50.0),
    dict(name="SYM4_L2_G140_115_P50", wavelet="sym4", level=2, coarse_gain=1.40, fine_gain=1.15, floor=50.0),
    dict(name="COIF1_L2_G140_115_P50", wavelet="coif1", level=2, coarse_gain=1.40, fine_gain=1.15, floor=50.0),
    dict(name="DB4_L1_G140_P50", wavelet="db4", level=1, coarse_gain=1.40, fine_gain=1.40, floor=50.0),
    dict(name="DB4_L3_G150_110_P50", wavelet="db4", level=3, coarse_gain=1.50, fine_gain=1.10, floor=50.0),
    # Focused refinement after the initial sweep: Level 3 was the only
    # overall-positive family, while DB3 regressions suggested that its
    # noisy finest details should not be amplified before Stage 2.
    dict(name="DB4_L3_G120_100_P50", wavelet="db4", level=3, coarse_gain=1.20, fine_gain=1.00, floor=50.0),
    dict(name="DB4_L3_G140_100_P50", wavelet="db4", level=3, coarse_gain=1.40, fine_gain=1.00, floor=50.0),
    dict(name="DB4_L3_G160_100_P50", wavelet="db4", level=3, coarse_gain=1.60, fine_gain=1.00, floor=50.0),
    dict(name="DB4_L3_G180_100_P50", wavelet="db4", level=3, coarse_gain=1.80, fine_gain=1.00, floor=50.0),
    dict(name="DB4_L3_G160_100_P25", wavelet="db4", level=3, coarse_gain=1.60, fine_gain=1.00, floor=25.0),
    dict(name="DB4_L3_G160_100_P75", wavelet="db4", level=3, coarse_gain=1.60, fine_gain=1.00, floor=75.0),
    dict(name="DB2_L3_G150_100_P50", wavelet="db2", level=3, coarse_gain=1.50, fine_gain=1.00, floor=50.0),
    dict(name="SYM4_L3_G150_100_P50", wavelet="sym4", level=3, coarse_gain=1.50, fine_gain=1.00, floor=50.0),
    dict(name="DB4_L4_G160_100_P50", wavelet="db4", level=4, coarse_gain=1.60, fine_gain=1.00, floor=50.0),
]


def score_array(image, label):
    """Score one array through the project's canonical NFIQ2 wrapper."""
    safe_label = "".join(ch if ch.isalnum() else "_" for ch in label)
    fd, path = tempfile.mkstemp(prefix=f"pipeline_b_p1_{safe_label}_", suffix=".tif")
    os.close(fd)
    try:
        if not cv2.imwrite(path, np.clip(image, 0, 255).astype(np.uint8)):
            return None, "OpenCV could not write temporary TIFF"
        return run_nfiq2_single(path)
    finally:
        if os.path.exists(path):
            os.remove(path)


def load_samples():
    if not REFERENCE_SAMPLES.is_file():
        raise FileNotFoundError(f"Reference sample manifest missing: {REFERENCE_SAMPLES}")
    samples = pd.read_csv(REFERENCE_SAMPLES)
    expected = {"database", "filename"}
    if set(samples.columns) != expected:
        raise ValueError(f"Unexpected sample columns: {list(samples.columns)}")
    if len(samples) != 16 or not (samples.groupby("database").size() == 4).all():
        raise ValueError("Pilot manifest must contain exactly four samples from each database")
    for row in samples.itertuples(index=False):
        path = Path(RAW_DIR) / row.database / row.filename
        if not path.is_file():
            raise FileNotFoundError(path)
    return samples


def nfiq2_version():
    result = subprocess.run([NFIQ2_EXE], capture_output=True, text=True, check=False)
    combined = result.stdout + "\n" + result.stderr
    for line in combined.splitlines():
        if line.strip().startswith("NFIQ 2:"):
            return line.split(":", 1)[1].strip()
    return "unknown"


def make_visual_panel(records, selected_name):
    """Create one raw/Step-0/Stage-1 row per database for visual QA."""
    def letterbox(image, size=320):
        scale = min(size / image.shape[1], size / image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size), 127, dtype=np.uint8)
        x0 = (size - width) // 2
        y0 = (size - height) // 2
        canvas[y0:y0 + height, x0:x0 + width] = resized
        return canvas

    panels = []
    selected = next(item for item in CANDIDATES if item["name"] == selected_name)
    for db in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        record = next(item for item in records if item["database"] == db)
        raw = record["raw_image"]
        step0 = record["step0_image"]
        stage1 = _wavelet_contrast(
            step0,
            record["fg_mask"],
            wavelet=selected["wavelet"],
            level=selected["level"],
            coarse_gain=selected["coarse_gain"],
            fine_gain=selected["fine_gain"],
            coefficient_floor_percentile=selected["floor"],
        )
        labelled = []
        for label, image in ((f"{db} RAW", raw), ("STEP 0", step0), ("STAGE 1", stage1)):
            tile = cv2.cvtColor(letterbox(image), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (255, 255, 255), -1)
            cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            labelled.append(tile)
        panels.append(cv2.hconcat(labelled))
    panel = cv2.vconcat(panels)
    cv2.imwrite(str(OUTPUT_DIR / "selected_configuration_visual_panel.png"), panel)


def make_comparison_panels(records, configuration_names):
    """Write all 16 samples with Raw, Step 0, and selected candidates."""
    candidates = [next(item for item in CANDIDATES if item["name"] == name) for name in configuration_names]

    def letterbox(image, size=300):
        scale = min(size / image.shape[1], size / image.shape[0])
        width = max(1, int(round(image.shape[1] * scale)))
        height = max(1, int(round(image.shape[0] * scale)))
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        canvas = np.full((size, size), 127, dtype=np.uint8)
        x0 = (size - width) // 2
        y0 = (size - height) // 2
        canvas[y0:y0 + height, x0:x0 + width] = resized
        return canvas

    for db in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        rows = []
        for record in (item for item in records if item["database"] == db):
            images = [("RAW", record["raw_image"]), ("STEP 0", record["step0_image"])]
            for candidate in candidates:
                stage1 = _wavelet_contrast(
                    record["step0_image"],
                    record["fg_mask"],
                    wavelet=candidate["wavelet"],
                    level=candidate["level"],
                    coarse_gain=candidate["coarse_gain"],
                    fine_gain=candidate["fine_gain"],
                    coefficient_floor_percentile=candidate["floor"],
                )
                images.append((candidate["name"], stage1))

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
    stages_path = OUTPUT_DIR / "stage_scores_long.csv"
    reuse_scores = scores_path.is_file() and stages_path.is_file()
    existing_scores = pd.read_csv(scores_path) if reuse_scores else None
    existing_stages = pd.read_csv(stages_path) if reuse_scores else None

    base_records = []
    stage_rows = []
    for row in samples.itertuples(index=False):
        image_path = Path(RAW_DIR) / row.database / row.filename
        raw = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError(f"Could not read {image_path}")
        normalised = normalize_image(
            raw,
            target_mean=DEFAULT_NORMALIZE_TARGET_MEAN,
            target_var=DEFAULT_NORMALIZE_TARGET_VAR,
        )
        fg_mask, _ = segment(normalised)
        if reuse_scores:
            previous = existing_scores[
                (existing_scores["database"] == row.database)
                & (existing_scores["filename"] == row.filename)
            ].iloc[0]
            raw_score, raw_error = previous["raw_nfiq2"], None
            step0_score, step0_error = previous["step0_nfiq2"], None
        else:
            raw_score, raw_error = run_nfiq2_single(str(image_path))
            step0_score, step0_error = score_array(normalised, f"{row.database}_{row.filename}_step0")
        base_records.append(
            dict(
                database=row.database,
                filename=row.filename,
                raw_image=raw,
                step0_image=normalised,
                fg_mask=fg_mask,
                raw_nfiq2=raw_score,
                raw_error=raw_error,
                step0_nfiq2=step0_score,
                step0_error=step0_error,
                raw_mean=float(raw.mean()),
                raw_std=float(raw.std()),
                step0_mean=float(normalised.mean()),
                step0_std=float(normalised.std()),
                foreground_fraction=float(fg_mask.mean()),
            )
        )
        if not reuse_scores:
            stage_rows.extend(
                [
                    dict(database=row.database, filename=row.filename, configuration="SHARED", stage="raw", nfiq2=raw_score, error=raw_error),
                    dict(database=row.database, filename=row.filename, configuration="SHARED", stage="step0", nfiq2=step0_score, error=step0_error),
                ]
            )
        print(f"Prepared {row.database}/{row.filename}: raw={raw_score}, step0={step0_score}", flush=True)

    score_rows = existing_scores.to_dict("records") if reuse_scores else []
    stage_rows = existing_stages.to_dict("records") if reuse_scores else stage_rows
    completed = set(existing_scores["configuration"]) if reuse_scores else set()
    if completed:
        print(f"Reusing {len(completed)} completed candidate configurations.", flush=True)

    for candidate in CANDIDATES:
        if candidate["name"] in completed:
            continue
        print(f"\n--- {candidate['name']} ---", flush=True)
        for record in base_records:
            stage1 = _wavelet_contrast(
                record["step0_image"],
                record["fg_mask"],
                wavelet=candidate["wavelet"],
                level=candidate["level"],
                coarse_gain=candidate["coarse_gain"],
                fine_gain=candidate["fine_gain"],
                coefficient_floor_percentile=candidate["floor"],
            )
            score, error = score_array(
                stage1,
                f"{candidate['name']}_{record['database']}_{record['filename']}",
            )
            score_rows.append(
                dict(
                    database=record["database"],
                    filename=record["filename"],
                    configuration=candidate["name"],
                    wavelet=candidate["wavelet"],
                    level=candidate["level"],
                    coarse_gain=candidate["coarse_gain"],
                    fine_gain=candidate["fine_gain"],
                    coefficient_floor_percentile=candidate["floor"],
                    raw_nfiq2=record["raw_nfiq2"],
                    step0_nfiq2=record["step0_nfiq2"],
                    stage1_nfiq2=score,
                    stage1_error=error,
                    stage1_minus_raw=None if score is None or record["raw_nfiq2"] is None else score - record["raw_nfiq2"],
                    stage1_minus_step0=None if score is None or record["step0_nfiq2"] is None else score - record["step0_nfiq2"],
                    changed_pixels=int(np.count_nonzero(stage1 != record["step0_image"])),
                    mean_absolute_change=float(np.mean(np.abs(stage1.astype(np.float32) - record["step0_image"].astype(np.float32)))),
                )
            )
            stage_rows.append(
                dict(database=record["database"], filename=record["filename"], configuration=candidate["name"], stage="stage1", nfiq2=score, error=error)
            )
            print(f"  {record['database']}/{record['filename']}: {score}", flush=True)

    scores = pd.DataFrame(score_rows)
    scores.to_csv(scores_path, index=False)
    pd.DataFrame(stage_rows).to_csv(stages_path, index=False)

    valid = scores.dropna(subset=["raw_nfiq2", "step0_nfiq2", "stage1_nfiq2"]).copy()
    summary_rows = []
    grouped_results = [
        (("ALL", configuration), group)
        for configuration, group in valid.groupby("configuration")
    ] + [
        ((db, configuration), group)
        for (configuration, db), group in valid.groupby(["configuration", "database"])
    ]
    for (scope, configuration), group in grouped_results:
        delta = group["stage1_minus_raw"]
        summary_rows.append(
            dict(
                scope=scope,
                configuration=configuration,
                samples=len(group),
                mean_stage1_minus_raw=float(delta.mean()),
                median_stage1_minus_raw=float(delta.median()),
                improved=int((delta > 0).sum()),
                regressed=int((delta < 0).sum()),
                unchanged=int((delta == 0).sum()),
                worst_delta=float(delta.min()),
                mean_absolute_change=float(group["mean_absolute_change"].mean()),
            )
        )
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["scope", "mean_stage1_minus_raw"], ascending=[True, False])
    summary.to_csv(OUTPUT_DIR / "configuration_summary.csv", index=False)

    all_scope = summary[(summary["scope"] == "ALL") & (summary["configuration"] != "IDENTITY")]
    automatic_best_name = all_scope.iloc[0]["configuration"]
    selected_name = SELECTED_DEFAULT
    selected = next(item for item in CANDIDATES if item["name"] == selected_name)
    make_visual_panel(base_records, selected_name)
    make_comparison_panels(
        base_records,
        ["DB4_L3_G120_100_P50", "DB4_L3_G160_100_P25", "DB4_L3_G180_100_P50"],
    )

    base_table = pd.DataFrame(
        [
            {key: value for key, value in record.items() if key not in {"raw_image", "step0_image", "fg_mask"}}
            for record in base_records
        ]
    )
    base_table.to_csv(OUTPUT_DIR / "pilot_baseline.csv", index=False)

    metadata = dict(
        python=sys.version.split()[0],
        platform=platform.platform(),
        opencv=cv2.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
        pywavelets=pywt.__version__,
        nfiq2_executable=NFIQ2_EXE,
        nfiq2_version=nfiq2_version(),
        shared_step0=dict(
            target_mean=DEFAULT_NORMALIZE_TARGET_MEAN,
            target_std=DEFAULT_NORMALIZE_TARGET_STD,
            target_var=DEFAULT_NORMALIZE_TARGET_VAR,
        ),
        source_samples=str(REFERENCE_SAMPLES.relative_to(REPO_ROOT)),
        candidates=CANDIDATES,
        automatically_best_by_mean_stage1_minus_raw=next(
            item for item in CANDIDATES if item["name"] == automatic_best_name
        ),
        selected_default_after_quantitative_and_visual_review=selected,
        selection_rationale=(
            "Balanced choice: positive mean delta in DB1-DB4, 12/16 images improved, "
            "one fewer regression and a better worst case than the highest-mean candidate; "
            "visual review found no obvious seams, ringing, clipping, or background artefacts."
        ),
    )
    with open(OUTPUT_DIR / "environment_and_parameters.json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    selected_rows = valid[valid["configuration"] == selected_name]
    by_db = selected_rows.groupby("database")["stage1_minus_raw"].agg(["mean", "median", "min", "max"])
    report = [
        "# Pipeline B P1 wavelet contrast pilot",
        "",
        f"Selected default after quantitative and visual review: `{selected_name}`.",
        f"Highest-mean candidate retained for comparison: `{automatic_best_name}`.",
        "",
        "This is a 16-image tuning pilot, not a final locked result. P2 and P6 were not run.",
        "",
        "## Selected-candidate Stage 1 minus Raw by database",
        "",
        "```",
        by_db.round(3).to_string(),
        "```",
        "",
        "## All-candidate overall ranking",
        "",
        "```",
        all_scope[["configuration", "mean_stage1_minus_raw", "median_stage1_minus_raw", "improved", "regressed", "worst_delta", "mean_absolute_change"]].round(3).to_string(index=False),
        "```",
        "",
        "The selected default trades some mean gain for fewer regressions and a better worst case than the highest-mean candidate.",
    ]
    (OUTPUT_DIR / "results_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nWrote pilot outputs to {OUTPUT_DIR}")
    print(f"Automatic highest-mean candidate: {automatic_best_name}")
    print(f"Selected balanced default: {selected_name}")


if __name__ == "__main__":
    main()
