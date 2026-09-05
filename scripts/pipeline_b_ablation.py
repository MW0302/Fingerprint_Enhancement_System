"""Run Pipeline B's frozen cumulative NFIQ2 ablation on FVC2002 Set B.

Stages are cumulative and call Pipeline B's real internal functions:
Raw -> P1 wavelet contrast -> P1+P2 wavelet shrinkage
    -> P1+P2+P6 orientation-steered morphology.

The full run validates the 80/80/80/80 manifest, checkpoints after every
image, resumes only against matching code/data/parameter metadata, and writes
the final repository CSVs only after all 320 rows pass integrity checks.
"""

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "utils"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_b"))

from common import normalize_image, orientation_field, run_nfiq2_single, segment  # noqa: E402
from config import NFIQ2_EXE, RAW_DIR, RESULTS_DIR  # noqa: E402
from pipeline_b import (  # noqa: E402
    _orientation_steered_morphology,
    _wavelet_contrast,
    _wavelet_shrinkage_denoise,
    enhance,
)


DATABASES = ("DB1_B", "DB2_B", "DB3_B", "DB4_B")
EXPECTED_PER_DB = 80
EXPECTED_TOTAL = 320
FROZEN_PARAMETERS = {
    "wavelet": "db4",
    "wavelet_level": 3,
    "wavelet_coarse_gain": 1.60,
    "wavelet_fine_gain": 1.00,
    "wavelet_coefficient_floor_percentile": 25.0,
    "wavelet_contrast_blend": 1.0,
    "denoise_wavelet": "db4",
    "denoise_wavelet_level": 3,
    "denoise_threshold_scale": 1.00,
    "denoise_finest_levels": 1,
    "denoise_blend": 1.0,
    "denoise_noise_adaptive": True,
    "denoise_noise_reference_sigma": 5.0,
    "denoise_noise_adaptive_power": 4.0,
    "denoise_minimum_scale_factor": 0.10,
    "morph_kernel_length": 7,
    "morph_orientation_bins": 12,
    "morph_strength": 0.50,
    "morph_coherence_floor": 0.20,
    "morph_coherence_power": 1.0,
    "morph_max_darkening": 16.0,
}
RESULT_COLUMNS = (
    "database", "filename", "relative_path",
    "raw_nfiq2", "stage1_p1_nfiq2", "stage2_p1_p2_nfiq2",
    "stage3_p1_p2_p6_nfiq2", "delta_p1", "delta_p2", "delta_p6",
    "delta_final", "processing_status", "error", "raw_error", "stage1_error",
    "stage2_error", "stage3_error", "stage3_equals_enhance",
    "shape", "dtype", "processing_seconds", "scoring_seconds", "total_seconds",
)
SCORE_COLUMNS = (
    "raw_nfiq2", "stage1_p1_nfiq2", "stage2_p1_p2_nfiq2",
    "stage3_p1_p2_p6_nfiq2",
)
DELTA_COLUMNS = ("delta_p1", "delta_p2", "delta_p6", "delta_final")


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(temporary, path)


def _manifest():
    items = []
    counts = {}
    for database in DATABASES:
        paths = sorted((Path(RAW_DIR) / database).glob("*.tif"))
        counts[database] = len(paths)
        for path in paths:
            items.append({
                "database": database,
                "filename": path.name,
                "relative_path": path.relative_to(REPO_ROOT).as_posix(),
                "path": path.resolve(),
                "size": path.stat().st_size,
            })
    if counts != {database: EXPECTED_PER_DB for database in DATABASES}:
        raise RuntimeError(f"Expected 80 TIFF files per database; found {counts}.")
    if len(items) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected 320 images; found {len(items)}.")
    return items, counts


def _metadata(manifest, nfiq2_exe):
    pipeline_path = REPO_ROOT / "src" / "pipeline_b" / "pipeline_b.py"
    common_path = REPO_ROOT / "src" / "utils" / "common.py"
    manifest_text = "\n".join(
        f"{item['relative_path']}|{item['size']}" for item in manifest
    ).encode("utf-8")
    return {
        "pipeline": "pipeline_b",
        "evaluation_design": "locked cumulative ablation",
        "frozen_parameters": FROZEN_PARAMETERS,
        "pipeline_sha256": _sha256(pipeline_path),
        "common_sha256": _sha256(common_path),
        "manifest_sha256": hashlib.sha256(manifest_text).hexdigest(),
        "nfiq2_executable": str(nfiq2_exe),
        "nfiq2_sha256": _sha256(nfiq2_exe),
    }


def _validate_stage(name, image, expected_shape):
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name} is not a numpy array.")
    if image.shape != expected_shape or image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(
            f"{name} contract failed: shape={image.shape}, dtype={image.dtype}; "
            f"expected grayscale uint8 {expected_shape}."
        )


def ablate(raw):
    """Return the exact three cumulative outputs used by enhance()."""
    normalized = normalize_image(raw)
    foreground, _ = segment(normalized)
    stage1 = _wavelet_contrast(
        normalized,
        fg_mask_blocks=foreground,
        wavelet=FROZEN_PARAMETERS["wavelet"],
        level=FROZEN_PARAMETERS["wavelet_level"],
        coarse_gain=FROZEN_PARAMETERS["wavelet_coarse_gain"],
        fine_gain=FROZEN_PARAMETERS["wavelet_fine_gain"],
        coefficient_floor_percentile=FROZEN_PARAMETERS["wavelet_coefficient_floor_percentile"],
        blend=FROZEN_PARAMETERS["wavelet_contrast_blend"],
    )
    stage2 = _wavelet_shrinkage_denoise(
        stage1,
        fg_mask_blocks=foreground,
        wavelet=FROZEN_PARAMETERS["denoise_wavelet"],
        level=FROZEN_PARAMETERS["denoise_wavelet_level"],
        threshold_scale=FROZEN_PARAMETERS["denoise_threshold_scale"],
        denoise_finest_levels=FROZEN_PARAMETERS["denoise_finest_levels"],
        blend=FROZEN_PARAMETERS["denoise_blend"],
        noise_adaptive=FROZEN_PARAMETERS["denoise_noise_adaptive"],
        noise_reference_sigma=FROZEN_PARAMETERS["denoise_noise_reference_sigma"],
        noise_adaptive_power=FROZEN_PARAMETERS["denoise_noise_adaptive_power"],
        minimum_scale_factor=FROZEN_PARAMETERS["denoise_minimum_scale_factor"],
    )
    theta, coherence = orientation_field(stage2)
    stage3 = _orientation_steered_morphology(
        stage2, theta, coherence,
        fg_mask_blocks=foreground,
        kernel_length=FROZEN_PARAMETERS["morph_kernel_length"],
        orientation_bins=FROZEN_PARAMETERS["morph_orientation_bins"],
        strength=FROZEN_PARAMETERS["morph_strength"],
        coherence_floor=FROZEN_PARAMETERS["morph_coherence_floor"],
        coherence_power=FROZEN_PARAMETERS["morph_coherence_power"],
        max_darkening=FROZEN_PARAMETERS["morph_max_darkening"],
    )
    for name, image in (("Stage 1", stage1), ("Stage 2", stage2), ("Stage 3", stage3)):
        _validate_stage(name, image, raw.shape)
    return stage1, stage2, stage3


def _score_path(path, nfiq2_exe):
    # The shared helper uses a fixed temporary CSV, so evaluation is sequential.
    stale_csv = Path(tempfile.gettempdir()) / "nfiq2_single_tmp.csv"
    for attempt in range(20):
        try:
            stale_csv.unlink(missing_ok=True)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)
    score, error = run_nfiq2_single(str(path), str(nfiq2_exe))
    if score is not None and not 0 <= float(score) <= 100:
        raise RuntimeError(f"NFIQ2 score outside 0..100: {score}")
    return (None if score is None else float(score)), str(error or "")


def _process_one(item, nfiq2_exe):
    started = time.perf_counter()
    row = {column: "" for column in RESULT_COLUMNS}
    row.update({key: item[key] for key in ("database", "filename", "relative_path")})
    try:
        raw = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise RuntimeError("cv2.imread could not read image")
        _validate_stage("Raw", raw, raw.shape)
        processing_started = time.perf_counter()
        stage1, stage2, stage3 = ablate(raw)
        official = enhance(raw, params=FROZEN_PARAMETERS)
        equivalent = bool(np.array_equal(stage3, official))
        if not equivalent:
            raise RuntimeError("Stage 3 is not pixel-identical to enhance().")
        row["processing_seconds"] = time.perf_counter() - processing_started
        scoring_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="pipeline_b_ablation_") as directory:
            temp_dir = Path(directory)
            paths = [item["path"]]
            for index, image in enumerate((stage1, stage2, stage3), start=1):
                path = temp_dir / f"stage{index}.tif"
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"Could not write {path}")
                paths.append(path)
            scored = [_score_path(path, nfiq2_exe) for path in paths]
        row["scoring_seconds"] = time.perf_counter() - scoring_started
        scores = [item[0] for item in scored]
        errors = [item[1] for item in scored]
        for field, score in zip(SCORE_COLUMNS, scores):
            row[field] = "" if score is None else score
        for field, error in zip(
            ("raw_error", "stage1_error", "stage2_error", "stage3_error"), errors
        ):
            row[field] = error
        deltas = (
            None if scores[0] is None or scores[1] is None else scores[1] - scores[0],
            None if scores[1] is None or scores[2] is None else scores[2] - scores[1],
            None if scores[2] is None or scores[3] is None else scores[3] - scores[2],
            None if scores[0] is None or scores[3] is None else scores[3] - scores[0],
        )
        row.update({
            **{field: "" if value is None else value for field, value in zip(DELTA_COLUMNS, deltas)},
            "processing_status": "success" if all(score is not None for score in scores) else "nfiq2_partial",
            "error": "; ".join(
                f"{name}: {error or 'no score'}"
                for name, score, error in zip(("raw", "stage1", "stage2", "stage3"), scores, errors)
                if score is None
            ),
            "stage3_equals_enhance": True,
            "shape": "x".join(map(str, raw.shape)),
            "dtype": str(raw.dtype),
        })
    except Exception as exc:
        row["processing_status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["total_seconds"] = time.perf_counter() - started
    return row


def _read_progress(path, allowed_paths):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["relative_path"] for row in rows]
    if len(keys) != len(set(keys)) or not set(keys) <= set(allowed_paths):
        raise RuntimeError("Checkpoint contains duplicate or unexpected paths.")
    return {row["relative_path"]: row for row in rows}


def _number(row, field):
    value = row.get(field, "")
    return None if value in (None, "") else float(value)


def _validate_results(rows):
    counts = Counter(row["database"] for row in rows)
    paths = [row["relative_path"] for row in rows]
    errors = []
    if len(rows) != EXPECTED_TOTAL or len(paths) != len(set(paths)):
        errors.append("row count or uniqueness failed")
    if counts != Counter({database: EXPECTED_PER_DB for database in DATABASES}):
        errors.append(f"database counts failed: {dict(counts)}")
    for row in rows:
        if row["processing_status"] not in ("success", "nfiq2_partial"):
            errors.append(f"failed row {row['relative_path']}: {row['error']}")
            continue
        values = [_number(row, field) for field in SCORE_COLUMNS]
        if any(value is not None and not 0 <= value <= 100 for value in values):
            errors.append(f"out-of-range score: {row['relative_path']}")
        pairs = ((values[0], values[1]), (values[1], values[2]), (values[2], values[3]), (values[0], values[3]))
        for field, (before, after) in zip(DELTA_COLUMNS, pairs):
            recorded = _number(row, field)
            expected = None if before is None or after is None else after - before
            if (recorded is None) != (expected is None) or (
                recorded is not None and abs(recorded - expected) > 1e-12
            ):
                errors.append(f"delta mismatch {field}: {row['relative_path']}")
        if str(row["stage3_equals_enhance"]).lower() != "true":
            errors.append(f"enhance mismatch: {row['relative_path']}")
    if errors:
        raise RuntimeError("Integrity check failed:\n" + "\n".join(errors[:20]))
    return counts


def _stats(values):
    values = [value for value in values if value is not None]
    improved = sum(value > 0 for value in values)
    regressed = sum(value < 0 for value in values)
    return {
        "n": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "minimum": min(values),
        "maximum": max(values), "improved": improved,
        "regressed": regressed, "unchanged": len(values) - improved - regressed,
    }


def _summary_rows(rows):
    output = []
    for scope in (*DATABASES, "OVERALL"):
        subset = rows if scope == "OVERALL" else [row for row in rows if row["database"] == scope]
        for field in (*SCORE_COLUMNS, *DELTA_COLUMNS):
            stats = _stats([_number(row, field) for row in subset])
            output.append({"scope": scope, "metric": field, **stats})
    return output


def _worst_regressions(rows):
    regressions = [
        row for row in rows
        if _number(row, "delta_final") is not None
        and _number(row, "delta_final") < 0
    ]
    return [
        {field: row[field] for field in (
            "database", "filename", "relative_path", *SCORE_COLUMNS, *DELTA_COLUMNS
        )}
        for row in sorted(regressions, key=lambda item: (_number(item, "delta_final"), item["relative_path"]))
    ]


def _finish(rows, output_dir):
    counts = _validate_results(rows)
    results_root = Path(RESULTS_DIR)
    final_path = results_root / "pipeline_b_ablation.csv"
    summary_path = results_root / "pipeline_b_ablation_summary.csv"
    worst_path = results_root / "pipeline_b_worst_regressions.csv"
    _atomic_csv(final_path, rows, RESULT_COLUMNS)
    summary = _summary_rows(rows)
    summary_columns = (
        "scope", "metric", "n", "mean", "median", "minimum", "maximum",
        "improved", "regressed", "unchanged",
    )
    _atomic_csv(summary_path, summary, summary_columns)
    worst = _worst_regressions(rows)
    worst_columns = (
        "database", "filename", "relative_path", *SCORE_COLUMNS, *DELTA_COLUMNS,
    )
    _atomic_csv(worst_path, worst, worst_columns)
    report = {
        "complete": True, "rows": len(rows), "database_counts": dict(counts),
        "stage3_equals_enhance_count": sum(str(row["stage3_equals_enhance"]).lower() == "true" for row in rows),
        "available_score_counts": {
            field: sum(_number(row, field) is not None for row in rows)
            for field in SCORE_COLUMNS
        },
        "nfiq2_partial_rows": [
            {"relative_path": row["relative_path"], "error": row["error"]}
            for row in rows if row["processing_status"] == "nfiq2_partial"
        ],
        "negative_final_rows": len(worst),
        "outputs": [str(final_path), str(summary_path), str(worst_path)],
    }
    _atomic_json(output_dir / "integrity_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def run_full(manifest, metadata, nfiq2_exe, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    progress_path = output_dir / "progress.csv"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            if json.load(handle) != metadata:
                raise RuntimeError("Checkpoint metadata differs from current frozen evaluation.")
    else:
        _atomic_json(metadata_path, metadata)
    paths = [item["relative_path"] for item in manifest]
    completed = _read_progress(progress_path, paths)
    resumed = sum(row.get("processing_status") == "success" for row in completed.values())
    print(f"Validated 80/80/80/80 manifest; resuming {resumed} successful rows.", flush=True)
    for index, item in enumerate(manifest, start=1):
        previous = completed.get(item["relative_path"])
        if previous and previous.get("processing_status") in ("success", "nfiq2_partial"):
            print(f"[{index:03d}/320] RESUME {item['relative_path']}", flush=True)
            continue
        row = _process_one(item, nfiq2_exe)
        completed[item["relative_path"]] = row
        ordered = [completed[path] for path in paths if path in completed]
        _atomic_csv(progress_path, ordered, RESULT_COLUMNS)
        if row["processing_status"] in ("success", "nfiq2_partial"):
            scores = "/".join(
                "NA" if row[field] in (None, "") else str(int(float(row[field])))
                for field in SCORE_COLUMNS
            )
            detail = scores
        else:
            detail = row["error"]
        print(f"[{index:03d}/320] {row['processing_status'].upper()} {item['relative_path']} {detail} ({float(row['total_seconds']):.2f}s)", flush=True)
    rows = [completed[path] for path in paths]
    _finish(rows, output_dir)


def smoke(item, nfiq2_exe, output_dir):
    row = _process_one(item, nfiq2_exe)
    if row["processing_status"] not in ("success", "nfiq2_partial"):
        raise RuntimeError(row["error"])
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "smoke_test.json", row)
    print(json.dumps(row, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--smoke-image", default="DB1_B/101_1.tif")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR) / "pipeline_b_ablation_work")
    parser.add_argument("--nfiq2-exe", type=Path, default=Path(NFIQ2_EXE))
    args = parser.parse_args()
    nfiq2_exe = args.nfiq2_exe.resolve()
    if not nfiq2_exe.is_file():
        raise SystemExit(f"NFIQ2 executable not found: {nfiq2_exe}")
    manifest, counts = _manifest()
    print(f"Dataset counts: {counts}; total={len(manifest)}", flush=True)
    output_dir = args.output_dir.resolve()
    if args.smoke:
        wanted = f"data/raw/{args.smoke_image.replace(os.sep, '/')}"
        matches = [item for item in manifest if item["relative_path"] == wanted]
        if len(matches) != 1:
            raise SystemExit(f"Smoke image not found in manifest: {wanted}")
        smoke(matches[0], nfiq2_exe, output_dir)
    else:
        run_full(manifest, _metadata(manifest, nfiq2_exe), nfiq2_exe, output_dir)


if __name__ == "__main__":
    main()
