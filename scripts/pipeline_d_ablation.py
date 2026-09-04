"""Run Pipeline D's frozen cumulative NFIQ2 ablation.

The script calls Pipeline D's existing internal helpers directly; it does not
reimplement or tune any enhancement technique. The cumulative stages are:

    Raw
    -> Stage 1: FFT high-frequency emphasis (P1)
    -> Stage 2: frequency-domain Wiener filtering (P1 + P2)
    -> Stage 3: STFT orientation-frequency reconstruction (P1 + P2 + P6)

Run the required equivalence smoke test before the full evaluation:

    python scripts/pipeline_d_ablation.py --smoke
    python scripts/pipeline_d_ablation.py --full

The full run validates an exact 80/80/80/80 dataset manifest before starting.
It writes one row atomically after every image and safely resumes successful
rows only when the saved manifest, code hash, executable and frozen parameters
all still match. Outputs live under the gitignored ``results/`` directory.

Interpretation note: the earlier fixed small-sample P2 holdout gate remains
classified as FAIL, whereas the full 320-image cumulative ablation produced a
positive mean P2 delta. Both findings are retained without retrospective
relabelling. The discrepancy may reflect greater sampling variability in the
small holdout and must not be used for retrospective parameter tuning.
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
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline_d"))

from common import (  # noqa: E402
    DEFAULT_NORMALIZE_TARGET_MEAN,
    DEFAULT_NORMALIZE_TARGET_STD,
    DEFAULT_NORMALIZE_TARGET_VAR,
    normalize_image,
    run_nfiq2_single,
    segment,
)
from config import DBS, NFIQ2_EXE, RAW_DIR, RESULTS_DIR  # noqa: E402
from pipeline_d import (  # noqa: E402
    _fft_high_frequency_emphasis,
    _frequency_domain_wiener_filter,
    _stft_orientation_frequency_reconstruct,
    enhance,
)


EXPECTED_DATABASES = ("DB1_B", "DB2_B", "DB3_B", "DB4_B")
EXPECTED_IMAGES_PER_DATABASE = 80
EXPECTED_TOTAL_IMAGES = 320

# These values reproduce pipeline_d.enhance() at the approved P6 checkpoint.
# They are deliberately explicit so the run metadata can prove that every
# resumed row used the same frozen evaluation condition.
FROZEN_PARAMETERS = {
    "fft_cutoff_ratio": 0.06,
    "fft_low_gain": 0.95,
    "fft_high_boost": 0.75,
    "fft_percentile_low": 1.0,
    "fft_percentile_high": 99.0,
    "fft_blend": 0.55,
    "wiener_noise_radius_low": 0.35,
    "wiener_noise_radius_high": 0.48,
    "wiener_noise_percentile": 25.0,
    "wiener_psd_smooth_radius": 2,
    "wiener_dc_protect_ratio": 0.03,
    "wiener_ridge_radius_low": 0.04,
    "wiener_ridge_radius_high": 0.25,
    "wiener_ridge_min_gain": 0.80,
    "wiener_min_gain": 0.35,
    "wiener_blend": 0.20,
    "wiener_pad_ratio": 0.10,
    "stft_window_size": 32,
    "stft_overlap_ratio": 0.75,
    "stft_frequency_low": 0.04,
    "stft_frequency_high": 0.25,
    "stft_radial_bandwidth": 0.025,
    "stft_angular_bandwidth": 20.0,
    "stft_min_reliability": 0.12,
    "stft_reconstruction_blend": 0.25,
    "stft_pad_mode": "reflect",
}

RESULT_COLUMNS = (
    "database",
    "filename",
    "relative_path",
    "raw_nfiq2",
    "stage1_p1_nfiq2",
    "stage2_p1_p2_nfiq2",
    "stage3_p1_p2_p6_nfiq2",
    "delta_p1",
    "delta_p2",
    "delta_p6",
    "delta_final",
    "processing_status",
    "error",
    "raw_error",
    "stage1_error",
    "stage2_error",
    "stage3_error",
    "shape",
    "dtype",
    "stage_contracts_ok",
    "processing_seconds",
    "scoring_seconds",
    "total_seconds",
)

STAGE_FIELDS = {
    "raw": "raw_nfiq2",
    "stage1_p1": "stage1_p1_nfiq2",
    "stage2_p1_p2": "stage2_p1_p2_nfiq2",
    "stage3_p1_p2_p6": "stage3_p1_p2_p6_nfiq2",
}
DELTA_FIELDS = {
    "delta_p1": "delta_p1",
    "delta_p2": "delta_p2",
    "delta_p6": "delta_p6",
    "delta_final": "delta_final",
}
EXTREME_FIELDS = {
    "delta_p1": ("raw_nfiq2", "stage1_p1_nfiq2"),
    "delta_p2": ("stage1_p1_nfiq2", "stage2_p1_p2_nfiq2"),
    "delta_p6": ("stage2_p1_p2_nfiq2", "stage3_p1_p2_p6_nfiq2"),
    "delta_final": ("raw_nfiq2", "stage3_p1_p2_p6_nfiq2"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _discover_manifest():
    if tuple(DBS) != EXPECTED_DATABASES:
        raise RuntimeError(
            f"Configured databases {tuple(DBS)!r} do not match "
            f"the required order {EXPECTED_DATABASES!r}."
        )

    manifest = []
    counts = {}
    for database in EXPECTED_DATABASES:
        directory = Path(RAW_DIR) / database
        paths = sorted(directory.glob("*.tif"), key=lambda item: item.name.casefold())
        counts[database] = len(paths)
        if len(paths) != EXPECTED_IMAGES_PER_DATABASE:
            raise RuntimeError(
                f"{database} must contain exactly {EXPECTED_IMAGES_PER_DATABASE} "
                f"TIF images; found {len(paths)}."
            )
        for path in paths:
            manifest.append(
                {
                    "database": database,
                    "filename": path.name,
                    "relative_path": path.relative_to(REPO_ROOT).as_posix(),
                    "path": path,
                }
            )

    relative_paths = [item["relative_path"] for item in manifest]
    if len(manifest) != EXPECTED_TOTAL_IMAGES:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_IMAGES} images; found {len(manifest)}."
        )
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Dataset manifest contains duplicate relative paths.")
    return manifest, counts


def _metadata(manifest, nfiq2_exe):
    pipeline_path = REPO_ROOT / "src" / "pipeline_d" / "pipeline_d.py"
    return {
        "schema_version": 1,
        "databases": list(EXPECTED_DATABASES),
        "images_per_database": EXPECTED_IMAGES_PER_DATABASE,
        "total_images": EXPECTED_TOTAL_IMAGES,
        "manifest": [item["relative_path"] for item in manifest],
        "pipeline_d_sha256": _sha256(pipeline_path),
        "nfiq2_executable": str(nfiq2_exe.resolve()),
        "nfiq2_sha256": _sha256(nfiq2_exe),
        "normalisation": {
            "target_mean": DEFAULT_NORMALIZE_TARGET_MEAN,
            "target_std": DEFAULT_NORMALIZE_TARGET_STD,
            "target_var": DEFAULT_NORMALIZE_TARGET_VAR,
        },
        "frozen_parameters": FROZEN_PARAMETERS,
    }


def _validate_uint8_stage(name, image, expected_shape):
    valid = bool(
        isinstance(image, np.ndarray)
        and image.ndim == 2
        and image.shape == expected_shape
        and image.dtype == np.uint8
        and np.isfinite(image).all()
        and int(image.min()) >= 0
        and int(image.max()) <= 255
    )
    if not valid:
        raise ValueError(
            f"{name} failed the shape/uint8/finite/0-255 output contract."
        )


def ablate(raw_image):
    """Return the three cumulative stages using Pipeline D's frozen helpers."""
    normalised = normalize_image(
        raw_image,
        target_mean=DEFAULT_NORMALIZE_TARGET_MEAN,
        target_var=DEFAULT_NORMALIZE_TARGET_VAR,
    )
    foreground_blocks, _block_variance = segment(normalised)

    stage1 = _fft_high_frequency_emphasis(
        normalised,
        cutoff_ratio=FROZEN_PARAMETERS["fft_cutoff_ratio"],
        low_gain=FROZEN_PARAMETERS["fft_low_gain"],
        high_boost=FROZEN_PARAMETERS["fft_high_boost"],
        percentile_low=FROZEN_PARAMETERS["fft_percentile_low"],
        percentile_high=FROZEN_PARAMETERS["fft_percentile_high"],
        blend=FROZEN_PARAMETERS["fft_blend"],
    )
    stage2 = _frequency_domain_wiener_filter(
        stage1,
        foreground_blocks,
        noise_radius_low=FROZEN_PARAMETERS["wiener_noise_radius_low"],
        noise_radius_high=FROZEN_PARAMETERS["wiener_noise_radius_high"],
        noise_percentile=FROZEN_PARAMETERS["wiener_noise_percentile"],
        psd_smooth_radius=FROZEN_PARAMETERS["wiener_psd_smooth_radius"],
        dc_protect_ratio=FROZEN_PARAMETERS["wiener_dc_protect_ratio"],
        ridge_radius_low=FROZEN_PARAMETERS["wiener_ridge_radius_low"],
        ridge_radius_high=FROZEN_PARAMETERS["wiener_ridge_radius_high"],
        ridge_min_gain=FROZEN_PARAMETERS["wiener_ridge_min_gain"],
        min_gain=FROZEN_PARAMETERS["wiener_min_gain"],
        blend=FROZEN_PARAMETERS["wiener_blend"],
        pad_ratio=FROZEN_PARAMETERS["wiener_pad_ratio"],
    )
    stage3 = _stft_orientation_frequency_reconstruct(
        stage2,
        foreground_blocks,
        window_size=FROZEN_PARAMETERS["stft_window_size"],
        overlap_ratio=FROZEN_PARAMETERS["stft_overlap_ratio"],
        frequency_low=FROZEN_PARAMETERS["stft_frequency_low"],
        frequency_high=FROZEN_PARAMETERS["stft_frequency_high"],
        radial_bandwidth=FROZEN_PARAMETERS["stft_radial_bandwidth"],
        angular_bandwidth=FROZEN_PARAMETERS["stft_angular_bandwidth"],
        min_reliability=FROZEN_PARAMETERS["stft_min_reliability"],
        reconstruction_blend=FROZEN_PARAMETERS["stft_reconstruction_blend"],
        pad_mode=FROZEN_PARAMETERS["stft_pad_mode"],
    )

    for name, image in (("Stage 1", stage1), ("Stage 2", stage2), ("Stage 3", stage3)):
        _validate_uint8_stage(name, image, raw_image.shape)
    return stage1, stage2, stage3


def _score_path(path, nfiq2_exe):
    # common.run_nfiq2_single() uses this fixed temporary CSV. Remove a stale
    # file left by a previously interrupted process so it cannot be mistaken
    # for the current invocation's output. Evaluation remains sequential.
    runner_csv = Path(tempfile.gettempdir()) / "nfiq2_single_tmp.csv"
    for attempt in range(20):
        try:
            if runner_csv.exists():
                runner_csv.unlink()
            break
        except PermissionError:
            if attempt == 19:
                raise
            # On Windows, NFIQ2 can retain its CSV handle very briefly after
            # the process returns. This retries only runner housekeeping; it
            # does not alter or retry a completed quality-score computation.
            time.sleep(0.1)
    score, error = run_nfiq2_single(str(path), str(nfiq2_exe))
    return score, "" if error is None else str(error)


def _score_stages(raw_path, stages, nfiq2_exe):
    scores = {}
    errors = {}
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pipeline_d_ablation_") as directory:
        temporary_directory = Path(directory)
        score_paths = {"raw": raw_path}
        for name, image in stages.items():
            path = temporary_directory / f"{name}.tif"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not write temporary NFIQ2 input {path}.")
            score_paths[name] = path

        for name in ("raw", "stage1", "stage2", "stage3"):
            scores[name], errors[name] = _score_path(score_paths[name], nfiq2_exe)
    return scores, errors, time.perf_counter() - started


def _empty_result(item):
    return {column: "" for column in RESULT_COLUMNS} | {
        "database": item["database"],
        "filename": item["filename"],
        "relative_path": item["relative_path"],
        "processing_status": "error",
        "stage_contracts_ok": False,
    }


def _process_one(item, nfiq2_exe):
    total_started = time.perf_counter()
    row = _empty_result(item)
    try:
        raw = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise ValueError("cv2.imread could not load the image as greyscale.")
        _validate_uint8_stage("Raw", raw, raw.shape)

        processing_started = time.perf_counter()
        stage1, stage2, stage3 = ablate(raw)
        processing_seconds = time.perf_counter() - processing_started
        scores, errors, scoring_seconds = _score_stages(
            item["path"],
            {"stage1": stage1, "stage2": stage2, "stage3": stage3},
            nfiq2_exe,
        )

        row.update(
            {
                "raw_nfiq2": scores["raw"],
                "stage1_p1_nfiq2": scores["stage1"],
                "stage2_p1_p2_nfiq2": scores["stage2"],
                "stage3_p1_p2_p6_nfiq2": scores["stage3"],
                "raw_error": errors["raw"],
                "stage1_error": errors["stage1"],
                "stage2_error": errors["stage2"],
                "stage3_error": errors["stage3"],
                "shape": "x".join(map(str, raw.shape)),
                "dtype": str(raw.dtype),
                "stage_contracts_ok": True,
                "processing_seconds": processing_seconds,
                "scoring_seconds": scoring_seconds,
            }
        )
        missing = [name for name, value in scores.items() if value is None]
        delta_inputs = {
            "delta_p1": (scores["raw"], scores["stage1"]),
            "delta_p2": (scores["stage1"], scores["stage2"]),
            "delta_p6": (scores["stage2"], scores["stage3"]),
            "delta_final": (scores["raw"], scores["stage3"]),
        }
        for field, (before, after) in delta_inputs.items():
            if before is not None and after is not None:
                row[field] = after - before
        if missing:
            details = "; ".join(
                f"{name}: {errors[name] or 'missing score'}" for name in missing
            )
            row["error"] = details
            row["processing_status"] = "error"
        else:
            row["processing_status"] = "success"
            warnings = [
                f"{name}: {message}"
                for name, message in errors.items()
                if message
            ]
            row["error"] = "; ".join(warnings)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["processing_status"] = "error"
    row["total_seconds"] = time.perf_counter() - total_started
    return row


def smoke_test(item, nfiq2_exe, output_directory):
    raw = cv2.imread(str(item["path"]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Could not load smoke-test image {item['relative_path']}.")
    _validate_uint8_stage("Raw", raw, raw.shape)
    stage1, stage2, stage3 = ablate(raw)
    official = enhance(raw)
    _validate_uint8_stage("enhance() output", official, raw.shape)
    equivalent = bool(np.array_equal(stage3, official))
    scores, errors, scoring_seconds = _score_stages(
        item["path"],
        {"stage1": stage1, "stage2": stage2, "stage3": stage3},
        nfiq2_exe,
    )
    payload = {
        "database": item["database"],
        "filename": item["filename"],
        "relative_path": item["relative_path"],
        "shape": list(raw.shape),
        "stage_contracts_ok": True,
        "stage3_equals_enhance": equivalent,
        "scores": scores,
        "errors": errors,
        "scoring_seconds": scoring_seconds,
    }
    _atomic_json(output_directory / "smoke_test.json", payload)
    if not equivalent:
        raise RuntimeError("Stage 3 is not pixel-identical to pipeline_d.enhance().")
    if any(value is None for value in scores.values()):
        raise RuntimeError(f"Smoke-test NFIQ2 scoring failed: {errors}")
    return payload


def _read_progress(path, manifest_paths):
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    relative_paths = [row["relative_path"] for row in rows]
    if len(relative_paths) != len(set(relative_paths)):
        raise RuntimeError("Progress CSV contains duplicate relative_path values.")
    unexpected = sorted(set(relative_paths) - set(manifest_paths))
    if unexpected:
        raise RuntimeError(f"Progress CSV contains paths outside this manifest: {unexpected}")
    return {row["relative_path"]: row for row in rows}


def _as_float(row, field):
    value = row.get(field)
    if value in (None, ""):
        return None
    return float(value)


def _descriptive(values):
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _delta_descriptive(values):
    result = _descriptive(values)
    improved = sum(value > 0 for value in values)
    declined = sum(value < 0 for value in values)
    unchanged = sum(value == 0 for value in values)
    result.update(
        {
            "improved": improved,
            "declined": declined,
            "unchanged": unchanged,
            "positive_rate": improved / len(values) if values else None,
            "negative_rate": declined / len(values) if values else None,
        }
    )
    return result


def _integrity(rows):
    counts = Counter(row["database"] for row in rows)
    relative_paths = [row["relative_path"] for row in rows]
    duplicate_paths = sorted(
        path for path, count in Counter(relative_paths).items() if count > 1
    )
    failed_rows = [
        {"relative_path": row["relative_path"], "error": row.get("error", "")}
        for row in rows
        if row.get("processing_status") != "success"
    ]
    missing_scores = []
    delta_mismatches = []
    delta_inputs = {
        "delta_p1": ("raw_nfiq2", "stage1_p1_nfiq2"),
        "delta_p2": ("stage1_p1_nfiq2", "stage2_p1_p2_nfiq2"),
        "delta_p6": ("stage2_p1_p2_nfiq2", "stage3_p1_p2_p6_nfiq2"),
        "delta_final": ("raw_nfiq2", "stage3_p1_p2_p6_nfiq2"),
    }
    for row in rows:
        scores = [_as_float(row, field) for field in STAGE_FIELDS.values()]
        if any(value is None for value in scores):
            missing_scores.append(row["relative_path"])
        for field, (before_field, after_field) in delta_inputs.items():
            before = _as_float(row, before_field)
            after = _as_float(row, after_field)
            if before is None or after is None:
                continue
            value = after - before
            recorded = _as_float(row, field)
            if recorded is None or abs(recorded - value) > 1e-12:
                delta_mismatches.append(
                    {
                        "relative_path": row["relative_path"],
                        "field": field,
                        "recorded": recorded,
                        "expected": value,
                    }
                )

    complete = bool(
        len(rows) == EXPECTED_TOTAL_IMAGES
        and all(counts[database] == EXPECTED_IMAGES_PER_DATABASE for database in EXPECTED_DATABASES)
        and not duplicate_paths
        and not failed_rows
        and not missing_scores
        and not delta_mismatches
    )
    return {
        "complete": complete,
        "total_rows": len(rows),
        "database_counts": dict(counts),
        "duplicate_relative_paths": duplicate_paths,
        "failed_rows": failed_rows,
        "missing_score_paths": missing_scores,
        "delta_mismatches": delta_mismatches,
    }


def _summarise(rows):
    summaries = {}
    summary_csv_rows = []
    scopes = list(EXPECTED_DATABASES) + ["OVERALL"]
    for scope in scopes:
        subset = rows if scope == "OVERALL" else [
            row for row in rows if row["database"] == scope
        ]
        stage_summary = {}
        delta_summary = {}
        for name, field in STAGE_FIELDS.items():
            values = [_as_float(row, field) for row in subset]
            stats = _descriptive([value for value in values if value is not None])
            stage_summary[name] = stats
            summary_csv_rows.append(
                {
                    "scope": scope,
                    "kind": "stage",
                    "metric": name,
                    **stats,
                    "improved": "",
                    "declined": "",
                    "unchanged": "",
                    "positive_rate": "",
                    "negative_rate": "",
                }
            )
        for name, field in DELTA_FIELDS.items():
            values = [_as_float(row, field) for row in subset]
            stats = _delta_descriptive([value for value in values if value is not None])
            delta_summary[name] = stats
            summary_csv_rows.append(
                {"scope": scope, "kind": "delta", "metric": name, **stats}
            )
        summaries[scope] = {"stages": stage_summary, "deltas": delta_summary}
    return summaries, summary_csv_rows


def _extremes(rows):
    payload = {}
    csv_rows = []
    for delta_name, (before_field, after_field) in EXTREME_FIELDS.items():
        ordered_regressions = sorted(
            (
                row for row in rows
                if _as_float(row, delta_name) is not None
                and _as_float(row, delta_name) < 0
            ),
            key=lambda row: (_as_float(row, delta_name), row["relative_path"]),
        )[:10]
        ordered_improvements = sorted(
            (
                row for row in rows
                if _as_float(row, delta_name) is not None
                and _as_float(row, delta_name) > 0
            ),
            key=lambda row: (-_as_float(row, delta_name), row["relative_path"]),
        )[:10]
        payload[delta_name] = {"regressions": [], "improvements": []}
        for direction, selected in (
            ("regression", ordered_regressions),
            ("improvement", ordered_improvements),
        ):
            for rank, row in enumerate(selected, start=1):
                record = {
                    "metric": delta_name,
                    "direction": direction,
                    "rank": rank,
                    "database": row["database"],
                    "filename": row["filename"],
                    "relative_path": row["relative_path"],
                    "before": _as_float(row, before_field),
                    "after": _as_float(row, after_field),
                    "delta": _as_float(row, delta_name),
                }
                payload[delta_name][direction + "s"].append(record)
                csv_rows.append(record)
    return payload, csv_rows


def full_run(manifest, metadata, nfiq2_exe, output_directory):
    metadata_path = output_directory / "run_metadata.json"
    progress_path = output_directory / "pipeline_d_ablation_progress.csv"
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            saved_metadata = json.load(handle)
        if saved_metadata != metadata:
            raise RuntimeError(
                "Existing run metadata does not match the current manifest/code/"
                "parameters/NFIQ2 executable; refusing an unsafe resume."
            )
    else:
        _atomic_json(metadata_path, metadata)

    manifest_paths = [item["relative_path"] for item in manifest]
    rows_by_path = _read_progress(progress_path, manifest_paths)
    successful_resume_rows = sum(
        row.get("processing_status") == "success" for row in rows_by_path.values()
    )
    print(
        f"Manifest validated: 80/80/80/80 = 320. "
        f"Resuming {successful_resume_rows} successful rows.",
        flush=True,
    )

    for index, item in enumerate(manifest, start=1):
        previous = rows_by_path.get(item["relative_path"])
        if previous is not None and previous.get("processing_status") == "success":
            print(f"[{index:03d}/320] RESUME {item['relative_path']}", flush=True)
            continue
        row = _process_one(item, nfiq2_exe)
        rows_by_path[item["relative_path"]] = row
        ordered_rows = [
            rows_by_path[path] for path in manifest_paths if path in rows_by_path
        ]
        _atomic_csv(progress_path, ordered_rows, RESULT_COLUMNS)
        score_text = (
            f"{row['raw_nfiq2']}/{row['stage1_p1_nfiq2']}/"
            f"{row['stage2_p1_p2_nfiq2']}/{row['stage3_p1_p2_p6_nfiq2']}"
            if row["processing_status"] == "success"
            else row["error"]
        )
        print(
            f"[{index:03d}/320] {row['processing_status'].upper()} "
            f"{item['relative_path']} {score_text} ({float(row['total_seconds']):.2f}s)",
            flush=True,
        )

    rows = [rows_by_path[path] for path in manifest_paths]
    integrity = _integrity(rows)
    summaries, summary_csv_rows = _summarise(rows)
    extremes, extreme_csv_rows = _extremes(rows)

    final_csv = output_directory / "pipeline_d_ablation.csv"
    summary_csv = output_directory / "pipeline_d_ablation_summary.csv"
    extremes_csv = output_directory / "pipeline_d_ablation_extremes.csv"
    final_json = output_directory / "pipeline_d_ablation_results.json"
    _atomic_csv(final_csv, rows, RESULT_COLUMNS)
    _atomic_csv(
        summary_csv,
        summary_csv_rows,
        (
            "scope", "kind", "metric", "count", "mean", "median",
            "standard_deviation", "minimum", "maximum", "improved",
            "declined", "unchanged", "positive_rate", "negative_rate",
        ),
    )
    _atomic_csv(
        extremes_csv,
        extreme_csv_rows,
        (
            "metric", "direction", "rank", "database", "filename",
            "relative_path", "before", "after", "delta",
        ),
    )
    _atomic_json(
        final_json,
        {
            "metadata": metadata,
            "integrity": integrity,
            "summaries": summaries,
            "extremes": extremes,
            "outputs": {
                "per_image_csv": str(final_csv.resolve()),
                "summary_csv": str(summary_csv.resolve()),
                "extremes_csv": str(extremes_csv.resolve()),
                "progress_csv": str(progress_path.resolve()),
            },
        },
    )
    print(json.dumps(integrity, indent=2), flush=True)
    print(f"Per-image CSV: {final_csv.resolve()}", flush=True)
    print(f"Summary CSV: {summary_csv.resolve()}", flush=True)
    print(f"Extremes CSV: {extremes_csv.resolve()}", flush=True)
    print(f"Results JSON: {final_json.resolve()}", flush=True)
    if not integrity["complete"]:
        raise SystemExit("Evaluation is incomplete; inspect failed_rows and missing scores.")
    return integrity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Run one DB1 equivalence/scoring check.")
    mode.add_argument("--full", action="store_true", help="Run or safely resume all 320 images.")
    parser.add_argument(
        "--smoke-image",
        default="DB1_B/101_1.tif",
        help="DB1 image relative to data/raw for --smoke (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(RESULTS_DIR) / "pipeline_d_ablation",
        help="Gitignored evaluation output directory.",
    )
    parser.add_argument(
        "--nfiq2-exe",
        type=Path,
        default=Path(NFIQ2_EXE),
        help="NFIQ2 executable (default: %(default)s).",
    )
    args = parser.parse_args()

    output_directory = args.output_dir.resolve()
    nfiq2_exe = args.nfiq2_exe.resolve()
    if not nfiq2_exe.is_file():
        raise SystemExit(f"NFIQ2 executable not found: {nfiq2_exe}")
    manifest, counts = _discover_manifest()
    print(f"Dataset counts: {counts}; total={len(manifest)}", flush=True)
    metadata = _metadata(manifest, nfiq2_exe)

    if args.smoke:
        requested = f"data/raw/{args.smoke_image.replace(os.sep, '/')}"
        matches = [item for item in manifest if item["relative_path"] == requested]
        if len(matches) != 1 or matches[0]["database"] != "DB1_B":
            raise SystemExit(f"Smoke image must resolve to one DB1 manifest entry: {requested}")
        result = smoke_test(matches[0], nfiq2_exe, output_directory)
        print(json.dumps(result, indent=2), flush=True)
    else:
        full_run(manifest, metadata, nfiq2_exe, output_directory)


if __name__ == "__main__":
    main()
