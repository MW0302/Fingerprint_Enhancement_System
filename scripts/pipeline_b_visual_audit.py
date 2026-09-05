"""Create reproducible contact sheets for Pipeline B's final visual audit."""

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pipeline_b_ablation import ablate  # noqa: E402


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _panel(image, heading, target_width=250):
    scale = target_width / image.shape[1]
    resized = cv2.resize(
        image,
        (target_width, int(round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(canvas, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
    cv2.putText(canvas, heading, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)
    return canvas


def _row(row):
    raw = cv2.imread(str(ROOT / row["relative_path"]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"Could not read {row['relative_path']}")
    stage1, stage2, stage3 = ablate(raw)
    values = [row.get(field, "") for field in (
        "raw_nfiq2", "stage1_p1_nfiq2", "stage2_p1_p2_nfiq2", "stage3_p1_p2_p6_nfiq2"
    )]
    panels = [
        _panel(image, f"{name}  NFIQ2={value or 'NA'}")
        for name, value, image in zip(("Raw", "P1", "P1+P2", "P1+P2+P6"), values, (raw, stage1, stage2, stage3))
    ]
    height = max(panel.shape[0] for panel in panels)
    panels = [
        cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for panel in panels
    ]
    strip = cv2.hconcat(panels)
    label = f"{row['database']}/{row['filename']}   final delta={row.get('delta_final') or 'NA'}"
    strip = cv2.copyMakeBorder(strip, 30, 8, 0, 0, cv2.BORDER_CONSTANT, value=(225, 232, 240))
    cv2.putText(strip, label, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)
    return strip


def _sheet(rows):
    strips = [_row(row) for row in rows]
    width = max(strip.shape[1] for strip in strips)
    strips = [
        cv2.copyMakeBorder(strip, 0, 0, 0, width - strip.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        for strip in strips
    ]
    return cv2.vconcat(strips)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "pipeline_b_ablation.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "pipeline_b_visual_audit")
    args = parser.parse_args()
    rows = _read_rows(args.results)
    scored = [row for row in rows if row.get("delta_final") not in (None, "")]
    worst = sorted(scored, key=lambda row: (float(row["delta_final"]), row["relative_path"]))[:8]
    worst_paths = {row["relative_path"] for row in worst}
    rng = random.Random(2133)
    random_rows = []
    for database in ("DB1_B", "DB2_B", "DB3_B", "DB4_B"):
        candidates = [row for row in rows if row["database"] == database and row["relative_path"] not in worst_paths]
        random_rows.extend(rng.sample(candidates, 2))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "worst_regressions.png": worst,
        "random_samples.png": random_rows,
    }
    for filename, selected in outputs.items():
        path = args.output_dir / filename
        if not cv2.imwrite(str(path), _sheet(selected)):
            raise RuntimeError(f"Could not write {path}")
        print(path.resolve())


if __name__ == "__main__":
    main()
