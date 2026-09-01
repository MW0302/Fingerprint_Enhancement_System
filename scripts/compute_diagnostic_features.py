"""
Computes the 7 diagnostic degradation features defined in
Dataset_Problem_Analysis_and_Revised_Pipelines.md, Section 2, for every raw
image in data/raw/DB1_B..DB4_B/. The original analyze_dataset.py that
produced these features historically is not present in this repo, so this
recomputes them from scratch, reusing segment() and orientation_field()
from src/utils/common.py exactly as pipeline_c.py already does (see its
_aggressiveness_alpha(), which reads coherence_field[fg_mask_blocks].mean()
the same way this script does for feature 7).

Output: results/per_image_metrics.csv, columns:
    file, db, contrast_std, michelson_contrast, illumination_unevenness,
    noise_sigma, sharpness_laplacian_var, foreground_ratio,
    orientation_coherence

Run with:
    python scripts/compute_diagnostic_features.py
"""

import os
import sys
import glob

import cv2
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "utils"))

from common import segment, orientation_field  # noqa: E402
from config import RAW_DIR, RESULTS_DIR, DBS  # noqa: E402

BLOCK = 16

# Immerkaer (1996) fast noise-variance estimator kernel.
_NOISE_KERNEL = np.array([[1, -2, 1],
                           [-2, 4, -2],
                           [1, -2, 1]], dtype=np.float64)


def contrast_std(img):
    return float(img.std())


def michelson_contrast(img):
    # 5th/95th percentile, not 1st/99th: diagnosed against the full 320-image
    # set, ~20% of DB4_B images have a small patch of literal 0-valued
    # (saturated) pixels inside the segmented fingerprint region itself (not
    # background leaking in -- restricting to the foreground mask doesn't
    # change it), which pins p1=0 and michelson=1.0 for those images under
    # the 1st percentile. That flattened a continuous contrast measure into
    # a near-binary "did any patch saturate" indicator, which then dominated
    # the K-Means clustering after z-scoring. Widening the margin to 5th/95th
    # requires a larger, more representative dark/light region to move the
    # statistic, so an isolated saturated patch smaller than ~5% of the image
    # no longer pins the value at the ceiling.
    p5, p95 = np.percentile(img, [5, 95])
    return float((p95 - p5) / (p95 + p5 + 1e-8))


def illumination_unevenness(img, block=BLOCK):
    h, w = img.shape
    h2, w2 = (h // block) * block, (w // block) * block
    blocks = img[:h2, :w2].astype(np.float64).reshape(
        h2 // block, block, w2 // block, block
    )
    block_means = blocks.mean(axis=(1, 3))
    return float(block_means.std() / (img.mean() + 1e-8))


def noise_sigma(img):
    h, w = img.shape
    conv = cv2.filter2D(img.astype(np.float64), -1, _NOISE_KERNEL)
    sigma = np.sqrt(np.pi / 2) * np.abs(conv).sum() / (6 * (w - 2) * (h - 2))
    return float(sigma)


def sharpness_laplacian_var(img):
    return float(cv2.Laplacian(img.astype(np.float64), cv2.CV_64F).var())


def foreground_ratio(fg_mask_blocks):
    return float(fg_mask_blocks.mean())


def orientation_coherence(coherence_field, fg_mask_blocks):
    if fg_mask_blocks.any():
        return float(coherence_field[fg_mask_blocks].mean())
    return float(coherence_field.mean())


def main():
    rows = []
    for db in DBS:
        paths = sorted(glob.glob(os.path.join(RAW_DIR, db, "*.tif")))
        for path in paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"WARNING: could not read {path}, skipping")
                continue

            fg_mask_blocks, _ = segment(img)
            theta_field, coherence_field = orientation_field(img)

            rows.append(dict(
                file=os.path.basename(path),
                db=db,
                contrast_std=contrast_std(img),
                michelson_contrast=michelson_contrast(img),
                illumination_unevenness=illumination_unevenness(img),
                noise_sigma=noise_sigma(img),
                sharpness_laplacian_var=sharpness_laplacian_var(img),
                foreground_ratio=foreground_ratio(fg_mask_blocks),
                orientation_coherence=orientation_coherence(coherence_field, fg_mask_blocks),
            ))
        print(f"{db}: {len(paths)} images processed")

    df = pd.DataFrame(rows)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "per_image_metrics.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
