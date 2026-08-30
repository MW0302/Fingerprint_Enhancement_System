"""
Shared utilities for all four enhancement pipelines.

Updated 30 August 2026: only orientation_field() and run_nfiq2_single() are
still part of the required pipeline. segment() and normalize_image() were
dropped from every pipeline's enhance() during the P1/P2/P6 pivot (see
Dataset_Problem_Analysis_and_Revised_Pipelines.md, Sections 5-6) — they
targeted P5/P7, which are no longer in scope, and having all four pipelines
call the same two functions the same way would have counted as a repeated
technique across pipelines. They are left here, unused, in case you want
them for your own experimentation — don't call them from your enhance()
unless you have a specific reason tied to your own technique.

Functions:
    segment(img)            -> foreground mask + block-variance map
                                (no longer called by any pipeline)
    normalize_image(img)    -> block-wise / global intensity-normalised image
                                (no longer called by any pipeline)
    orientation_field(img)  -> per-block ridge orientation (theta) and
                                coherence, via structure tensor. Used
                                internally by Pipelines A, B, and C to steer
                                their orientation-based step.
    run_nfiq2_single(path)  -> QualityScore for one image, using the same
                                nfiq2.exe CLI pattern already verified to
                                work for the raw-baseline batch run
"""

import os
import csv
import tempfile
import subprocess

import cv2
import numpy as np

from config import NFIQ2_EXE  # noqa: E402

BLOCK = 16


# ---------------------------------------------------------------------------
# Segmentation (block-variance + Otsu) — no longer called by any pipeline;
# kept for optional individual experimentation only.
# ---------------------------------------------------------------------------

def _block_reduce_var(img, block=BLOCK):
    h, w = img.shape
    h2, w2 = (h // block) * block, (w // block) * block
    img = img[:h2, :w2].astype(np.float64)
    blocks = img.reshape(h2 // block, block, w2 // block, block)
    return blocks.var(axis=(1, 3))


def _otsu_threshold(values):
    v = values.flatten().astype(np.float64)
    hist, bin_edges = np.histogram(v, bins=256)
    hist = hist.astype(np.float64)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    total = hist.sum()
    sum_all = (hist * bin_mids).sum()
    sum_bg, w_bg, best_thresh, best_var = 0.0, 0.0, bin_mids[0], -1
    for i in range(len(hist)):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += hist[i] * bin_mids[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_all - sum_bg) / w_fg
        between_var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between_var > best_var:
            best_var = between_var
            best_thresh = bin_mids[i]
    return best_thresh


def segment(img, block=BLOCK):
    """Returns (fg_mask_blocks, block_var). fg_mask_blocks is a boolean array,
    one entry per BLOCK x BLOCK block, True = fingerprint, False = background.
    Not called by any pipeline anymore — see the module docstring."""
    block_var = _block_reduce_var(img, block)
    thresh = _otsu_threshold(block_var)
    fg_mask_blocks = block_var > thresh
    return fg_mask_blocks, block_var


# ---------------------------------------------------------------------------
# Block-wise intensity normalisation — no longer called by any pipeline;
# kept for optional individual experimentation only.
# ---------------------------------------------------------------------------

def normalize_image(img, target_mean=100.0, target_var=100.0):
    """Rescales pixel intensities so the whole image has a fixed mean/variance.
    This does NOT change contrast/ridge structure, only puts every image on
    the same intensity scale. Not called by any pipeline anymore — see the
    module docstring."""
    img = img.astype(np.float64)
    mean = img.mean()
    var = img.var() + 1e-8
    normalized = target_mean + np.sign(img - mean) * np.sqrt(
        target_var * (img - mean) ** 2 / var
    )
    return np.clip(normalized, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Ridge orientation field (structure tensor) — supporting calculation used
# internally by Pipelines A, B, and C.
# ---------------------------------------------------------------------------

def orientation_field(img, block=BLOCK):
    """Returns (theta_field, coherence_field), both shaped (H//block, W//block).
    theta_field is the dominant local ridge angle per block, in radians.
    coherence_field is 0..1 (1 = strong consistent direction, 0 = ambiguous).
    Used internally by Pipelines A, B, and C, which all need the actual
    angles (not just a mean coherence score) to steer their orientation-based
    step (P6). Pipeline D does not use this — its STFT step estimates local
    orientation directly from each window's frequency spectrum instead."""
    gx = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy

    h, w = img.shape
    h2, w2 = (h // block) * block, (w // block) * block

    def reduce_sum(a):
        a = a[:h2, :w2]
        return a.reshape(h2 // block, block, w2 // block, block).sum(axis=(1, 3))

    Gxx, Gyy, Gxy = reduce_sum(gxx), reduce_sum(gyy), reduce_sum(gxy)

    # 0.5*arctan2(2*Gxy, Gxx-Gyy) is the structure tensor's dominant
    # eigenvector direction, which points along the dominant GRADIENT (i.e.
    # ACROSS the ridge, since intensity changes fastest crossing a ridge, not
    # along it). Rotating by 90 degrees converts this to the ridge direction
    # itself, which is what every caller of this function actually needs.
    grad_theta = 0.5 * np.arctan2(2 * Gxy, (Gxx - Gyy) + 1e-8)
    theta_field = grad_theta + np.pi / 2
    theta_field = np.where(theta_field > np.pi / 2, theta_field - np.pi, theta_field)

    num = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2)
    den = Gxx + Gyy + 1e-8
    coherence_field = num / den

    return theta_field, coherence_field


# ---------------------------------------------------------------------------
# NFIQ2 scoring for a single image — reuses the exact CLI flags already
# verified against the batch raw-baseline run (see nfiq2_summary.py)
# ---------------------------------------------------------------------------

def run_nfiq2_single(image_path, nfiq2_exe=NFIQ2_EXE):
    """Runs nfiq2.exe on one image file and returns (score, error_message).
    score is a float 0-100, or None if NFIQ2 could not score the image
    (error_message will explain why, e.g. 'fingerprint area too small')."""
    tmp_csv = os.path.join(tempfile.gettempdir(), "nfiq2_single_tmp.csv")
    cmd = [nfiq2_exe, "-i", image_path, "-a", "-F", "-o", tmp_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.isfile(tmp_csv):
        return None, f"nfiq2.exe did not produce output. stderr: {result.stderr}"

    with open(tmp_csv, newline="") as f:
        row = next(csv.DictReader(f), None)

    os.remove(tmp_csv)

    if row is None:
        return None, "nfiq2.exe produced an empty output file."

    error = row.get("OptionalError") or row.get('"OptionalError"')
    try:
        return float(row["QualityScore"]), (None if error == "NA" else error)
    except (TypeError, ValueError, KeyError):
        return None, error or "Could not parse QualityScore from nfiq2 output."
