"""
Shared utilities for all four enhancement pipelines.

Updated 30 August 2026 (THIRD revision): normalize_image() and segment() are
back in use by ALL FOUR pipelines, as a shared PREPROCESSING stage that runs
before each pipeline's own three primary techniques. This does not reopen
the "same technique repeated across pipelines" problem that got them
dropped in the first revision (see Dataset_Problem_Analysis_and_Revised_
Pipelines.md, Sections 5-6) — preprocessing that CONDITIONS the image for
the three counted techniques, without independently solving P1/P2/P6
itself, was already established as exempt from the no-repeated-technique
rule (see orientation_field()'s docstring below, already shared by
Pipelines A, B, and C). The second revision brought normalize_image()/
segment() back for Pipeline C only, to follow its own methodological anchor
(Shams et al., 2023, "LR2") more closely; this third revision extends that
to all four pipelines, because leaving Pipeline C the only one with
conditioned input would have violated the group's own "Fair Experimental
Conditions" principle (Handover Notes, Section 14 — avoid changing multiple
experimental conditions between pipelines, since it makes the comparison
difficult to defend). Every pipeline now starts from the same conditioned
input; what differs between them is only their three counted techniques.

FOURTH revision (30 August 2026): normalize_image()'s target_var is now a
FLOOR rather than a fixed target — see its own docstring for the full
reasoning, including a two-stage bug history (an initial fix that reversed
the comparison and didn't help, then a real batch-data-caught clipping bug
in the fix that replaced it — both documented in detail there, not repeated
here). Short version of the FINAL, validated behaviour: a real batch run of
Pipeline C across all four DBs showed DB1_B (naturally higher raw contrast
than DB3/DB4) was getting its contrast compressed at this exact shared
step, before any pipeline's own technique even ran, and no amount of tuning
downstream could recover from that. The fix stops COMPRESSING images whose
raw variance is already at or above target_var (they now pass through with
no mean shift and no variance rescale at all); images below target_var are
UNCHANGED from before — they still go through the original mean-shift-and-
boost path, since that behaviour was never the problem.

Functions:
    segment(img)            -> foreground mask + block-variance map. Used by
                                all four pipelines as preprocessing.
    normalize_image(img)    -> block-wise / global intensity-normalised image.
                                Used by all four pipelines as preprocessing.
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
# Segmentation (block-variance + Otsu) — preprocessing used by all four
# pipelines (see module docstring).
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


def _fill_enclosed_background(fg_mask):
    """Any BACKGROUND block not reachable from the mask's own border by
    walking through other background blocks is an enclosed hole inside the
    foreground, and gets flipped to foreground. Implemented as a multi-seed
    flood fill starting from every border block that is background, rather
    than a fixed-size morphological closing, so it fills a hole of ANY size
    — not just ones smaller than some kernel. See segment()'s docstring for
    why this hole exists in the first place.

    NOTE: this catches holes that are topologically ENCLOSED by foreground
    on all sides, but not holes that leak out to the image border through a
    thin, irregular chain of background-misclassified blocks (found on
    DB3_B/103_5.tif — see _largest_component_hull below, which is what
    segment() actually calls now). Kept here because it's cheap, still
    correct for true enclosed holes, and run first."""
    h, w = fg_mask.shape
    bg = ~fg_mask
    visited = np.zeros((h, w), dtype=bool)
    from collections import deque
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if bg[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if bg[y, x] and not visited[y, x]:
                visited[y, x] = True
                q.append((y, x))
    while q:
        cy, cx = q.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and bg[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))
    enclosed_holes = bg & ~visited
    return fg_mask | enclosed_holes


def _largest_component_hull(fg_mask):
    """Fills in the CONVEX HULL of the largest connected foreground
    component. This is the fix for holes that _fill_enclosed_background
    can't catch — ones that leak out to the image border through a thin,
    irregular chain of low-variance blocks rather than being fully enclosed
    (found on DB3_B/103_5.tif: a hole spanning roughly half the print,
    where _fill_enclosed_background only recovered 3 of the missing
    blocks). Restricting to the LARGEST connected component first (instead
    of hulling every foreground block in the image) ignores small stray
    noise blobs far from the print, and a convex hull can never extend past
    the extreme foreground points it's built from — so it can only fill in
    concave pockets *within* the print's own footprint (like a
    misclassified whorl core), not expand outward into genuine background.
    Verified visually on 8 test images (the 5 regressed ones from a real
    batch run plus 3 previously-good baselines, including a large-background
    DB1 image): the hull consistently hugs the print's actual silhouette
    and does not eat into background corners. Superset of
    _fill_enclosed_background's effect, so segment() runs this one alone."""
    fg_u8 = fg_mask.astype(np.uint8)
    num, labels = cv2.connectedComponents(fg_u8, connectivity=4)
    if num <= 1:
        return fg_mask.copy()
    sizes = [(labels == i).sum() for i in range(1, num)]
    largest_label = 1 + int(np.argmax(sizes))
    ys, xs = np.where(labels == largest_label)
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    return fg_mask | hull_mask.astype(bool)


def segment(img, block=BLOCK, fill_holes=True):
    """Returns (fg_mask_blocks, block_var). fg_mask_blocks is a boolean array,
    one entry per BLOCK x BLOCK block, True = fingerprint, False = background.
    Used by all four pipelines as a preprocessing step (see module
    docstring).

    fill_holes=True (default) fills in any enclosed hole in the raw Otsu
    mask before returning it. Block-variance segmentation alone tends to
    mis-classify a whorl/loop CORE as background: ridges there are so
    tightly curved that a single block can span more than one ridge/valley
    cycle, which lowers that block's variance even though it's genuinely
    foreground. The size of this hole varies a lot by image, and so does its
    shape — three fixes were tried, in order:
    (1) a small 5x5 morphological closing: closed a small hole cleanly on
        DB3_B/101_1.tif, but DB3_B/103_5.tif turned out to have a hole
        spanning roughly half the fingerprint, far too big for this kernel,
        and it visibly hurt NFIQ2 on the dashboard (that image's core was
        left uncorrected by Pipeline C's Log-Gabor step, which skips
        background blocks — see pipeline_c.py).
    (2) border-flood-fill (_fill_enclosed_background): fills a hole of ANY
        size, but only if it's fully ENCLOSED by foreground on all sides.
        Tested on 103_5.tif and only recovered 3 of the missing blocks —
        that hole isn't topologically enclosed, it leaks out to the true
        background through a thin, irregular chain of low-variance blocks
        elsewhere in the grid (a real segmentation failure, not just a
        "hole").
    (3) largest-component convex hull (_largest_component_hull): fills the
        convex hull of the largest connected foreground blob. This is what
        segment() actually calls now — verified on 103_5.tif (raw fg
        fraction 0.42 -> 0.82, hull visually hugs the print silhouette) and
        on 7 other images (the other 4 regressed images from a real batch
        run, plus 3 previously-good baselines including a large-background
        DB1 image) with no visible over-inclusion into background corners.
    This is still post-processing of Otsu's own output, not a different
    segmentation technique, so it doesn't change what to cite — still Otsu
    (1979) / Hong, Wan, & Jain (1998). Pass fill_holes=False if you
    specifically need the raw, unfilled mask."""
    block_var = _block_reduce_var(img, block)
    thresh = _otsu_threshold(block_var)
    fg_mask_blocks = block_var > thresh
    if fill_holes:
        fg_mask_blocks = _largest_component_hull(fg_mask_blocks)
    return fg_mask_blocks, block_var


# ---------------------------------------------------------------------------
# Block-wise intensity normalisation — preprocessing used by all four
# pipelines (see module docstring).
# ---------------------------------------------------------------------------

def normalize_image(img, target_mean=100.0, target_var=1600.0):
    """Rescales pixel intensities so the whole image has a fixed mean/variance.
    This does NOT change contrast/ridge structure (it's a monotonic per-pixel
    rescale, so relative ridge/valley ordering is preserved) — it only puts
    every image on the same intensity scale. Used by all four pipelines as a
    preprocessing step (see module docstring).

    target_var=1600 (i.e. target std=40) was chosen empirically for this
    dataset, NOT Hong et al.'s (1998) own textbook reference values (they use
    a much smaller target_var=100, i.e. std=10). That reference value turned
    out to actively hurt this project: tested with Pipeline A's CLAHE step,
    normalizing to std=10 first and then running CLAHE (clip_limit=2.0)
    produced roughly a third less contrast than running CLAHE on the raw
    image directly (std 17 vs 49 on DB3_B/108_6.tif) — CLAHE's clip_limit
    deliberately caps how much local contrast it will manufacture, so it
    can't fully recover from a heavily pre-compressed input. std=40 was
    picked because it's close to this dataset's own typical raw std (roughly
    40-70 across subsets — see the analysis document, Section 3), so
    normalisation still standardises every image onto a common baseline
    (the actual point of this step) without pre-crushing the dynamic range
    every pipeline's own contrast technique needs to work with. If your
    pipeline's technique still behaves oddly on this baseline, test it
    explicitly rather than assuming normalisation is neutral for you too.

    PASS-THROUGH fix (30 August 2026, SECOND attempt — see below for why the
    first one didn't work): an image whose own raw variance already meets
    or exceeds target_var is now left COMPLETELY untouched (no mean shift,
    no variance rescale), instead of only skipping the variance rescale.
    Found necessary once Pipeline C's real batch results came in across all
    four DBs: DB1_B (raw std typically 60-90, well above the std=40 picked
    for DB3/DB4's typically lower contrast) was getting compressed at this
    exact step, before any pipeline's own technique even runs, and no
    amount of softening downstream parameters could recover from an
    already-halved starting contrast (verified directly on DB1_B/101_6.tif:
    raw std 72 -> normalised std 40 under the old fixed-target behaviour,
    even with every one of Pipeline C's three techniques at their gentlest
    setting, NFIQ2 still dropped 86 -> 44).

    FIRST attempt (also 30 August 2026, kept only as a cautionary note —
    replaced by the version below): tried clamping ONLY the variance
    (`effective_target_var = max(target_var, var)`) while still recentring
    the mean to target_mean=100. Looked right in isolation (output std for
    DB1_B/101_6.tif went from a crushed 40 back up to 58, closer to its raw
    72) and was pushed to the dashboard — but a real batch run showed DB1's
    regression was unchanged, even slightly worse (86% -> 88% regressed).
    Root cause, found by checking what fraction of pixels were actually
    hitting the clip(0, 255) bounds rather than just the output std: DB1's
    raw images have a HIGH mean (frequently 200-230, since large-background
    DB1 images are mostly bright platen) alongside that high variance.
    Recentring a wide, high-mean distribution down to target_mean=100 while
    KEEPING its full original spread pushed a large chunk of pixels (up to
    ~17% of the whole image on the worst cases) below 0, clipping them to
    pure black — trading one distortion (crushed contrast) for a different,
    similarly damaging one (a chunk of the image turned into flat black
    blocks). Checking only the aggregate std after clipping hid this, since
    a clipped, mean-shifted image can still report a "reasonable" std while
    having a badly mangled intensity distribution.

    Fix: don't partially process these images (rescale variance but still
    force the mean) — leave them alone entirely. If var already meets
    target_var, there's no contrast problem to fix here in the first place,
    so there's no reason to touch the mean either and risk clipping. Only
    images that actually need the variance BOOST (var < target_var —
    DB3/DB4's typical case, and the case this function was originally tuned
    against) go through the original mean-shift-and-rescale path, completely
    unchanged from before. Verified this time by checking clip-fraction, not
    just std, on DB1_B's four worst-regressed images (102_6, 101_6, 105_2,
    106_4): 0% of pixels clipped under this version vs 8-17% under the
    first attempt. Since this function is shared preprocessing for all four
    pipelines (see module docstring), this benefits whichever of A/B/D end
    up processing DB1-like high-contrast input too, not just Pipeline C."""
    img = img.astype(np.float64)
    mean = img.mean()
    var = img.var() + 1e-8
    if var >= target_var:
        # Already at or above the target contrast — leave completely alone
        # (see the docstring's SECOND-attempt note above for why even a
        # mean-only shift isn't safe here).
        return np.clip(img, 0, 255).astype(np.uint8)
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
