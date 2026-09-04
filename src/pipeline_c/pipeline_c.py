"""
Pipeline C — Member C: Coherence-Preserving, Multi-Filter Enhancement
(Normalisation + Segmentation preprocessing, then Homomorphic Filtering +
Coherence Diffusion + 2D Log-Gabor Filtering)

Finalised design (see Dataset_Problem_Analysis_and_Revised_Pipelines.md,
Sections 5-6, revised 29 August 2026): all four pipelines target the SAME
three evidenced problems, each with a different classical technique family,
so the group's NFIQ2 results support a genuine technique-vs-technique
comparison rather than four pipelines solving four disjoint problems:
    P1 — low global contrast (DB3)              -> Homomorphic filtering
    P2 — high random noise (DB3)                 -> Coherence diffusion
    P6 — weak/inconsistent ridge orientation
         (DB4, DB3)                              -> 2D Log-Gabor filtering

Normalisation and segmentation, THIRD revision (30 August 2026): these were
dropped from ALL FOUR pipelines in the first pivot (see the original note
below) because having all four call the identical Otsu/Hong et al. steps
would have violated the "no repeated technique across pipelines" rule. The
second revision brought them back for Pipeline C only, to follow this
pipeline's methodological anchor, Shams et al. (2023) ("LR2") — LR2's actual
published pipeline is Normalisation -> Segmentation -> Coherence Diffusion
-> Log-Gabor filtering, run on a conditioned, foreground-only image. This
third revision extends normalize_image()/segment() to ALL FOUR pipelines
(see common.py's module docstring): leaving Pipeline C the only pipeline
with conditioned input would have violated the group's own "Fair
Experimental Conditions" principle (Handover Notes — avoid changing
multiple experimental conditions between pipelines, since it makes the
comparison difficult to defend). This is still not a repeated PRIMARY
technique across pipelines — see the next paragraph.

Where this pipeline still deviates from LR2, deliberately: LR2 applies its
Log-Gabor step as a direct replacement of each windowed patch. We tried that
twice — once on a raw (un-normalised, unsegmented) image, and again (this
revision) on the now-normalised + segmented, foreground-gated image — and
both times it hurt NFIQ2 and looked blurrier than the raw input on the
dashboard (see _log_gabor_enhance's docstring for the full history and the
reasoning: a single Log-Gabor band is a narrow, sinusoidal slice of a real
ridge pattern's broadband, square-wave-like energy, so replacing the real
image with it — at any gain — trades away the sharp contrast that makes a
ridge pattern legible). Step 4 here instead ADDS a gain-scaled copy of the
bandpass response onto the real (diffused) image rather than replacing it —
this is the one place this pipeline knowingly departs from LR2's literal
method, for a tested, documented reason rather than a citation-accuracy
oversight.

A third robustness issue, found afterwards on DB1_B/102_5.tif (a
large-background DB1 image, foreground ratio ~0.30 — see P5, Section 3.1):
homomorphic filtering (step 1) attenuates low spatial frequencies and boosts
high ones, and a bright, uniform background is almost entirely low-frequency
content, so the same filtering that sharpens ridge contrast pushed the
background toward near-black once rescaled (background corner mean 254 raw
-> 27 post-homomorphic; NFIQ2 dropped 49 -> 38 on the dashboard). This isn't
a coding bug, it's an inherent consequence of running homomorphic filtering
on a full frame that includes a lot of non-ridge content — see
_homomorphic_filter's docstring and enhance()'s step 1b for the fix (a
smooth, mask-gated blend back toward the pre-filter image over background
regions only).

A fourth issue, found on DB3_B/109_7.tif and 105_3.tif: DB3's severe noise
(P2) gets amplified right alongside genuine ridge detail by step 1's
high-frequency boost, producing a speckled/blotchy texture that the
original diffusion parameters (iterations=15, kappa=15.0) didn't fully
clean up. Fixed by raising diffusion's iterations/kappa (see step 3's
comment in enhance()) — this pipeline's own P2 technique doing more work to
clean up what its P1 technique amplifies, not a new step.

A fifth issue, found by finally running a real batch across all four DBs
instead of single test images: every parameter tuned above (gamma_high,
iterations/kappa, add_gain) was tuned ONLY against DB3, and hurt DB1
badly as a result — DB1_B's raw images are already high quality (mean raw
NFIQ2 62 vs DB3's 24.6), and the same aggressive settings that rescue a
noisy DB3 print actively damage an already-clean DB1 one (mean NFIQ2
-20.2, 90% of DB1_B regressed; DB3_B was +16.5, 93.7% improved, over the
same batch run). Fixed by making those three parameters quality-adaptive
per image instead of one fixed setting for everyone — see the module-level
comment above _COHERENCE_FULL_GENTLE, just below the imports, for the full
mechanism and the real correlation data behind it.

A sixth issue, found after common.py's normalize_image() default changed
from target_var=1600 (std=40) to target_var=100 (std=10) (4 September
2026 — see common.py's own SIXTH revision for why: that std=40 default
turned out to only have looked necessary because of the mean-crush bug
FIFTH-revision-fixed there, not because of variance-boosting itself). At
std=10, every image in this dataset already clears the pass-through
threshold (raw std is never below ~22 here), so normalize_image() is now a
full no-op for all 320 images — Step 0 no longer pre-boosts DB3's
naturally-low contrast toward std=40, and no longer pre-cushions DB1's
naturally-high contrast either. This directly changed what Step 1
(homomorphic filtering) sees as input for every image, and a full-batch
re-run with every one of this pipeline's parameters left untouched showed
it: DB2/DB3/DB4 all improved on their own just from Step 0 changing
(DB3_B especially: +13.78 -> +17.05 mean delta), but DB1_B got slightly
WORSE (-6.84 -> -7.47) — the gentle end of _HOMOMORPHIC_GAMMA_HIGH_RANGE
(1.2) still applied a mild boost even to DB1's cleanest, highest-coherence
images, and without Step 0's old cushioning that boost now landed on
genuinely raw (not pre-softened) high-contrast input. Fixed by lowering
that gentle endpoint to 1.0 — the true no-op point for gamma_high (no
high-frequency boost at all, only gamma_low's fixed illumination
suppression still runs) — and re-validated with a full batch: DB1_B
-7.47 -> -6.39 (the best DB1_B result found across this project's entire
tuning history), DB2_B +10.07 -> +11.43, with DB3_B/DB4_B essentially
unchanged (+17.25/+15.59, within noise of the pre-change +17.05/+15.68).
The aggressive endpoint (2.0) and every other parameter (diffusion
iterations/kappa, Log-Gabor add_gain, the coherence thresholds themselves)
were checked and left alone: mean foreground coherence on the new,
effectively-raw Step 0 output correlates with raw NFIQ2 at r=0.720 (DB
means 0.67/0.70/0.57/0.54), nearly identical to the r=0.72 this scheme was
originally built on, since orientation_field()'s coherence measure is
largely invariant to a monotonic intensity rescale — so the alpha-adaptive
scaling itself didn't need retuning, only the one range it was scaling
that was directly exposed to Step 0's raw-vs-normalised input shift.
Diffusion and Log-Gabor are comparatively insulated from this because they
run on _homomorphic_filter's own percentile-rescaled output, which is
already restandardised to a consistent 0-255 range regardless of what
Step 0 handed it.

Normalisation and segmentation are PREPROCESSING here, not a fourth primary
technique: like orientation_field() (a supporting calculation used
internally, see below), they condition the image for the three techniques
that follow rather than independently solving one of P1/P2/P6 themselves.
Pipeline C's technique count for the marking rubric is still exactly three:
homomorphic filtering (P1), coherence diffusion (P2), Log-Gabor filtering
(P6).

Segmentation's foreground mask is used INTERNALLY only, to decide which
blocks the Log-Gabor step (step 4) is allowed to add its bandpass boost to
— background blocks get no boost either way, since there's no real ridge
signal there to reinforce. This does NOT blank or recolour the background in
the final output image; the visible output stays a full-frame greyscale
image, same as the other three pipelines, so NFIQ2 (which was
tuned/validated against un-masked images across this project) sees a
consistent kind of input from every pipeline.

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Hong, Wan, & Jain (1998) — block-wise normalisation, Otsu segmentation,
                                and structure-tensor orientation field (all
                                supporting/preprocessing steps here, not one
                                of the three primary techniques)
    Oppenheim, Schafer, & Stockham (1968) — homomorphic filtering
    Perona & Malik (1990) — anisotropic / coherence-enhancing diffusion
    Field (1987) — Log-Gabor filtering
    Shams et al. (2023) — methodological inspiration (this is the group's own
                           Literature Review 2 — anchor your lit review here;
                           also the direct source for using normalisation +
                           segmentation as preprocessing ahead of diffusion
                           and Log-Gabor filtering)

Pipeline, in order:
    0a. Block-wise intensity normalisation (preprocessing; Hong et al., 1998)
    0b. Otsu/block-variance segmentation (preprocessing; Hong et al., 1998) —
        produces the foreground mask used internally by step 4
    1. Homomorphic filtering (P1 — contrast)
    2. Orientation field (supporting calculation; steers steps 3 & 4)
    3. Coherence-enhancing anisotropic diffusion (P2 — noise)
    4. Ridge-frequency estimation + 2D Log-Gabor filtering (P6 — orientation)

Run this file directly (`python pipeline_c.py`) to sanity-check it against
one test image before running it over the full dataset.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import (  # noqa: E402
    orientation_field, normalize_image, segment,
    DEFAULT_NORMALIZE_TARGET_MEAN, DEFAULT_NORMALIZE_TARGET_VAR,
)
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


# -----------------------------------------------------------------------
# Quality-adaptive aggressiveness (added 30 August 2026, FOURTH revision,
# after a full-dataset batch run — see enhance()'s Step 0c for how this is
# used, and used_defaults's docstring for the full story).
#
# All three of this pipeline's own techniques (homomorphic gamma_high,
# diffusion iterations/kappa, Log-Gabor add_gain) had been tuned ONLY
# against DB3 (P1/P2's own source subset, and by far the noisiest/lowest-
# contrast one — see Section 3). Running a real batch across all four DBs
# exposed the cost of that: DB3_B improved a lot (mean NFIQ2 +16.5, 93.7%
# of images improved) and DB4_B improved too (+9.3, 81.25%), but DB1_B —
# whose raw images are already clean (mean raw NFIQ2 62 vs DB3's 24.6) —
# got WORSE on 90% of its images (mean -20.2; worst case 108_8.tif dropped
# 79 -> 25). DB2_B (medium quality) was roughly a wash (-1.0, 53.75%
# regressed). The same "aggressive" parameters that rescue a noisy DB3
# print actively damage an already-clean DB1 print: there's no real
# contrast/noise/orientation problem there for homomorphic/diffusion/
# Log-Gabor to fix, so all three techniques mostly just add processing
# artifacts on top of a print that didn't need help.
#
# Fix: scale each technique's strength by how degraded the INPUT actually
# is, per image, instead of using one fixed setting for every image
# regardless of subset. The degradation signal used is the mean
# orientation-field coherence (already computed for steps 2-4, so this is
# free) over the image's own foreground blocks, measured on the
# normalised-but-not-yet-filtered image (Step 0c, before homomorphic
# filtering runs) — high coherence means the ridge orientation is already
# clean and consistent (a good print), low coherence means it's noisy/
# ambiguous (a bad print). Checked against the real batch data above:
# mean foreground coherence correlates with raw NFIQ2 at r=0.72 across all
# 319 scored images (DB1 mean 0.67, DB2 0.70, DB3 0.57, DB4 0.54) — a
# strong enough signal to steer aggressiveness by.
#
# Scope, deliberately kept narrow: only the three parameters that were
# actually tuned this session for DB3 (gamma_high, iterations+kappa,
# add_gain) are made adaptive. Parameters that define a technique's basic
# SHAPE rather than its strength — homomorphic's cutoff/sharpness and
# gamma_low, diffusion's dt, Log-Gabor's window/sigma_onf/sigma_theta_deg/
# field_blur_sigma — are left fixed. Making every parameter adaptive at
# once would be harder to validate and to explain in the report; if DB1
# still isn't good enough after this first pass, those are the next things
# to consider.
_COHERENCE_FULL_GENTLE = 0.68   # foreground coherence at/above this -> alpha=0 (gentlest settings)
# 0.50 originally (matching DB3's own mean coherence, 0.57, with a little
# margin) -> widened to 0.40 after a teammate report on DB1_B/105_8.tif
# (raw NFIQ2 64, a genuinely good print): its mean foreground coherence
# (0.487) landed just under the old 0.50 cutoff, so it got snapped to
# alpha=1.0 — the SAME full-aggressive settings meant for DB3/DB4's worst
# images — and came out with the same fake-wavy-texture artifact as
# DB4_B/108_2.tif (NFIQ2 64 -> 42). Per-block coherence is noisy (see the
# heatmap-style check in this fix's investigation — a single image's raw
# per-block values swing widely even where the print itself looks fine), so
# a single image's mean can land close to a DB-level average by chance
# without that image actually being as degraded as a typical DB3/DB4 print.
# Widening the band (0.50 -> 0.40) gives more room before alpha snaps to
# 1.0, trading a little more of DB3/DB4's average aggressiveness for fewer
# of these misclassified, over-processed edge cases. RESTORED to 0.50 after
# a real 320-image batch run: DB1_B's regression didn't move (still 86%
# regressed) while DB3_B/DB4_B's gains dropped measurably (see the note
# above _DIFFUSION_ITERATIONS_RANGE for the actual numbers) — the edge-case
# misclassification this widening targeted (DB1_B/105_8.tif) is handled
# well enough by the per-block/per-pixel confidence gating kept in
# _log_gabor_enhance and _coherence_diffusion; widening this threshold on
# top of that cost more (blanket, dataset-wide) than it fixed (a handful of
# borderline images).
_COHERENCE_FULL_AGGRESSIVE = 0.50  # foreground coherence at/below this -> alpha=1 (current DB3-tuned settings)
# RE-VALIDATED (4 September 2026), both this and _COHERENCE_FULL_GENTLE below,
# following the parameter audit in docs/pipeline_c_parameter_audit.md: full
# 320-image, 4-DB sweep of _COHERENCE_FULL_AGGRESSIVE over {0.45, 0.50, 0.55,
# 0.60, 0.65} (gentle fixed at 0.68) found 0.50 -- the value already in use --
# beats every other candidate on overall mean delta_total (9.445, vs. 9.235 at
# 0.45 and a monotonic decline above 0.50 down to 7.824 at 0.65). Then, with
# aggressive fixed at that confirmed 0.50, a second sweep of
# _COHERENCE_FULL_GENTLE over {0.55, 0.60, 0.65, 0.68, 0.75} found 0.68 --
# also already in use -- likewise best (9.445; DB3 keeps climbing with a
# higher gentle threshold, 13.1 -> 18.25, but DB1/DB2 start losing ground past
# 0.68, netting a lower overall past that point). Both thresholds confirmed
# optimal as-is among the candidates tested -- no change made.

# (gentle, aggressive) endpoints for each adaptive parameter. "aggressive"
# is exactly this session's DB3-tuned default; "gentle" is a light-touch
# value chosen so the technique still visibly runs (it's still one of this
# pipeline's three graded techniques) without actively damaging a print
# that was already clean.
_HOMOMORPHIC_GAMMA_HIGH_RANGE = (1.0, 2.0)  # gentle endpoint 1.2 -> 1.0 (4 Sep
# 2026, "a sixth issue" above) once normalize_image()'s shared default
# dropped to std=10 and stopped cushioning DB1's naturally high contrast:
# gamma_high=1.0 is the true no-op point (no high-frequency boost at all),
# whereas 1.2 still mildly over-boosted DB1's cleanest images once they
# started arriving genuinely raw instead of pre-softened. Full-batch
# verified: DB1_B -7.47 -> -6.39, DB2_B +10.07 -> +11.43, DB3_B/DB4_B
# unchanged within noise (+17.05->+17.25 / +15.68->+15.59).
# Diffusion's aggressive endpoint was pulled back from (25, 25.0) to (15,
# 15.0) for one revision, then RESTORED to (25, 25.0) here after a real
# 320-image, all-four-DB batch run showed the pullback wasn't paying for
# itself: DB1_B's regression was completely unmoved by it (-13.9 -> -14.5
# mean delta, 86% regressed either way), while DB3_B's improvement dropped
# by nearly 40% (+15.8 -> +9.7 mean delta, 89.9% -> 78.5% improved) and
# DB4_B's dropped too (+12.4 -> +10.75). The wavy-artifact failure mode this
# pullback targeted (see _coherence_diffusion's SIXTH-revision docstring
# note, and DB4_B/108_2.tif / DB1_B/105_8.tif) is real, but it turned out to
# affect a small enough number of images that softening the ceiling for
# EVERY low-coherence image cost far more than it fixed. The two per-pixel/
# per-block CONFIDENCE-GATING fixes (Log-Gabor's FIFTH revision, diffusion's
# SIXTH) are kept — they're targeted at exactly the unreliable pixels/blocks
# causing the artifact, at near-zero cost elsewhere — but blanket-softening
# the ceiling for every image in the low-coherence half of the dataset was
# the wrong lever. DB1's actual dominant problem was never really about
# diffusion or Log-Gabor being too strong (see normalize_image()'s
# target_var in common.py instead, which crushes DB1's naturally higher raw
# contrast before any of these three techniques even run — softening these
# three parameters can't reach a problem that happens upstream of them).
# UPDATE (4 September 2026): that upstream problem is fixed now — see "a
# sixth issue" above — and normalize_image() no longer touches DB1 at all
# (100% pass-through at the new std=10 default). DB1's residual regression
# after that fix traced to _HOMOMORPHIC_GAMMA_HIGH_RANGE instead (see its
# own comment above), not to diffusion/Log-Gabor here — this paragraph's
# original conclusion (soften ranges elsewhere != DB1's real problem) still
# held, just for a different, now-fixed upstream cause.
_DIFFUSION_ITERATIONS_RANGE = (8, 25)
_DIFFUSION_KAPPA_RANGE = (10.0, 25.0)
_LOG_GABOR_ADD_GAIN_RANGE = (1.2, 1.5)
# gentle=1.2, aggressive=1.5: best in tested grid {1.5,2.0,2.5,3.0,3.5} x
# {0.4,0.6,0.8,1.0,1.2}, but aggressive trended monotonically better toward
# the lower bound across all three sweep rounds -- this is a boundary
# approximation, not a confirmed extremum. Not pursued further (time
# constraint, 4 September 2026). If revisited: extend the grid below 1.5,
# but do not let aggressive drop below gentle (breaks the "more coherence +
# less noise = more gain" design assumption _aggressiveness_alpha depends
# on). Across every (gentle, aggressive) combination swept, delta_p6 (this
# step's own marginal NFIQ2 contribution, stage3 - stage2) was NEGATIVE
# overall every time -- Log-Gabor never became a net-positive contributor
# in this dataset regardless of add_gain, this tuning pass included.


def _aggressiveness_alpha(coherence_field, fg_mask_blocks):
    """Returns alpha in [0, 1]: 0 = gentlest settings (clean input), 1 =
    this session's DB3-tuned aggressive settings (degraded input). See the
    module-level comment above _COHERENCE_FULL_GENTLE for the full
    reasoning and the real batch data behind it."""
    if fg_mask_blocks is not None and fg_mask_blocks.any():
        mean_coherence = float(coherence_field[fg_mask_blocks].mean())
    else:
        mean_coherence = float(coherence_field.mean())
    span = _COHERENCE_FULL_GENTLE - _COHERENCE_FULL_AGGRESSIVE
    alpha = (_COHERENCE_FULL_GENTLE - mean_coherence) / span
    return float(np.clip(alpha, 0.0, 1.0))


def _lerp(range_, alpha):
    gentle, aggressive = range_
    return gentle + alpha * (aggressive - gentle)


def _homomorphic_filter(image, cutoff=0.06, gamma_low=0.5, gamma_high=2.0, sharpness=1.0):
    """
    Step 1: homomorphic filtering (Oppenheim, Schafer, & Stockham, 1968).

    Models an image as illumination(x,y) * reflectance(x,y), where
    illumination varies slowly (low spatial frequency) and reflectance
    carries the fine ridge/valley detail (high spatial frequency). Taking a
    log turns that product into a sum, which a single frequency-domain
    filter can then split apart: attenuate the low-frequency illumination
    term (gamma_low < 1) while boosting the high-frequency reflectance term
    (gamma_high > 1), then undo the log with exp().

    This directly targets P1 (DB3's systematically low, narrow-SD global
    contrast — Section 3.2) and, as a side effect, P3 (uneven illumination),
    since both are driven by the same low-frequency component this filter
    suppresses. It runs on the normalised image (step 0a) so DB1/DB4's wider
    frame-to-frame brightness swings don't also get baked into the gamma
    curve here, and before orientation estimation (step 2) so the downstream
    diffusion and Log-Gabor steps are steered using a contrast-corrected
    image rather than DB3's original flat one.

    NOTE: this function's raw output should not be used directly on images
    with a large background area (e.g. DB1 — see P5, Section 3.1) — a
    bright, uniform background is almost entirely low-frequency content, so
    it gets attenuated toward near-black by the same gamma_low term that
    correctly suppresses illumination in the foreground. This is a real
    property of homomorphic filtering, not a bug in this implementation.
    enhance() (step 1b) fixes this by blending this function's output back
    toward the pre-filter image over background regions, using
    fg_mask_blocks. Call this function directly (e.g. for testing) only if
    you're prepared to handle that yourself.
    """
    img = image.astype(np.float64) + 1.0  # +1 avoids log(0) on pure-black pixels
    log_img = np.log(img)

    h, w = image.shape
    Fshift = np.fft.fftshift(np.fft.fft2(log_img))

    u = (np.arange(w) - w // 2) / w
    v = (np.arange(h) - h // 2) / h
    U, V = np.meshgrid(u, v)
    D = np.sqrt(U ** 2 + V ** 2)

    # High-pass emphasis transfer function: ~gamma_low near DC (attenuate
    # illumination), rising smoothly to ~gamma_high at high frequencies
    # (boost reflectance/detail); `cutoff` sets where the transition sits,
    # `sharpness` how quickly it happens.
    H = (gamma_high - gamma_low) * (1 - np.exp(-sharpness * (D ** 2) / (cutoff ** 2 + 1e-12))) + gamma_low

    filtered = np.fft.ifft2(np.fft.ifftshift(Fshift * H))
    result = np.exp(np.real(filtered)) - 1.0

    # Rescale to 0-255 using the 1st/99th percentile, not the true min/max:
    # a handful of outlier pixels (e.g. DB1's bright platen corners) can
    # otherwise dominate a min/max stretch and crush the entire ridge/valley
    # structure into a narrow dark band. Same robust-percentile idea already
    # used for Michelson contrast in Section 2 of the dataset analysis.
    lo, hi = np.percentile(result, [1, 99])
    if hi > lo:
        result = (result - lo) / (hi - lo) * 255.0
    else:
        result = result - result.min()
    return np.clip(result, 0, 255).astype(np.uint8)


def _coherence_diffusion(image, theta_field, coherence_field, iterations=15, dt=0.2, kappa=15.0,
                          confidence_ceiling=0.45):
    """
    Coherence-enhancing anisotropic diffusion (Perona & Malik, 1990), steered
    by the local ridge orientation from Step 2.

    At every pixel we build a 2x2 diffusion tensor whose two eigen-directions
    are the local ridge direction (theta_field) and the direction
    perpendicular to it. The two matching eigenvalues (c_along, c_across)
    control how freely intensity is allowed to blur in each direction:
        - along the ridge:  smooth A LOT. A real ridge is fairly uniform
          along its own length, so blurring along it removes noise without
          destroying real structure.
        - across the ridge: smooth CAUTIOUSLY, using a Perona-Malik
          conductance function that backs off wherever the cross-ridge
          gradient is large (i.e. a real ridge/valley boundary), so that
          boundary is preserved rather than blurred away.
    Wherever coherence_field says the orientation estimate is unreliable, the
    tensor is blended back toward isotropic diffusion, so we never smooth
    confidently along a direction we can't actually trust.

    SIXTH revision (30 August 2026), per-pixel confidence damping: the
    isotropic fallback above stops diffusion from smoothing confidently in
    the WRONG direction when coherence is low, but it does nothing about
    HOW MUCH smoothing still happens — even isotropic diffusion, run for
    many iterations at a generous kappa, can turn genuinely ambiguous
    texture into a smooth, organised-looking but fake periodic pattern
    (found on DB4_B/108_2.tif and DB1_B/105_8.tif, both reported directly by
    a teammate from the dashboard: NFIQ2 64 -> 42 on the second one, and in
    both cases the wavy artifact was already visible right after this step,
    confirmed by rendering stages separately). This mirrors the same fix
    already applied to Log-Gabor's add_gain (_log_gabor_enhance's FIFTH
    revision) — scale the ENTIRE diffusion tensor (both along and across)
    by a per-pixel confidence derived from coherence_field, so a pixel
    sitting on an unreliable orientation estimate gets little to no
    diffusion at all, of either kind, rather than a full dose of isotropic
    smoothing. confidence reaches 1 (full diffusion, as before) once
    coherence hits confidence_ceiling, and ramps down to 0 (no smoothing)
    as coherence drops toward 0 — same shape and same default ceiling as
    the Log-Gabor fix, for consistency.
    """
    img = image.astype(np.float32)
    h, w = img.shape

    # theta_field / coherence_field are one value per 16x16 block (see
    # orientation_field in common.py) — upsample to one value per pixel so
    # every pixel is steered by its local block's orientation.
    theta_px = cv2.resize(theta_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = cv2.resize(coherence_field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    coherence_px = np.clip(coherence_px, 0.0, 1.0)

    # Per-pixel overall diffusion confidence (SIXTH revision) — separate
    # from the directional (isotropic-fallback) use of coherence_px below.
    confidence_px = np.clip(coherence_px / confidence_ceiling, 0.0, 1.0)

    cos_t = np.cos(theta_px)
    sin_t = np.sin(theta_px)

    for _ in range(iterations):
        Iy, Ix = np.gradient(img)  # np.gradient returns (d/d(row), d/d(col)) = (dI/dy, dI/dx)

        g_along = Ix * cos_t + Iy * sin_t     # gradient measured along the ridge direction
        g_across = -Ix * sin_t + Iy * cos_t   # gradient measured across the ridge direction

        # Perona-Malik conductance: c(x) = exp(-(x/kappa)^2).
        # small gradient -> conductance near 1 (smooth freely)
        # large gradient -> conductance near 0 (treat as a real edge, stop)
        c_across = np.exp(-(g_across / kappa) ** 2)
        c_along = np.exp(-(g_along / (kappa * 4.0)) ** 2)  # much more tolerant: keep smoothing along the ridge

        # low coherence = orientation estimate is shaky here -> fall back
        # toward isotropic (c_along ~= c_across) instead of trusting it
        c_along = coherence_px * c_along + (1 - coherence_px) * c_across

        # 2x2 diffusion tensor per pixel, built from the two eigenvalues
        # (c_along, c_across) and eigenvectors (ridge direction, perpendicular).
        # confidence_px (SIXTH revision) additionally scales the WHOLE
        # tensor, so a pixel with an unreliable orientation estimate gets
        # little to no diffusion of either kind, not just isotropic-instead
        # -of-anisotropic diffusion.
        d11 = confidence_px * (c_along * cos_t ** 2 + c_across * sin_t ** 2)
        d22 = confidence_px * (c_along * sin_t ** 2 + c_across * cos_t ** 2)
        d12 = confidence_px * (c_along - c_across) * cos_t * sin_t

        Fx = d11 * Ix + d12 * Iy
        Fy = d12 * Ix + d22 * Iy

        _, dFx_dx = np.gradient(Fx)
        dFy_dy, _ = np.gradient(Fy)

        img = img + dt * (dFx_dx + dFy_dy)

    return np.clip(img, 0, 255).astype(np.uint8)


def _estimate_ridge_frequency(image, theta_field, block=16, window_len=32, window_width=16,
                               min_period=3, max_period=25, default_freq=1.0 / 9.0):
    """
    Local ridge-frequency estimation, following the x-signature procedure in
    Hong, Wan, & Jain (1998).

    For each block, an oriented window aligned with the block's ridge
    direction (theta_field) is sampled: `window_width` parallel lines run
    ALONG the ridge, each `window_len` samples long, and are averaged pixel-
    for-pixel to build a 1D "x-signature" that profiles intensity ACROSS the
    ridges/valleys (averaging along the ridge cancels most of the noise a
    single scanline would carry). The average spacing between consecutive
    peaks in that profile is the local ridge period in pixels; frequency is
    1/period.

    Blocks whose x-signature doesn't yield at least two peaks with a
    plausible fingerprint period (e.g. background, or a low-coherence block
    where "the ridge direction" isn't meaningful) fall back to
    `default_freq` rather than reporting a wild estimate — a bad frequency
    would otherwise mistune the Log-Gabor filter for that whole block in
    step 5.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape
    freq_field = np.full_like(theta_field, default_freq, dtype=np.float64)

    half_len = window_len // 2
    half_width = window_width // 2
    xs = np.arange(-half_width, half_width)
    ys = np.arange(-half_len, half_len)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")  # shape (window_len, window_width)

    for by in range(nby):
        for bx in range(nbx):
            theta = theta_field[by, bx]
            cy = by * block + block // 2
            cx = bx * block + block // 2

            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # xv is the across-ridge offset, yv the along-ridge offset;
            # rotate that ridge-aligned frame into image (x, y) coordinates.
            sample_x = cx + xv * (-sin_t) + yv * cos_t
            sample_y = cy + xv * cos_t + yv * sin_t

            if (sample_x.min() < 0 or sample_x.max() >= w - 1 or
                    sample_y.min() < 0 or sample_y.max() >= h - 1):
                continue  # oriented window falls off the image; keep default_freq

            x0 = np.floor(sample_x).astype(int)
            y0 = np.floor(sample_y).astype(int)
            fx = sample_x - x0
            fy = sample_y - y0
            x1 = np.clip(x0 + 1, 0, w - 1)
            y1 = np.clip(y0 + 1, 0, h - 1)
            x0 = np.clip(x0, 0, w - 1)
            y0 = np.clip(y0, 0, h - 1)

            Ia, Ib = img[y0, x0], img[y0, x1]
            Ic, Id = img[y1, x0], img[y1, x1]
            sampled = (Ia * (1 - fx) * (1 - fy) + Ib * fx * (1 - fy) +
                       Ic * (1 - fx) * fy + Id * fx * fy)

            x_signature = sampled.mean(axis=0)  # average along the ridge -> profile across ridges

            peaks = [k for k in range(1, len(x_signature) - 1)
                     if x_signature[k] > x_signature[k - 1] and x_signature[k] >= x_signature[k + 1]]

            if len(peaks) >= 2:
                period = float(np.mean(np.diff(peaks)))
                if min_period <= period <= max_period:
                    freq_field[by, bx] = 1.0 / period
                # else: implausible spacing for a real ridge -> keep default_freq

    return freq_field


def _log_gabor_filter_2d(shape, freq0, theta0, sigma_onf=0.65, sigma_theta_deg=20.0):
    """
    Builds one 2D Log-Gabor transfer function H(u,v) over a frequency-domain
    grid of the given shape, tuned to a single local ridge frequency (freq0,
    cycles/pixel, from step 4) and orientation (theta0, radians, from step 2)
    — Field (1987).

    A Log-Gabor filter is defined directly as a Gaussian in LOG-frequency
    space rather than linear frequency space (the ordinary Gabor filters
    Pipeline B already uses). That avoids the DC bias / limited bandwidth a
    linear-domain Gaussian carries, so the same filter shape stays well
    behaved whether it's tuned to DB3's tight high-frequency ridges or DB4's
    coarser ones — useful here since Pipeline C's own problem set (P2, P6)
    spans both.
    """
    rows, cols = shape
    u = (np.arange(cols) - cols // 2) / cols
    v = (np.arange(rows) - rows // 2) / rows
    U, V = np.meshgrid(u, v)
    radius = np.sqrt(U ** 2 + V ** 2)
    radius[rows // 2, cols // 2] = 1.0  # placeholder so log() below doesn't hit log(0) at DC

    theta_grid = np.arctan2(V, U)

    # radial component: Gaussian over log(radius/freq0)
    log_radius_ratio = np.log(radius / freq0 + 1e-12)
    radial = np.exp(-(log_radius_ratio ** 2) / (2 * np.log(sigma_onf) ** 2))
    radial[rows // 2, cols // 2] = 0.0  # zero the DC term explicitly

    # angular component: Gaussian around theta0. Ridge orientation is a
    # direction, not a vector (theta0 and theta0 + pi describe the same
    # ridge), so both lobes of the frequency plane 180 degrees apart get
    # folded onto the same angular response.
    sigma_theta = np.radians(sigma_theta_deg)
    dtheta = theta_grid - theta0
    dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))  # wrap to [-pi, pi]
    dtheta = np.minimum(np.abs(dtheta), np.abs(np.abs(dtheta) - np.pi))
    angular = np.exp(-(dtheta ** 2) / (2 * sigma_theta ** 2))

    H = radial * angular
    return np.fft.ifftshift(H)  # match np.fft.fft2's unshifted (DC-at-corner) layout


def _log_gabor_enhance(image, theta_field, freq_field, fg_mask_blocks, block=16, window=40,
                        sigma_onf=0.75, sigma_theta_deg=25.0, add_gain=2.5,
                        field_blur_sigma=1.0, coherence_field=None,
                        confidence_ceiling=0.45):
    """
    Step 4: block-wise 2D Log-Gabor filtering (Field, 1987), each block
    tuned to its own local orientation (theta_field, step 2) and ridge
    frequency (freq_field, step 4a) — following the general locally-tuned,
    block-wise filtering strategy in Shams et al. (2023) ("LR2"), built here
    from Log-Gabor transfer functions rather than Hong et al.'s (1998)
    spatial Gabor kernels (that spatial-domain approach is Pipeline B's
    technique; this frequency-domain one is Pipeline C's point of comparison
    against it for the same P2/P6 problems).

    Each block is filtered inside a larger `window`-sized neighbourhood, not
    just its own `block` pixels, so the FFT has enough context to represent
    the tuned frequency cleanly. Since `window` is more than twice `block`,
    neighbouring blocks' windows overlap; rather than hard-cutting each to
    its own centre region (which leaves a visible seam at every block
    boundary), every window is blended into the output with a 2D Hanning
    taper via overlap-add.

    IMPORTANT (status note, 30 August 2026, FOURTH revision — back to
    additive, this time for good): this function has gone through four
    designs, and the last two both failed the same way, which is why this
    revision stops trying to make replacement work at all.
      1st version REPLACED each window with its own bandpass-filtered
        response, rescaled to match that block's own local std exactly ->
        hallucinated "feathery" texture wherever the local estimate was
        near-zero (gains up to ~2.3 trillion x in the worst blocks).
      2nd version REPLACED each window with its bandpass response rescaled to
        one fixed global std instead of a per-block one -> visually more
        plausible, but NFIQ2 still dropped substantially on real test images
        (e.g. DB1_B/101_6.tif 86 -> 37).
      3rd version tried REPLACEMENT again, gated to foreground blocks only
        (fg_mask_blocks) on top of a newly-added normalize_image()+segment()
        preprocessing stage, to more closely match how Shams et al. (2023)
        ("LR2") describe this step. Tested on the dashboard: NFIQ2 dropped
        again (DB3_B/108_6.tif 37 -> 20) and the output was visibly blurrier
        than the raw image, not sharper — same failure mode as the 2nd
        version, just less severe.
      Both replacement attempts fail for the same underlying reason, not a
      tuning problem: a Log-Gabor filter keeps only a single narrow
      frequency band, i.e. one sine wave. A real ridge/valley pattern is
      much closer to a square wave — sharp, high-contrast transitions — with
      energy spread across its fundamental frequency AND several harmonics.
      Substituting the narrowband response for the real image, at any gain,
      necessarily replaces that sharp square-wave-like structure with a
      smoother sinusoidal one, which is exactly the "blurrier than raw"
      effect seen on the dashboard both times. No amount of rescaling fixes
      this — it needs either a multi-band/harmonic reconstruction (real
      added complexity and risk) or dropping replacement altogether.
      4th version (current, and the one to keep from here on): back to the
      ADDITIVE blend (enhanced = image + gain * filtered) that already
      proved itself on the dashboard earlier this project — keep the real
      (diffused) image as the base, add a modest gain-scaled copy of the
      bandpass component on top. `fg_mask_blocks` is still used, but now
      just to skip adding any boost to background blocks (add_gain applies
      to foreground blocks only) — background gets no boost either way, so
      this is a minor refinement, not the fix itself; the fix is being
      additive at all.

    theta_field and freq_field are lightly Gaussian-blurred before use
    (theta via its doubled-angle vector representation, since theta and
    theta+pi describe the same ridge direction and can't be averaged
    directly) — a single noisy per-block orientation/frequency estimate
    otherwise gets baked into a whole tuned filter and can produce a small,
    visually obvious ring/blob artifact wherever it disagrees sharply with
    its neighbours.

    FIFTH revision (30 August 2026), per-block confidence gating: found on
    DB4_B/108_2.tif (reported directly by a teammate, screenshot from the
    dashboard) — with the quality-adaptive alpha scheme (see the
    module-level comment above _COHERENCE_FULL_GENTLE) pushing add_gain
    toward its full aggressive value on low-coherence images, some blocks
    inside an already-degraded image came out as busy, wavy, non-ridge-like
    texture rather than sharper ridges. Root cause: `fg_mask_blocks` only
    gates on/off, foreground vs background — it says nothing about whether
    THIS block's own theta/freq estimate is actually trustworthy. A whole
    -image alpha pushes add_gain up for every foreground block in a
    low-average-coherence image, including the specific blocks whose local
    orientation is genuinely ambiguous (not just noisy-but-real, which is
    what DB3's low coherence usually means) — boosting a confidently wrong
    Log-Gabor filter into one of those blocks, at high gain, paints in
    plausible-looking but fake periodic texture instead of recovering real
    ridge structure. Fix: scale add_gain per block by that block's own
    (smoothed) coherence_field value, in addition to the existing fg/bg
    gate — a block with coherence at or above confidence_ceiling gets the
    full requested add_gain, a block with coherence near 0 gets close to
    none, tapering linearly in between. This is a strictly finer-grained
    version of the same idea fg_mask_blocks already applies (don't boost
    where there's no trustworthy signal to boost) — coherence_field is
    passed in optionally (defaults to None, meaning "trust every foreground
    block equally", the pre-fix behaviour) so existing callers/tests keep
    working.
    """
    img = image.astype(np.float64)
    h, w = img.shape
    nby, nbx = theta_field.shape

    # Smooth theta_field via its doubled-angle vector representation, and
    # freq_field directly — see the docstring note above.
    cos2 = np.cos(2 * theta_field)
    sin2 = np.sin(2 * theta_field)
    cos2_s = cv2.GaussianBlur(cos2.astype(np.float32), (0, 0), field_blur_sigma)
    sin2_s = cv2.GaussianBlur(sin2.astype(np.float32), (0, 0), field_blur_sigma)
    theta_field = 0.5 * np.arctan2(sin2_s, cos2_s)
    freq_field = cv2.GaussianBlur(freq_field.astype(np.float32), (0, 0), field_blur_sigma).astype(np.float64)

    # Per-block gain confidence (see docstring note above, FIFTH revision):
    # smooth coherence_field the same way as freq_field, then map it to a
    # 0-1 confidence that throttles add_gain block by block. confidence=1
    # (full add_gain) once coherence reaches confidence_ceiling; confidence
    # ramps down to 0 (no boost at all, same as a background block) as
    # coherence drops toward 0. coherence_field=None keeps every foreground
    # block at confidence=1, i.e. the old behaviour.
    if coherence_field is not None:
        coherence_smoothed = cv2.GaussianBlur(
            np.clip(coherence_field, 0.0, 1.0).astype(np.float32), (0, 0), field_blur_sigma
        ).astype(np.float64)
        gain_confidence = np.clip(coherence_smoothed / confidence_ceiling, 0.0, 1.0)
    else:
        gain_confidence = np.ones_like(theta_field)

    pad = window // 2
    padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    # 2D raised-cosine taper: 0 at a window's own edges, 1 at its centre, so
    # overlap-add blends neighbouring windows smoothly instead of seaming.
    taper_1d = np.hanning(window)
    taper_1d = np.clip(taper_1d, 1e-3, None)  # keep every contribution weighted, never exactly zero
    taper = np.outer(taper_1d, taper_1d)

    accum = np.zeros_like(padded)
    weight = np.zeros_like(padded)

    for by in range(nby):
        for bx in range(nbx):
            theta = theta_field[by, bx]
            freq = freq_field[by, bx]
            if not np.isfinite(freq) or freq <= 0:
                freq = 1.0 / 9.0

            cy = by * block + block // 2 + pad
            cx = bx * block + block // 2 + pad
            y0, y1 = cy - pad, cy - pad + window
            x0, x1 = cx - pad, cx - pad + window
            win = padded[y0:y1, x0:x1]
            if win.shape != (window, window):
                continue  # shouldn't happen given the reflect-padding, but be defensive

            is_fg = fg_mask_blocks is None or fg_mask_blocks[by, bx]
            if is_fg:
                H = _log_gabor_filter_2d((window, window), freq, theta,
                                          sigma_onf=sigma_onf, sigma_theta_deg=sigma_theta_deg)
                F = np.fft.fft2(win)
                filtered = np.real(np.fft.ifft2(F * H))
                # ADDITIVE blend: the real window stays the base; only a
                # gain-scaled copy of its bandpass component is added on top
                # — see the docstring note above for why this is kept over
                # a replacement approach. Gain is further scaled by this
                # block's own confidence (FIFTH revision) so a block with an
                # untrustworthy local orientation/frequency estimate doesn't
                # get a confidently-wrong filter boosted into it.
                block_gain = add_gain * gain_confidence[by, bx]
                enhanced_win = win + block_gain * filtered
            else:
                # Background block: no bandpass component to add — the
                # foreground mask exists so we don't waste a boost (or risk
                # amplifying noise) on blocks with no real ridge signal.
                enhanced_win = win

            accum[y0:y1, x0:x1] += enhanced_win * taper
            weight[y0:y1, x0:x1] += taper

    weight[weight == 0] = 1.0
    merged = (accum / weight)[pad:pad + h, pad:pad + w]
    return np.clip(merged, 0, 255).astype(np.uint8)


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 0a: block-wise intensity normalisation (Hong, Wan, & Jain, 1998)
    # — preprocessing, not one of the three primary techniques (see module
    # docstring). Puts every image on the same intensity scale before
    # anything else runs, matching Shams et al. (2023) ("LR2")'s own
    # Normalisation -> Segmentation -> ... pipeline order.
    normalized = normalize_image(
        image,
        target_mean=params.get("normalize_target_mean", DEFAULT_NORMALIZE_TARGET_MEAN),
        target_var=params.get("normalize_target_var", DEFAULT_NORMALIZE_TARGET_VAR),
    )

    # Step 0b: Otsu/block-variance segmentation (Hong, Wan, & Jain, 1998) —
    # preprocessing. Produces fg_mask_blocks, used internally by step 4 to
    # decide which blocks the Log-Gabor step is allowed to add its bandpass
    # boost to. This mask is NOT used to blank/recolour the final output
    # image (see module docstring) — the visible result stays a full-frame
    # image. segment()'s default hole-filling (see common.py) already
    # handles the whorl-core mis-segmentation issue found on DB3_B/101_1.tif.
    fg_mask_blocks, _block_var = segment(normalized)

    # Step 0c: quality-adaptive aggressiveness (see the module-level
    # comment above _COHERENCE_FULL_GENTLE for the full reasoning). Probes
    # the normalised-but-unfiltered image's own orientation coherence to
    # estimate how degraded THIS image actually is, before deciding how
    # hard steps 1/3/4 should work on it. This probe orientation_field()
    # call is separate from step 2's (below), which runs on the
    # contrast-corrected image to steer steps 3-4 as before — this one only
    # exists to set alpha.
    _theta_probe, coherence_probe = orientation_field(normalized)
    alpha = _aggressiveness_alpha(coherence_probe, fg_mask_blocks)

    # Step 1: homomorphic filtering (Oppenheim, Schafer, & Stockham, 1968)
    # — this pipeline's technique for P1 (low global contrast, systematic
    # to DB3). Runs on the normalised image (step 0a). gamma_high now
    # defaults to an alpha-scaled value (Step 0c) rather than a single
    # fixed setting, so an already-clean print doesn't get boosted as hard
    # as a genuinely low-contrast one.
    contrast_enhanced_raw = _homomorphic_filter(
        normalized,
        cutoff=params.get("homomorphic_cutoff", 0.06),
        gamma_low=params.get("homomorphic_gamma_low", 0.5),
        gamma_high=params.get("homomorphic_gamma_high", _lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha)),
        sharpness=params.get("homomorphic_sharpness", 1.0),
    )

    # Step 1b: feather the homomorphic result back toward the (unfiltered)
    # normalised image over background regions, using fg_mask_blocks (step
    # 0b). Found necessary on DB1_B/102_5.tif (foreground ratio ~0.30, one of
    # DB1's large-background images — see P5, Section 3): homomorphic
    # filtering attenuates low spatial frequencies (the illumination term)
    # and boosts high frequencies (the reflectance term); a bright, uniform
    # background is almost entirely low-frequency content, so the very step
    # that sharpens ridge contrast for P1 pushes a large flat background
    # toward near-black once rescaled — measured background corner mean
    # dropping from 254 (raw) to 27 (post-homomorphic) on that image, and
    # NFIQ2 dropped 49 -> 38 on the dashboard as a result. This isn't a
    # coding bug, it's what homomorphic filtering does to non-ridge content
    # by construction — DB3, the subset this technique specifically targets,
    # has a much higher foreground ratio (Section 3.1) so this is less
    # visible there, but DB1's large background area (P5) exposes it.
    # `fg_mask_blocks` is upsampled and Gaussian-blurred into a smooth 0-1
    # alpha (not a hard cutout, to avoid a visible seam at the fingerprint
    # boundary): background is blended back toward its normalised
    # brightness, foreground keeps the full homomorphic contrast boost.
    #
    # SIXTH issue, TRIED AND REJECTED (found by direct visual inspection of
    # fresh, non-flagged output on DB1, full-batch-tested by a teammate's
    # local run): a visible dark halo/vignette appears around the
    # fingerprint on some DB1 images (e.g. 101_1.tif, 104_5.tif) that isn't
    # present in the raw image — caused by `fg_mask_blocks` being only
    # block-resolution (16x16px), so it's inherently coarser/rounder than
    # the true (partly concave) fingerprint silhouette, making Step 1b
    # treat a ring of genuinely-background pixels as foreground and keep
    # the homomorphic filter's low-frequency attenuation there instead of
    # blending it back toward the normalised image.
    # A candidate fix — eroding a LOCAL COPY of the mask (3x3, 2 iterations)
    # before computing `fg_alpha` ONLY, leaving `fg_mask_blocks` itself
    # untouched for `_aggressiveness_alpha`/`_log_gabor_enhance` — was
    # implemented and spot-checked visually on 8 images (halo visibly
    # reduced on both flagged DB1 images, no regression seen on DB2/DB3/DB4
    # samples). But a full 320-image NFIQ2 batch run (before/after this
    # change) showed it made every subset WORSE, including the DB1 images
    # it targeted: DB1_B -8.25->-9.19 (18%/80% -> 12%/85% improved/
    # regressed), DB2_B +7.99->+6.86, DB3_B +13.78->+11.32, DB4_B
    # +12.21->+8.24. Reason: eroding `fg_alpha`'s mask shrinks the
    # full-contrast-boost region on EVERY image, not just the oversized-mask
    # DB1 cases — on images where the mask already tracked the print
    # reasonably well, it just trims real homomorphic contrast off a ring
    # of genuine ridge detail near the boundary, which cost more NFIQ2
    # across the board than the halo fix recovered anywhere, DB1 included.
    # REJECTED — reverted to the plain (unmodified) `fg_mask_blocks` below.
    # The DB1 halo remains a known, documented, visual-only cosmetic
    # artifact (block-segmentation resolution limit) — the already-accepted
    # numbers above (DB1_B -8.25 etc.) are with this artifact present, so
    # leaving it alone is a strictly better trade than any mask-erosion fix
    # tried so far. This reinforces the same conclusion as DB1's overall
    # net-negative result: DB1's specific failure modes resist targeted
    # local fixes without giving back gains elsewhere.
    h, w = normalized.shape
    fg_alpha = cv2.resize(fg_mask_blocks.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    fg_alpha = cv2.GaussianBlur(fg_alpha, (0, 0), 8.0)
    fg_alpha = np.clip(fg_alpha, 0.0, 1.0)
    contrast_enhanced = (
        fg_alpha * contrast_enhanced_raw.astype(np.float64)
        + (1 - fg_alpha) * normalized.astype(np.float64)
    )
    contrast_enhanced = np.clip(contrast_enhanced, 0, 255).astype(np.uint8)

    # Step 2: ridge orientation field estimation (Hong et al., 1998) — a
    # supporting calculation, not one of this pipeline's three primary
    # techniques (see module docstring). Computed on the contrast-corrected
    # image so steps 3-4 are steered by a cleaner orientation estimate than
    # DB3's original flat contrast would give.
    theta_field, coherence_field = orientation_field(contrast_enhanced)

    # Step 3: coherence-enhancing anisotropic diffusion (Perona & Malik,
    # 1990), steered by theta_field — this pipeline's technique for P2
    # (high random noise, systematic to DB3). Smooths MORE along the ridge
    # direction and LESS across it, preserving the ridge/valley boundary
    # itself.
    #
    # History: iterations/kappa were first raised 15/15.0 -> 25/25.0 because
    # homomorphic filtering (step 1) boosts high spatial frequencies to
    # sharpen ridge contrast, but that boost doesn't distinguish "high
    # frequency = ridge detail" from "high frequency = noise" — on DB3
    # specifically (P2, noise ~3-4x every other subset), this amplified
    # noise right alongside real detail, producing a speckled/blotchy
    # texture even after step 1's own contrast fix (verified on
    # DB3_B/109_7.tif and 105_3.tif by rendering each stage separately).
    # 25/25.0 fixed that. It was then pulled back DOWN to 15/15.0 as the
    # aggressive CEILING of the alpha-scaled range below, after a teammate
    # screenshot (DB4_B/108_2.tif) traced a different failure to this step:
    # diffusion needs theta_field to be a trustworthy direction to smooth
    # ALONG, and on a block where the orientation estimate itself is
    # unreliable, more iterations at high kappa doesn't clean up noise, it
    # organises noise into a smooth, confident-looking but FAKE periodic
    # pattern instead. That pullback was then RESTORED back to 25/25.0 (see
    # the module-level comment above _DIFFUSION_ITERATIONS_RANGE) once a
    # real 320-image batch run showed it wasn't paying for itself: DB1_B's
    # regression was unmoved by it, while DB3_B/DB4_B's gains dropped
    # measurably. The wavy-artifact fix that's kept is the per-pixel
    # CONFIDENCE DAMPING inside _coherence_diffusion itself (SIXTH
    # revision), which targets only the specific unreliable pixels rather
    # than softening the ceiling for every low-coherence image.
    # iterations/kappa now default to alpha-scaled values (Step 0c) instead
    # of one fixed setting — see the module-level comment above
    # _COHERENCE_FULL_GENTLE.
    diffusion_iterations = params.get("diffusion_iterations", int(round(_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha))))
    diffusion_dt = params.get("diffusion_dt", 0.2)
    diffusion_kappa = params.get("diffusion_kappa", _lerp(_DIFFUSION_KAPPA_RANGE, alpha))
    diffused = _coherence_diffusion(
        contrast_enhanced,
        theta_field,
        coherence_field,
        iterations=diffusion_iterations,
        dt=diffusion_dt,
        kappa=diffusion_kappa,
        # Per-pixel confidence damping (SIXTH revision) — see
        # _coherence_diffusion's docstring. Same default ceiling as the
        # Log-Gabor confidence fix, for consistency.
        confidence_ceiling=params.get("diffusion_confidence_ceiling", 0.45),
    )

    # Step 4a: local ridge-frequency estimation (supporting calculation for
    # step 4b; Hong, Wan, & Jain, 1998 x-signature method)
    freq_field = _estimate_ridge_frequency(
        diffused,
        theta_field,
        window_len=params.get("freq_window_len", 32),
        window_width=params.get("freq_window_width", 16),
        min_period=params.get("freq_min_period", 3),
        max_period=params.get("freq_max_period", 25),
    )

    # Step 4b: 2D Log-Gabor filtering (Field, 1987; Shams et al., 2023),
    # tuned per block by theta_field (step 2) and freq_field (step 4a) —
    # this pipeline's technique for P6 (weak/inconsistent ridge orientation,
    # DB4/DB3). Applied as an ADDITIVE blend on foreground blocks only
    # (fg_mask_blocks, from step 0b) — see _log_gabor_enhance's docstring:
    # a direct LR2-style replacement was tried twice (with and without
    # normalisation/segmentation preprocessing) and both times hurt NFIQ2
    # and looked blurrier than the raw image, because a single Log-Gabor
    # band can't represent a real ridge's full (harmonic-rich) energy.
    enhanced = _log_gabor_enhance(
        diffused,
        theta_field,
        freq_field,
        fg_mask_blocks,
        window=params.get("log_gabor_window", 40),
        sigma_onf=params.get("log_gabor_sigma_onf", 0.75),
        sigma_theta_deg=params.get("log_gabor_sigma_theta_deg", 25.0),
        # add_gain now defaults to an alpha-scaled value (Step 0c) instead
        # of the single DB3-tuned setting — see the module-level comment
        # above _COHERENCE_FULL_GENTLE.
        add_gain=params.get("log_gabor_add_gain", _lerp(_LOG_GABOR_ADD_GAIN_RANGE, alpha)),
        # coherence_field lets _log_gabor_enhance throttle add_gain per
        # block, not just per whole image — see its docstring's FIFTH
        # revision note (found on DB4_B/108_2.tif).
        coherence_field=coherence_field,
        confidence_ceiling=params.get("log_gabor_confidence_ceiling", 0.45),
    )

    return enhanced


if __name__ == "__main__":
    test_path = os.path.join(RAW_DIR, "DB3_B", "101_1.tif")
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not load {test_path} — did you copy your dataset into data/raw/? See README.")
    else:
        out = enhance(img)
        cv2.imwrite("pipeline_c_test_output.png", out)
        print("Saved pipeline_c_test_output.png — open it and compare against the input image.")
