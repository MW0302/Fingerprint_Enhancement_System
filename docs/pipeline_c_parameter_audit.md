# Pipeline C parameter audit

Audited 4 September 2026. Read-only audit — no code changed as part of producing
this table. Scope: every tunable parameter/constant/threshold in
`src/pipeline_c/pipeline_c.py` and the shared `src/utils/common.py` helpers it
calls (`normalize_image()`, `segment()`, `orientation_field()`).

Goal: not global optimality — finding any second "std=40-style" hazard, where a
parameter is treated as "already tuned, fine" but was never actually validated
against downstream NFIQ2.

**"Verified" bar** (same standard as the Pipeline A CLAHE `normalize_target_std`
sweep): NFIQ2 as the metric, multiple candidate values compared, evaluated
across the full 320-image / 4-DB dataset (or a clearly representative subset).
Anything short of that is **Suspicious** (some reasoning or a single-image/
single-candidate check exists, but doesn't meet the bar) or **Undocumented**
(no discussion at all — a bare literal with no rationale on record).

Sources checked: `pipeline_c.py`'s own docstrings/comments (the authoritative
record — see below), `common.py`, `git log --follow` on `pipeline_c.py` (the
intermediate tuning history is squashed into one large commit, `fa843c2
"update many things ToT"`, with no further detail in the commit message
itself — the in-file docstrings are the only surviving record of that work),
`Team_Member_Starter_Packets.docx`, and the two rejected experiment scripts
under `scripts/experiments/` (they test different mechanisms — confidence
gating, Log-Gabor replace mode — not sweeps of these base numeric constants,
so they add no additional evidence here).

## P1 — Homomorphic Filtering

| Parameter | Current value | How it was originally validated | Classification |
|---|---|---|---|
| `gamma_high` gentle endpoint | 1.0 | NFIQ2, full 320 images, all 4 DBs, two rounds (std=40 vs std=10 Step 0, then 1.2 vs 1.0 gentle endpoint) | **Verified** |
| `gamma_high` aggressive endpoint | 2.0 | Docstring only says "this session's DB3-tuned default" — no candidate values, sample size, or metric on record | **Suspicious** |
| `cutoff` (frequency cutoff) | 0.06 | Bare function-signature default, passed through by `enhance()` unchanged; never discussed in any docstring | **Undocumented** |
| `gamma_low` (illumination suppression) | 0.5 | Same — bare default, zero discussion | **Undocumented** |
| `sharpness` (transition steepness) | 1.0 | Same — bare default, zero discussion | **Undocumented** |
| Output rescale via 1st/99th percentile (not min/max) | Hardcoded inside `_homomorphic_filter`, not a `params` option | Mechanism has a stated physical rationale (avoid outlier-driven crush), but the 1st/99th choice itself was never compared against other margins (e.g. 2nd/98th, 5th/95th) | **Suspicious** |
| Step 1b background-feathering Gaussian blur sigma | **8.0** | Hardcoded directly in `enhance()`, not even routed through `params.get()`. The *mechanism* (feathering itself) is NFIQ2-validated (DB1_B/102_5.tif 49→38 triggered the fix), but the specific value 8.0 was never discussed or compared against alternatives | **Undocumented** (and not overridable via `params`) |

## P2 — Coherence Diffusion

| Parameter | Current value | How it was originally validated | Classification |
|---|---|---|---|
| `iterations` aggressive endpoint | 25 | NFIQ2, full 320 images, all 4 DBs, explicit 15 vs 25 A/B comparison with numbers on record (DB1 -13.9→-14.5, DB3 +15.8→+9.7, etc.) | **Verified** (only 2 candidates compared, not a wide sweep) |
| `kappa` aggressive endpoint | 25.0 | Same 15/15.0 vs 25/25.0 test, tested jointly with iterations | **Verified** (same 2-candidate caveat) |
| `iterations` gentle endpoint | 8 | Described only as "a light-touch value chosen so the technique still visibly runs" — no comparison against other gentle candidates (e.g. 5, 10) | **Suspicious** |
| `kappa` gentle endpoint | 10.0 | Same | **Suspicious** |
| `dt` (diffusion time step) | 0.2 | Bare default, never discussed | **Undocumented** |
| Along-ridge kappa tolerance multiplier (`kappa * 4.0`) | Hardcoded `4.0` inside `_coherence_diffusion` | No discussion of why 4x specifically, not 3x or 5x | **Undocumented** |
| `confidence_ceiling` (per-pixel diffusion confidence) | 0.45 | Docstring explicitly states this reuses Log-Gabor's own value "for consistency" — not independently derived for diffusion | **Suspicious** (explicitly borrowed, not independently tested) |

## P6 — 2D Log-Gabor

| Parameter | Current value | How it was originally validated | Classification |
|---|---|---|---|
| `add_gain` gentle endpoint | 0.8 | Only the additive-vs-replace *mechanism* was NFIQ2-tested (replace tried twice, failed both times); the magnitude 0.8 itself was never swept | **Suspicious** |
| `add_gain` aggressive endpoint | 2.5 | Same — no candidate-value comparison on record | **Suspicious** |
| `sigma_onf` (log-frequency bandwidth) | 0.75 in actual use (note: `_log_gabor_filter_2d`'s own signature default is 0.65, but every caller always passes 0.75 explicitly, so 0.65 is dead/unused — worth cleaning up if this file is touched again) | Bare default, zero discussion | **Undocumented** |
| `sigma_theta_deg` (angular bandwidth) | 25.0 in actual use (inner function's own default is 20.0, similarly unused) | Bare default, zero discussion | **Undocumented** |
| `window` (per-block filter window size) | 40 | Only a functional rationale ("bigger than block, so the FFT has context"), never compared against other window sizes (e.g. 32, 48) | **Suspicious** |
| `field_blur_sigma` (theta/freq/coherence field smoothing) | 1.0 | **Not exposed via `enhance()`'s `params` at all** — can only be changed by editing the function default itself. Zero discussion | **Undocumented** (and not overridable via `params`) |
| `block` (per-block filter size) | 16 (inherited from `common.BLOCK`) | Never independently tested; only a structural consistency argument with the rest of the codebase | **Undocumented** (structural rationale exists, no empirical one) |
| Filter-bank orientation/frequency channel count | **N/A** — this is a single adaptively-tuned Log-Gabor filter per block (one theta/freq pair), not a multi-channel filter bank | — | Not applicable — flagged here only to close out the audit request explicitly, not a gap |
| `_estimate_ridge_frequency`: `window_len` | 32 | Zero discussion | **Undocumented** |
| `_estimate_ridge_frequency`: `window_width` | 16 | Zero discussion | **Undocumented** |
| `_estimate_ridge_frequency`: `min_period` | 3 | Zero discussion; plausible on physical grounds but untested | **Undocumented** |
| `_estimate_ridge_frequency`: `max_period` | 25 | Same | **Undocumented** |
| `_estimate_ridge_frequency`: `default_freq` (fallback when x-signature fails) | 1/9 | **Not exposed via `enhance()`'s `params`**, zero discussion | **Undocumented** (and not overridable via `params`) |
| `confidence_ceiling` (Log-Gabor per-block confidence) | 0.45 | This is where 0.45 was *originally* introduced (FIFTH revision) — but even there, no candidate-value comparison, only a qualitative before/after description (found DB4_B/108_2.tif's artefact, added a linear ramp) | **Suspicious** |

## Coherence Gating (the alpha-adaptive scheme itself)

| Parameter | Current value | How it was originally validated | Classification |
|---|---|---|---|
| `_COHERENCE_FULL_AGGRESSIVE` (alpha=1 floor) | 0.50 | **Re-validated 4 September 2026**: NFIQ2, full 320 images, all 4 DBs, 5-candidate sweep {0.45, 0.50, 0.55, 0.60, 0.65} (gentle fixed at 0.68) — 0.50 confirmed best on overall Δtotal (9.445; monotonic decline above 0.50 down to 7.824 at 0.65) | **Verified** (Tier 1 sweep, `results/coherence_sweep_aggressive.csv`) |
| `_COHERENCE_FULL_GENTLE` (alpha=0 ceiling) | 0.68 | **Re-validated 4 September 2026**: NFIQ2, full 320 images, all 4 DBs, 5-candidate sweep {0.55, 0.60, 0.65, 0.68, 0.75} (aggressive fixed at the confirmed 0.50) — 0.68 confirmed best on overall Δtotal (9.445; DB3 keeps climbing past 0.68 but DB1/DB2 lose more than DB3 gains) | **Verified** (Tier 1 sweep, `results/coherence_sweep_gentle.csv`) |
| Alpha interpolation shape (linear vs. any other curve) | Linear: `(GENTLE - coherence) / span`, clipped to [0,1] | Structural choice, never compared against a non-linear mapping (e.g. sigmoid) | **Undocumented** (structural, but it shapes every value all three adaptive parameters actually take) |

## Other / Shared (easy to overlook)

| Parameter | Current value | How it was originally validated | Classification |
|---|---|---|---|
| `segment()`'s `block` | 16 (`common.BLOCK`, shared by `segment()`/`orientation_field()`/Log-Gabor) | Never independently swept; only a cross-module consistency argument | **Undocumented** |
| `segment()`'s `fill_holes` / convex-hull logic | Default `True`, uses `_largest_component_hull` | Validated by **visual inspection on 8 images** (5 regressed + 3 baseline) — not NFIQ2, far short of full-dataset sample size | **Suspicious** (explicitly falls short of the verified bar despite having a real validation record) |
| `normalize_image()`'s `target_var`/`target_std` | 10.0 | NFIQ2, full 320 images, all 4 DBs, 7 candidate values (10/20/30/40/50/60/70) | **Verified** — the only parameter in the whole project that has actually received CLAHE-sweep-level coverage |
| `normalize_image()`'s headroom-cap margin `[2, 253]` | Hardcoded | Zero discussion — why 2/253 rather than 1/254 or 5/250 is never addressed | **Undocumented** |

## Summary

At the time of the original audit (4 September 2026, before any Tier 1
sweeps), only **4 items** in the entire pipeline met the "verified" bar:
`normalize_image()`'s `target_std` (the only one with genuine wide-sweep
coverage at the time), diffusion's `iterations`/`kappa` aggressive endpoints
(2-candidate comparison), and coherence gating's `_COHERENCE_FULL_AGGRESSIVE`
(2-candidate comparison). Every gentle endpoint except the newly-fixed
`gamma_high`, and most of P1/P6's specific numeric values, were set once and
never compared against an alternative — the "std=40" situation
(reasonable-looking, never actually validated downstream) was not an isolated
case in this codebase, it was closer to the norm.

**Update (4 September 2026, same day, Tier 1 coherence-gating sweep):** both
`_COHERENCE_FULL_AGGRESSIVE` and `_COHERENCE_FULL_GENTLE` have now been
properly wide-swept (5 candidates each, full 320-image/4-DB, evaluated on
final Δtotal since these two thresholds jointly shape all three adaptive
parameters) — see the updated rows above. Unlike the `normalize_image`/CLAHE
case, this sweep **confirmed the existing values were already optimal** among
the candidates tested; no change was made. Worth noting as a counterexample
to "undocumented/suspicious == wrong" — not every parameter that lacked a
rigorous validation record turns out to have been mistuned, but the only way
to know was to actually check, which is the whole point of this audit.
Remaining Tier 1/2 items (Log-Gabor `add_gain`, homomorphic `gamma_high`
aggressive endpoint, `confidence_ceiling`, etc.) are still open — see the
project's own prioritisation for what's next.

Three values are both **undocumented and not even overridable via `params`**
(harder to spot than an ordinary undocumented default, since even trying an
alternative requires editing source rather than passing a parameter):
1. Step 1b's background-feathering blur sigma = 8.0
2. Log-Gabor's `field_blur_sigma` = 1.0
3. Log-Gabor's `default_freq` = 1/9
