# Alternative Hybrid combinations: does anything beat Pipeline C alone?

Generated from `scripts/hybrid_v2_ablation.py` against `src/hybrid/hybrid_v2a.py`
and `src/hybrid/hybrid_v2b.py`, all 320 raw images, real NFIQ2 runs. Built to
test two specific hypotheses raised by `docs/hybrid_validation_findings.md`:
that the current Hybrid's DB3_B scoreability collapse and DB1_B net
regression are artifacts of *which* P1/P2 pairing was used, not inherent to
combining techniques at all.

**Headline finding, read this first:** both alternative combinations fix
the DB3_B scoreability collapse completely (79/80 valid, same ceiling as
every other pipeline's own raw-limited count) — confirming the
feathering-blend hypothesis from Section 2 of the prior findings. But
**neither clearly beats Pipeline C alone**, and they fail in opposite,
mutually exclusive ways:

- **Variant 2a** (Pipeline C's own P1+diffusion, re-paired) nudges the
  overall mean *above* Pipeline C alone (51.95 vs 51.53, on the same real
  319-image basis) — but makes DB1_B **worse** than Pipeline C alone's
  already-negative DB1_B result (−7.20 vs C's own −5.26), the single worst
  DB1_B result of any combination tested so far, hybrid or otherwise.
- **Variant 2b** (Pipeline B's P1 paired with Pipeline B's *own* P2,
  avoiding Pipeline C's diffusion entirely) is the only combination tested
  anywhere in this project that makes DB1_B **net positive** (+1.49) — but
  its overall mean (50.69) falls *below* Pipeline C alone, because DB2_B,
  DB3_B, and DB4_B all underperform Pipeline C's own numbers there.

No parameter was retuned and no result was cherry-picked to reach this
conclusion — both variants are reported in full below, exactly as scored.

## 0. What each variant is

Neither modifies `pipeline_a.py`/`pipeline_b.py`/`pipeline_c.py`/`hybrid.py`
— both only import and call their real internal functions and stage
functions, structured the same way `hybrid.py` already is (separately
callable `stage0_preprocess`/stage1/stage2/`stage3_orientation`/`enhance`).

### Variant 2a (`src/hybrid/hybrid_v2a.py`) — restores the mechanical Section 2.4 P1 winner

| Slot | Technique | Function | Real locked params (confirmed against source `enhance()`) |
|---|---|---|---|
| P1 (contrast) | Pipeline C's own homomorphic filter + Step 1b feathering blend | `stage1_contrast_c()` (new) calling `_homomorphic_filter` (`pipeline_c.py`) | `cutoff=0.06, gamma_low=0.5, gamma_high=_lerp(_HOMOMORPHIC_GAMMA_HIGH_RANGE, alpha), sharpness=1.0`, then feather blend: `fg_alpha` = `fg_mask_blocks` resized + Gaussian-blurred (σ=8.0), clipped to [0,1]; `contrast_enhanced = fg_alpha*raw + (1-fg_alpha)*normalized` |
| P2 (noise) | Pipeline C's coherence-enhancing diffusion, **unchanged** | `hybrid.stage2_noise()` (reused directly) | `iterations=_lerp(_DIFFUSION_ITERATIONS_RANGE, alpha)` (rounded), `dt=0.2`, `kappa=_lerp(_DIFFUSION_KAPPA_RANGE, alpha)`, `confidence_ceiling=0.45` |
| P6 (orientation) | Pipeline A's oriented Gabor filtering, **unchanged** | `hybrid.stage3_orientation()` (reused directly) | `kernel_size=17, sigma=4.0, wavelength=8.0, gamma=0.5, strength=0.7, orientation_bins=16, coherence_floor=0.2` |

`alpha` (the quality-adaptive aggressiveness scalar) is computed once, from
`orientation_field(normalized)`'s coherence field and `fg_mask_blocks`, via
`_aggressiveness_alpha()` — identical probe to `hybrid.py`'s own Stage 2 and
to `pipeline_c.py`'s own Step 0c, deliberately duplicated (it's a
deterministic function of pre-P1 inputs only, so duplicating the call costs
nothing and keeps each stage function independently correct).

### Variant 2b (`src/hybrid/hybrid_v2b.py`) — keeps P1/P2 paired within Pipeline B

| Slot | Technique | Function | Real locked params (confirmed against source `enhance()`) |
|---|---|---|---|
| P1 (contrast) | Pipeline B's wavelet contrast enhancement, **unchanged** | `hybrid.stage1_contrast()` (reused directly) | `wavelet="db4", level=3, coarse_gain=1.60, fine_gain=1.00, coefficient_floor_percentile=25.0, blend=1.0` |
| P2 (noise) | Pipeline B's **own** wavelet shrinkage denoising (not Pipeline C's diffusion) | `stage2_noise_b()` (new) calling `_wavelet_shrinkage_denoise` (`pipeline_b.py`) | `wavelet="db4", level=3, threshold_scale=1.00, denoise_finest_levels=1, blend=1.0, noise_adaptive=True, noise_reference_sigma=5.0, noise_adaptive_power=4.0, minimum_scale_factor=0.10` — identical to the function's own signature defaults, i.e. `pipeline_b.py` never overrides them either |
| P6 (orientation) | Pipeline A's oriented Gabor filtering, **unchanged** | `hybrid.stage3_orientation()` (reused directly) | Same as variant 2a above |

Unlike Pipeline C's diffusion, Pipeline B's own denoise step has no
quality-adaptive alpha scheme — `stage2_noise_b()` takes no `normalized`
argument, only `stage1_output` and `fg_mask_blocks` (confirmed by
`tests/test_hybrid_v2b.py`'s `test_no_alpha_probe_needed_unlike_pipeline_c_diffusion`).

## 1. Full cumulative ablation: variant 2a

| DB | n (valid) | raw mean | final mean | Δ vs raw | improved | regressed | unchanged |
|---|---|---|---|---|---|---|---|
| DB1_B | 80 | 62.04 | 54.84 | **−7.20** | 23 | 55 | 2 |
| DB2_B | 80 | 50.45 | 59.82 | +9.38 | 55 | 25 | 0 |
| DB3_B | **79** | 24.65 | 47.65 | +23.00 | 77 | 2 | 0 |
| DB4_B | 80 | 28.76 | 45.45 | +16.69 | 72 | 7 | 1 |
| **OVERALL** | **319** | 41.53 | 51.95 | **+10.43** | 227 | 89 | 3 |

`n=319` here is the **true, un-flattered basis** — only the one image every
pipeline in this project fails to score raw (`DB3_B/110_5.tif`) is excluded,
unlike the current Hybrid's own headline number (n=300, 20 additional DB3_B
images silently excluded from both sides of that comparison — see
`docs/hybrid_validation_findings.md` §1's caveat). Variant 2a's DB3_B valid-n
(79/80) matches Pipeline C's own raw-limited ceiling exactly.

## 2. Full cumulative ablation: variant 2b

| DB | n (valid) | raw mean | final mean | Δ vs raw | improved | regressed | unchanged |
|---|---|---|---|---|---|---|---|
| DB1_B | 80 | 62.04 | 63.52 | **+1.49** | 44 | 33 | 3 |
| DB2_B | 80 | 50.45 | 59.51 | +9.06 | 61 | 18 | 1 |
| DB3_B | **79** | 24.65 | 40.34 | +15.70 | 76 | 3 | 0 |
| DB4_B | 80 | 28.76 | 39.24 | +10.48 | 72 | 3 | 5 |
| **OVERALL** | **319** | 41.53 | 50.69 | **+9.16** | 253 | 57 | 9 |

Same true 319-image basis. DB3_B valid-n is again 79/80 — no diffusion
anywhere in this variant, so the scoreability collapse never had a
mechanism to occur in the first place (consistent with
`docs/hybrid_validation_findings.md` §2's finding that `minus_p2`, the
no-diffusion current-Hybrid variant, was already collapse-free).

## 3. The DB3_B scoreability collapse: resolved in both variants

| Combination | DB3_B valid n / 80 | P1 technique | P2 technique |
|---|---|---|---|
| Current Hybrid (Pipeline B's P1 + Pipeline C's diffusion) | **60/80** | Pipeline B wavelet contrast | Pipeline C diffusion |
| Pipeline C alone | 79/80 | Pipeline C homomorphic+feather | Pipeline C diffusion |
| **Variant 2a** (Pipeline C's P1 + Pipeline C's diffusion) | **79/80** | Pipeline C homomorphic+feather | Pipeline C diffusion |
| **Variant 2b** (Pipeline B's P1 + Pipeline B's denoise) | **79/80** | Pipeline B wavelet contrast | Pipeline B wavelet shrinkage denoise |

This confirms the mechanism proposed in
`docs/hybrid_validation_findings.md` §2 directly, not just by elimination:
Pipeline C's diffusion is only safe on DB3_B's marginal-foreground images
when it runs on an image Pipeline C's own Step 1b has already feathered
back toward the pre-filter brightness over background/low-confidence
regions — swap in Pipeline B's wavelet contrast (no such feathering step)
ahead of the *same* diffusion call, and 20/80 images collapse (current
Hybrid); put Pipeline C's own feathered P1 back in front of it, and the
collapse disappears completely (variant 2a). Separately, avoiding
Pipeline C's diffusion altogether (variant 2b) sidesteps the issue by
construction, regardless of what precedes it.

## 4. DB1_B: three different outcomes, none matching the current Hybrid's own number

| Combination | DB1_B final mean | Δ vs raw |
|---|---|---|
| Pipeline C alone | 56.78 | −5.26 |
| Current Hybrid (B's P1 + C's diffusion + A's Gabor) | 58.76 | −3.28 |
| **Variant 2a** (C's P1 + C's diffusion + A's Gabor) | **54.84** | **−7.20** |
| **Variant 2b** (B's P1 + B's denoise + A's Gabor) | **63.52** | **+1.49** |

Variant 2a's DB1_B is the worst of any combination tested in this project
so far — worse than Pipeline C alone, worse than the current Hybrid, worse
than any single source pipeline. Variant 2b's DB1_B is the *only* net-positive
DB1_B result seen anywhere in this project. The two variants sit on opposite
sides of every other combination tested — pairing P1/P2 within the same
source pipeline evidently does not have a uniform effect on DB1_B; it
depends entirely on which pipeline's pairing is used.

## 5. Overall comparison — same 319-image real basis for every row

| Pipeline / combination | Overall mean | Δ vs raw | DB1_B Δ |
|---|---|---|---|
| Pipeline A | 48.46 | +6.93 | — |
| Pipeline B | 48.10 | +6.57 | — |
| Pipeline D | 45.35 | +3.82 | — |
| **Pipeline C** | **51.53*** | **+10.00*** | −5.26 |
| Current Hybrid (n=300, flattered — see §1 caveat) | 51.42 | +9.89 | −3.28 |
| **Variant 2a** | **51.95** | **+10.43** | −7.20 |
| **Variant 2b** | **50.69** | **+9.16** | +1.49 |

*Pipeline C's own figure is quoted from `docs/hybrid_validation_findings.md`
(sourced via `dashboard/results_loader.py`'s own Δ-vs-raw computation, the
41.53-baseline methodology every pipeline is measured against). A direct
recomputation from `results/pipeline_c_ablation.csv` in this session gives
51.61/+10.08 — the ~0.1 point difference is rounding/methodology noise
between the two computations, not a discrepancy worth chasing further; both
numbers are consistent with variant 2a's 51.95 being a small, real edge
over Pipeline C alone, and variant 2b's 50.69 being a small, real deficit.

**Read plainly:** variant 2a's overall margin over Pipeline C alone (+0.34
to +0.42 depending on which Pipeline C figure is used) is real but small,
and it is not free — it comes with DB1_B degrading by a further ~2 points
versus Pipeline C alone's own already-negative DB1_B. Variant 2b's DB1_B
fix is real and substantial, but it is not free either — DB2_B/DB3_B/DB4_B
all fall behind Pipeline C alone's own numbers there (see §6), costing
almost a full point overall. Neither trade is a clean win.

## 6. Per-component verdict: variant 2a

Comparing variant 2a's full final NFIQ2 against each single-component-removed
variant. DB3_B's P1 comparison specifically is excluded (valid n mismatches:
full/minus_p2/minus_p6 all score 79/80, but minus_p1 — diffusion applied
directly to the un-enhanced Step 0 image, no P1 at all — only scores 41/80,
the same worst-case pattern already documented for the current Hybrid in
`docs/hybrid_validation_findings.md` §2). DB3_B's P2 and P6 comparisons
**are** included below since those variants' valid-n all match the full
run's (79/80 on both sides).

| DB | full | minus_p1 (no P1) | **P1 helps?** | minus_p2 (no P2) | **P2 helps?** | minus_p6 (no P6) | **P6 helps?** |
|---|---|---|---|---|---|---|---|
| DB1_B | 54.84 | 58.01 | **No (−3.17)** | 58.20 | **No (−3.36)** | 55.08 | **No (−0.24)** |
| DB2_B | 59.82 | 55.15 | Yes (+4.67) | 61.59 | **No (−1.76)** | 60.95 | **No (−1.13)** |
| DB3_B | 47.65 | n/a (n-mismatch, see above) | — | 37.47 | Yes (+10.18) | 45.16 | Yes (+2.48) |
| DB4_B | 45.45 | 40.22 | Yes (+5.23) | 42.15 | Yes (+3.30) | 45.74 | **No (−0.29)** |
| **OVERALL*** | 53.37 | 51.69 | Yes (+1.68) | 53.98 | **No (−0.61)** | 53.92 | **No (−0.55)** |

*OVERALL row computed across DB1_B/DB2_B/DB4_B only (240 images, weighted
by each variant's actual valid-n — minus_p1's is 227, not 240, for the same
DB2_B/DB4_B partial-collapse reason documented in the current Hybrid's own
findings), for the same DB3_B-exclusion reason as the header.

**Read plainly:** every component hurts DB1_B individually in this specific
combination — there is no single component to blame or remove to fix it,
which matches DB1_B's persistent, hard-to-fix-locally pattern already
documented across this project. P2 (diffusion) and P6 (Gabor) both help
substantially on DB3_B (where P2 is doing exactly the coherence-restoring
job it was designed for, safely, now that it is paired with its own P1)
but are each mildly net-negative once DB3_B is excluded from the overall
row — nearly the same "helps standalone, doesn't help in the specific
3-DB overall metric" pattern already seen in the current Hybrid's own
validation.

## 7. Per-component verdict: variant 2b

All four variants (full, minus_p1, minus_p2, minus_p6) score DB3_B at
79/80 — fully clean, no exclusions needed anywhere in this table.

| DB | full | minus_p1 (no P1) | **P1 helps?** | minus_p2 (no P2) | **P2 helps?** | minus_p6 (no P6) | **P6 helps?** |
|---|---|---|---|---|---|---|---|
| DB1_B | 63.52 | 62.90 | Yes (+0.62) | 62.80 | Yes (+0.72) | 64.39 | **No (−0.86)** |
| DB2_B | 59.51 | 55.91 | Yes (+3.60) | 59.51 | Neutral (0.00) | 56.88 | Yes (+2.64) |
| DB3_B | 40.34 | 35.06 | Yes (+5.28) | 39.29 | Yes (+1.05) | 33.49 | Yes (+6.85) |
| DB4_B | 39.24 | 34.30 | Yes (+4.94) | 39.11 | Yes (+0.13) | 33.69 | Yes (+5.55) |
| **OVERALL** | 54.09 | 51.04 | Yes (+3.05) | 53.81 | Yes (+0.28) | 51.65 | Yes (+2.44) |

*OVERALL row computed across DB1_B/DB2_B/DB4_B only (240 images each, all
variants), for direct comparability with §6's table above; all four DB3_B
rows are still shown individually since they're clean here.

**Read plainly:** this is the most internally coherent combination tested
in this project so far — every component helps in every DB it's measured
in, except P6 on DB1_B (a small −0.86, easily the smallest single negative
number in either variant's whole component table). P1 and P6 each do real
work everywhere; P2 (Pipeline B's own denoise) is consistently small but
never negative — a much gentler, much less DB-dependent contribution than
Pipeline C's diffusion makes anywhere else in this project. The ceiling is
simply lower: none of these three techniques individually reaches Pipeline
C's own per-DB numbers on DB2_B/DB3_B/DB4_B (§5), so a combination built
entirely from them, however well-behaved, tops out below Pipeline C alone.

## 8. Verdict

Per the task, no parameter was retuned, no component was swapped after the
fact, and neither result was chosen over the other to look better than what
the real NFIQ2 runs produced.

- **Both alternative combinations fix the DB3_B scoreability collapse**
  that made the current Hybrid genuinely unsafe to ship as-is — this part
  of the original concern is resolved, by either variant.
- **Neither alternative combination is an unambiguous improvement over
  Pipeline C alone.** Variant 2a is marginally ahead overall (+0.3 to +0.4
  points) but at the cost of the worst DB1_B result of any combination
  tested in this project. Variant 2b fixes DB1_B outright (the only
  combination anywhere in this project to do so) but falls behind Pipeline
  C alone overall by about a point, because it never matches Pipeline C's
  own DB2_B/DB3_B/DB4_B numbers.
- **The honest conclusion is that no tested Hybrid combination — the
  original assembly or either alternative here — clearly beats Pipeline C
  alone.** Each hybrid variant trades Pipeline C's own weaknesses for a
  different set of weaknesses, rather than eliminating them. If DB1_B's
  net regression is the deciding concern, variant 2b is the only
  combination in this project that removes it, at a small overall-quality
  cost relative to Pipeline C alone. If overall mean NFIQ2 is the deciding
  metric and DB1_B's regression is acceptable, Pipeline C alone remains
  the simplest option that is at least as good as anything tested,
  including both hybrids built specifically to try to beat it.
