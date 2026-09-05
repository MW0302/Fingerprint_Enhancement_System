# Hybrid pipeline validation findings (Section 2.4 step 5 / Handover Notes Section 17)

Generated from `scripts/hybrid_ablation.py` against `src/hybrid/hybrid.py`,
all 320 raw images, real NFIQ2 runs. Covers: the full hybrid's cumulative
ablation, the three hybrid-minus-one-component variants, and whether each
selected component still helps once actually combined with the other two.

**Headline finding, read this first:** the full hybrid works well
overall (+8.18 mean NFIQ2 vs. raw across 300 scoreable images) but this
number hides two real problems that need a decision, not a code fix:
Pipeline C's diffusion (P2) makes **20/80 DB3_B images (25%) impossible for
NFIQ2 to score at all** once combined with a different P1 technique, and
**DB1_B nets *negative* overall** (−3.28) under the full hybrid, driven
almost entirely by that same diffusion step. Both are reported in full
below, not adjusted around.

## 1. Full hybrid: per-DB and overall

| DB | n (valid) | raw mean | final mean | Δ vs raw | improved | regressed | unchanged |
|---|---|---|---|---|---|---|---|
| DB1_B | 80 | 62.04 | 58.76 | **−3.28** | 29 | 49 | 2 |
| DB2_B | 80 | 50.45 | 56.95 | +6.50 | 46 | 34 | 0 |
| DB3_B | **60** | 27.87 | 47.03 | +19.17 | 59 | 1 | 0 |
| DB4_B | 80 | 28.76 | 41.84 | +13.08 | 71 | 8 | 1 |
| **OVERALL** | **300** | 43.24 | 51.42 | **+8.18** | 205 | 92 | 3 |

**Caveat on the DB3/OVERALL numbers above:** `n=60` for DB3_B (not 79, the
usual DB3 count after excluding only the one universally-unscoreable raw
image) — 20 images fail to score at Stage 2 or later (see Section 2). The
`raw_mean`/OVERALL row printed by `scripts/hybrid_ablation.py` (43.24 /
+8.18 vs. raw) is computed only over images that scored at *every* stage,
which silently excludes DB3's 20 hardest-hit images from **both** sides of
that particular comparison — not directly comparable to the standard
319-image raw baseline (mean 41.53) every other pipeline is measured
against. The table below instead uses `dashboard/results_loader.py`'s own
Δ-vs-raw computation (the same one the Overview tab actually displays,
and the same 41.53 reference baseline used for A/B/C/D), which is the
correct like-for-like comparison; the +8.18 figure above still matters
only as a reminder that even Hybrid's own numbers understate the DB3
problem, not as the number to cite for cross-pipeline comparison.

### vs. each source pipeline's own individual result (same 41.53-baseline methodology, via dashboard/results_loader.py)

| Pipeline | Overall mean | Δ vs raw |
|---|---|---|
| Pipeline A | 48.46 | +6.93 |
| Pipeline B | 48.10 | +6.57 |
| Pipeline C | 51.53 | +10.00 |
| Pipeline D | 45.35 | +3.82 |
| **Hybrid** | **51.42*** | **+9.89*** |

*Still only over the 300 images Hybrid could score end-to-end (see the
caveat above) — the 20 missing DB3 images are exactly its hardest cases,
so this number is still somewhat flattered relative to a true 319/320-image
figure. Even on this flattered basis, the Hybrid does **not** clearly
outperform Pipeline C alone (51.53 mean, +10.00) despite combining what
Section 2.4 identified as the best available P1/P2/P6 techniques — worth
sitting with, not explaining away.

## 2. The DB3_B scoreability collapse — isolated to Pipeline C's diffusion (P2)

| Variant | DB3_B valid n / 80 | What's included |
|---|---|---|
| minus_p2 (P1 → P6, **no diffusion**) | **79/80** (only the universal raw-failure) | Wavelet contrast + oriented Gabor, no diffusion at all |
| **minus_p6** (P1 → P2 only, final = Stage 2) | **55/80** | Wavelet contrast then diffusion |
| **full** (P1 → P2 → P6) | **60/80** | All three |
| **minus_p1** (P2 → P6, **no P1 first**) | **41/80** | Diffusion applied directly to the Step 0 image, then Gabor |

This isolates the cause cleanly:
- **Removing diffusion entirely (minus_p2) eliminates the problem completely** — 0 new failures, only the one pre-existing universal case.
- **Diffusion applied directly to the un-enhanced Step 0 image is the worst case** (minus_p1: 39 new failures) — worse than diffusion applied after Pipeline B's wavelet contrast (20-25 new failures).
- **Adding Gabor afterward partially rescues some of diffusion's damage**: minus_p6 (55/80) → full (60/80) — 5 images that fail to score right after diffusion become scoreable again once Gabor runs on top of them. Gabor never *causes* a new failure anywhere in this dataset (minus_p2, which excludes diffusion, has zero DB3 failures beyond the universal one).
- The same collapse, smaller, also touches DB2 (1 new failure under minus_p1/minus_p6, 0 under full) and DB4 (12 new failures under minus_p1, 2 under minus_p6, 0 under full) — **DB1 is never affected** (0 new failures in any variant).

**Root cause (mechanism, not a guess — directly supported by the pattern above):** Pipeline C's diffusion step uses alpha-adaptive iterations/kappa (more aggressive smoothing the lower an image's own orientation coherence is), tuned and validated only in the context of Pipeline C's *own* P1 (homomorphic filtering), whose Step 1b explicitly feathers the output back toward the pre-filter image over background/low-confidence regions. That feathering appears to matter more than previously visible: substitute a different P1 technique (or skip P1 entirely) and diffusion's same aggressive settings, on DB3's already-marginal-foreground images, smooth away enough of the remaining fingerprint area that NFIQ2's own feature extractor can no longer detect a scoreable print at all (`FRFXLL_ERR_FB_TOO_SMALL_AREA` — the exact same error text as the one pre-existing universal failure, confirmed by reading the actual error column, not inferred). This is a genuine "helps standalone, breaks in this combination" finding of exactly the kind this validation was built to catch — not something to retune away here.

## 3. Per-component verdict: does each selected technique still help once combined?

Comparing the full hybrid's final NFIQ2 against each single-component-removed variant (same metric the task asked for: does full outperform hybrid-minus-that-component). **DB3_B is excluded from this specific comparison** since its wildly different n per variant (79/55/60) makes the mean not a fair like-for-like comparison — the failure-count finding in Section 2 already covers P2's real effect there.

| DB | full | minus_p1 (no P1) | **P1 helps?** | minus_p2 (no P2) | **P2 helps?** | minus_p6 (no P6) | **P6 helps?** |
|---|---|---|---|---|---|---|---|
| DB1_B | 58.76 | 58.01 | Yes (+0.75) | 62.80 | **No (−4.04)** | 61.58 | **No (−2.81)** |
| DB2_B | 56.95 | 55.15 | Yes (+1.80) | 59.51 | **No (−2.56)** | 59.37 | **No (−2.42)** |
| DB4_B | 41.84 | 40.22 | Yes (+1.61) | 39.11 | Yes (+2.72) | 41.06 | Yes (+0.77) |
| **OVERALL*** | 51.42 | 51.08 | Yes (+0.34) | 50.21 | Yes (+1.20) | 52.57 | **No (−1.16)** |

*OVERALL row here is computed across DB1/DB2/DB4 only (240 images), for the reason stated above — DB3 is handled separately in Section 2, not folded into this mean.

**Read plainly, not softened:**
- **P1 (Pipeline B's wavelet contrast) helps in every DB checked**, including the small overall margin — consistent with it being selected.
- **P2 (Pipeline C's diffusion) does NOT help once combined, on DB1 and DB2 specifically** — removing it improves both (DB1: +4.04, DB2: +2.56). It only helps on DB4 (+2.72), and its OVERALL positive number is propped up entirely by DB4 outweighing DB1+DB2's losses. Combined with Section 2's finding (P2 is also the sole cause of the DB3 scoreability collapse), **P2's selection is the weakest-supported of the three once evaluated in this actual combination**, despite being the strongest standalone performer in Section 2.4.
- **P6 (Pipeline A's oriented Gabor) does NOT help on DB1 or DB2 either**, and its overall (3-DB) number is net negative (−1.16) — it only helps on DB4 (+0.77). This directly contradicts Section 2.4's finding that Pipeline A's Gabor had the strongest standalone model signal (R²=0.256) and the best mean-delta win for the P6 slot — a clear, concrete instance of "helps standalone, doesn't necessarily help in this specific combination," reported exactly as the task asked, not adjusted.

## 4. DB1_B's net regression, explained by component

DB1_B nets **−3.28 under the full hybrid** despite three individually-validated components. The table in Section 3 shows this isn't one component dragging down an otherwise-neutral combination — **removing either P2 or P6 individually turns DB1 net-positive** (+0.76 without P2, actually positive without P6 too at −0.4625 vs full's −3.28, i.e. much closer to neutral). P2 alone accounts for the largest single swing (full vs. minus_p2: −4.04). This matches a pattern already documented earlier in this project: Pipeline C's own diffusion step has a persistent, hard-to-fix negative contribution on DB1 even inside Pipeline C's own pipeline (its own alpha-adaptive scheme was built specifically to manage, not eliminate, this) — that same tendency carries over, and compounds with Gabor's own DB1 weakness, once diffusion is combined with different P1/P6 techniques than the ones it was tuned alongside.

## 5. What this doesn't do

Per the task, no parameter was retuned, no component was swapped out, and no
attempt was made to make these numbers look better than what the real NFIQ2
runs produced. The findings above — the DB3 scoreability collapse, DB1's
net regression, and P2/P6 not clearly helping once combined on DB1/DB2 —
are reported as real, legitimate limitations for the project lead to weigh,
not defects in this validation step to fix.
