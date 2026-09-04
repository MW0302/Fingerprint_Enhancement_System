# Pipeline A limited single-parameter candidate scoring

All four configurations use the same 16 samples, NFIQ2 2.3.0, shared Step 0,
serial `run_nfiq2_single()` scoring, and original-size grayscale uint8 TIFF
stage images. Each configuration independently recomputes Stage 1, Stage 2,
the orientation/coherence fields, and Stage 3. No downstream result is reused
from configuration A after an upstream parameter changes.

Configurations:

- A: CLAHE 2.0, bilateral sigma_color 35, Gabor strength 0.7.
- B: only CLAHE changes to 1.5.
- C: only bilateral sigma_color changes to 25.
- D: only Gabor strength changes to 0.4.

All 224 distinct scoring calls succeeded: 16 shared Raw + 16 shared Step 0 +
4 configurations × 16 samples × 3 technique stages. The long CSV repeats the
shared Raw/Step 0 scores under every configuration, yielding 320 recorded
configuration-stage rows. No missing score was replaced with zero or removed.

## Overall results

| Config | Valid | Mean Stage 1−Raw | Mean Stage 2−Stage 1 | Mean Stage 3−Stage 2 | Mean Stage 3−Raw | Mean vs A | Improved / regressed / unchanged | Worst final regression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A | 16 | +3.000 | +1.063 | +4.563 | +8.625 | 0.000 | 13 / 2 / 1 | −8, DB2_B/102_6 |
| B | 16 | +3.500 | −0.063 | +5.813 | +9.250 | +0.625 | 14 / 2 / 0 | −9, DB2_B/102_6 |
| C | 16 | +3.000 | −0.375 | +5.563 | +8.188 | −0.438 | 12 / 4 / 0 | −5, DB3_B/107_7 |
| D | 16 | +3.000 | +1.063 | +3.313 | +7.375 | −1.250 | 12 / 3 / 1 | −9, DB2_B/103_2 |

B has the highest mean final delta in this limited pilot, but its +0.625
advantage over A is small, varies by database, and comes with a slightly worse
single-image regression (−9 vs A's −8). This is not an automatic default
selection.

## Mean Stage 3−Raw by database

| Database | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| DB1_B | +13.25 | +12.00 | +13.00 | +9.50 |
| DB2_B | +3.00 | +5.25 | +5.00 | +2.00 |
| DB3_B | +5.50 | +5.75 | +3.25 | +4.50 |
| DB4_B | +12.75 | +14.00 | +11.50 | +13.50 |

B is above A on DB2, DB3, and DB4 but below A on DB1. C improves the DB2 mean
but is below A on the other three databases. D improves DB4 but is below A on
DB1–DB3. Each database contains only four pilot samples.

## Priority images

- `DB3_B/101_5.tif`: Raw 42; A 44, B 44, C 39, D 49. D is best on this image,
  while C regresses below Raw. This does not generalize to D's overall result.
- `DB2_B/102_6.tif`: Raw 66; A 58, B 57, C 65, D 73. D avoids A/B's large
  regression and is best here; B has the worst final delta (−9).
- `DB3_B/107_7.tif`: Raw 12; A 6, B 7, C 7, D 6. B/C are one point above A,
  but all four configurations still regress substantially from Raw.

Other images materially affect the aggregate: B gains +10 vs A on
`DB2_B/103_2` and +11 on `DB4_B/104_4`, but loses −6 on `DB1_B/102_7` and
`DB4_B/102_4`. Candidate choice therefore must not be based only on the three
priority images.

## Baseline consistency

Configuration A matches the previous cumulative pilot exactly for Raw, Step 0,
Stage 1, Stage 2, and Stage 3 on every one of the 16 samples. See
`baseline_A_consistency_check.csv` for the 80 boolean checks.

These findings are a deliberately limited single-parameter pilot. B/C/D were
not combined, no per-image dynamic selection was used, and the search was not
expanded.
