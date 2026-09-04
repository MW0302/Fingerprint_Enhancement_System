# Pipeline A 16-sample cumulative NFIQ2 pilot

## Environment and execution

- NFIQ2 version: 2.3.0, confirmed from the executable's internal Version Info.
- Shared helper: `src/utils/common.py::run_nfiq2_single()`.
- CLI pattern: `nfiq2.exe -i <image_path> -a -F -o <temp_csv>`.
- Smoke test: `DB1_B/102_7.tif` returned 31 with no error.
- All scoring calls were serial because the shared helper uses a single fixed
  temporary CSV filename.
- Stage images were saved as original-size, 2D grayscale `uint8` TIFF files
  with `cv2.imwrite`, matching the Pipeline C ablation/batch workflow.
- All 80 stage scores succeeded. No score was replaced with zero or removed.
- Stage 3 matched `enhance(raw)` pixel-for-pixel for all 16 images.

## Per-database mean deltas

Every database has four valid pairs for every comparison.

| Database | Step 0 − Raw | Stage 1 − Step 0 | Stage 1 − Raw | Stage 2 − Stage 1 | Stage 3 − Stage 2 | Stage 3 − Raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DB1_B | 0.00 | +3.75 | +3.75 | +3.50 | +6.00 | +13.25 |
| DB2_B | 0.00 | +1.00 | +1.00 | +2.50 | −0.50 | +3.00 |
| DB3_B | 0.00 | +1.50 | +1.50 | −3.25 | +7.25 | +5.50 |
| DB4_B | 0.00 | +5.75 | +5.75 | +1.50 | +5.50 | +12.75 |
| Overall (16) | 0.00 | +3.00 | +3.00 | +1.0625 | +4.5625 | +8.625 |

## Improved / regressed / unchanged counts

Counts below are `improved / regressed / unchanged`; each database row uses
four valid pairs. Exact valid/missing counts are retained in
`per_database_summary.csv`.

| Database | Step 0 − Raw | Stage 1 − Step 0 | Stage 1 − Raw | Stage 2 − Stage 1 | Stage 3 − Stage 2 | Stage 3 − Raw |
| --- | --- | --- | --- | --- | --- | --- |
| DB1_B | 0 / 0 / 4 | 3 / 1 / 0 | 3 / 1 / 0 | 3 / 1 / 0 | 4 / 0 / 0 | 4 / 0 / 0 |
| DB2_B | 0 / 0 / 4 | 2 / 2 / 0 | 2 / 2 / 0 | 3 / 1 / 0 | 2 / 1 / 1 | 2 / 1 / 1 |
| DB3_B | 0 / 0 / 4 | 2 / 2 / 0 | 2 / 2 / 0 | 2 / 2 / 0 | 3 / 1 / 0 | 3 / 1 / 0 |
| DB4_B | 0 / 0 / 4 | 3 / 0 / 1 | 3 / 0 / 1 | 2 / 1 / 1 | 3 / 1 / 0 | 4 / 0 / 0 |
| Overall (16) | 0 / 0 / 16 | 10 / 5 / 1 | 10 / 5 / 1 | 10 / 5 / 1 | 12 / 3 / 1 | 13 / 2 / 1 |

## Notable individual results

- `DB3_B/102_4.tif`: 26 → 26 → 31 → 37 → 50, total +24.
- `DB4_B/104_6.tif`: 31 → 31 → 34 → 37 → 54, total +23.
- `DB1_B/102_7.tif`: 31 → 31 → 42 → 47 → 52, total +21.
- `DB2_B/102_6.tif`: Stage 2 rose to 74, then Stage 3 fell to 58; total −8.
- `DB3_B/107_7.tif`: 12 → 12 → 10 → 7 → 6, total −6.
- `DB3_B/101_5.tif`: Stage 1 rose to 49, Stage 2 fell to 32, and Stage 3
  recovered to 44; total +2.

These pilot measurements describe only the selected 16 samples and do not
replace a full 320-image validation.
