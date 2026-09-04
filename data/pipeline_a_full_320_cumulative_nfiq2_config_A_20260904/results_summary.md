# Pipeline A full 320-image cumulative NFIQ2 validation

Fixed configuration A; NFIQ2 v2.3.0; serial shared scoring workflow.

The full 320 images include the prior 16 development pilot images and are not an independent test set.

| DB | Raw mean | Stage 3 mean | Mean delta | Improved | Regressed | Unchanged | Valid pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| DB1_B | 62.038 | 64.362 | 2.325 | 44 | 34 | 2 | 80 |
| DB2_B | 50.450 | 56.300 | 5.850 | 56 | 21 | 3 | 80 |
| DB3_B | 24.646 | 34.487 | 10.051 | 62 | 16 | 1 | 79 |
| DB4_B | 28.762 | 38.675 | 9.912 | 64 | 13 | 3 | 80 |
| Overall | 41.527 | 48.456 | 7.025 | 226 | 84 | 9 | 319 |

## Worst 10 final regressions

| DB | File | Raw | Stage 3 | Delta |
|---|---|---:|---:|---:|
| DB1_B | 102_6.tif | 87.0 | 59.0 | -28.0 |
| DB1_B | 106_4.tif | 77.0 | 53.0 | -24.0 |
| DB1_B | 104_7.tif | 76.0 | 56.0 | -20.0 |
| DB4_B | 108_2.tif | 23.0 | 3.0 | -20.0 |
| DB2_B | 104_7.tif | 69.0 | 51.0 | -18.0 |
| DB1_B | 105_3.tif | 77.0 | 60.0 | -17.0 |
| DB2_B | 104_3.tif | 82.0 | 66.0 | -16.0 |
| DB1_B | 107_7.tif | 76.0 | 61.0 | -15.0 |
| DB1_B | 101_7.tif | 65.0 | 51.0 | -14.0 |
| DB3_B | 108_1.tif | 33.0 | 19.0 | -14.0 |
