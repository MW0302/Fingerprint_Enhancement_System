# Pipeline B Stage 2: wavelet shrinkage denoising

## Scope and method

This revision implements only Pipeline B's P2 technique after the locked P1
output. `_wavelet_shrinkage_denoise()` performs a three-level `db4` transform,
retains the approximation and coarser detail levels, and applies subband-wise
BayesShrink soft thresholds to the finest horizontal, vertical, and diagonal
details. It returns a grayscale `uint8` image; P6 remains unimplemented.

The base noise standard deviation is estimated by the median absolute
deviation of foreground coefficients in the finest diagonal subband. For
each processed subband, the BayesShrink threshold is:

```text
T = sigma_noise^2 / sigma_signal
```

where `sigma_signal` is derived from observed variance minus estimated noise
variance. `pywt.threshold(..., mode="soft")` shrinks surviving coefficients
toward zero rather than discontinuously retaining their original magnitude.

Fingerprint ridge energy can also occur in the finest bands, so the base
threshold is continuously strength-scaled using Immerkær's 3x3 random-noise
estimate. The fixed scaling rule is `(measured_sigma / 5)^4`, clipped to
`[0.10, 1.00]`. The non-zero floor keeps P2 active on cleaner images while
avoiding the large DB2 regressions observed with uniform full-strength
shrinkage. This estimator only controls the strength of the same P2 method;
it is not a separate enhancement technique or a per-database rule.

## Fixed 16-image pilot

The pilot used the same four images from each FVC2002 Set B database as P1.
It compared 28 bounded configurations. The selected default produced:

| Scope | Mean Stage 2 - Stage 1 | Mean Stage 2 - Raw |
|---|---:|---:|
| DB1_B | +0.75 | +2.00 |
| DB2_B | 0.00 | +8.75 |
| DB3_B | +1.25 | +4.75 |
| DB4_B | +0.50 | +2.25 |
| Overall | +0.625 | +4.4375 |

Across 16 images, P2 improved 7, left 5 unchanged, regressed 4, and had a
worst Stage 2 minus Stage 1 change of -3. The highest-mean alternative was
not selected because it had one additional regression and a worse minimum
change. Visual review of all 16 images found no obvious ridge blurring,
ringing, foreground-mask seams, or grayscale clipping. The selected P2 step
reduced the pilot's mean Immerkær noise estimate from 5.13 to 3.52 in DB3.
These are pilot tuning results, not final 320-image evidence.

Reproduce with:

```powershell
python scripts/pipeline_b_p2_pilot.py
```

## References

1. S. G. Chang, B. Yu, and M. Vetterli, "Adaptive Wavelet Thresholding for
   Image Denoising and Compression," *IEEE Transactions on Image Processing*,
   9(9), 2000. <https://doi.org/10.1109/83.862633>
2. D. L. Donoho and I. M. Johnstone, "Ideal Spatial Adaptation by Wavelet
   Shrinkage," *Biometrika*, 81(3), 1994.
   <https://doi.org/10.1093/biomet/81.3.425>
3. J. Immerkær, "Fast Noise Variance Estimation," *Computer Vision and Image
   Understanding*, 64(2), 1996.
   <https://doi.org/10.1006/cviu.1996.0060>
4. PyWavelets developers, "Thresholding functions."
   <https://pywavelets.readthedocs.io/en/latest/ref/thresholding-functions.html>
