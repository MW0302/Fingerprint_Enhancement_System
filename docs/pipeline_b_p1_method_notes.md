# Pipeline B Stage 1: wavelet contrast enhancement

## Scope

This revision implements only Pipeline B's P1 technique. Shared Step 0 runs
first, then `_wavelet_contrast()` produces the Stage 1 grayscale image. P2
wavelet shrinkage denoising and P6 orientation-steered morphology remain
unimplemented so the three techniques can later be evaluated cumulatively.

## Method

The implementation performs a multilevel separable 2-D discrete wavelet
transform, preserves the approximation coefficients, and adaptively amplifies
the horizontal, vertical, and diagonal detail coefficients. The mapping gives
larger-magnitude coefficients more of the requested gain and leaves weak
coefficients closer to their input magnitude. This limits premature noise
amplification before Pipeline B's future denoising stage. The reconstructed
detail increment is applied only inside the shared foreground mask.

The default selected by the 16-image pilot is:

- Daubechies 4 (`db4`), symmetric boundary handling, three levels
- coarsest-detail gain 1.60, interpolated to finest-detail gain 1.00
- coefficient reliability floor at the 25th non-zero magnitude percentile
- approximation coefficients unchanged; full foreground blend

This is an original project implementation, not a reproduction of one cited
algorithm. The references support the use of multiresolution wavelet detail
processing and adaptive coefficient modification for fingerprint enhancement.

## Pilot decision

The fixed pilot uses four FVC2002 Set B images from each of DB1-DB4. Among 21
bounded candidates, the highest mean gain was not chosen automatically. The
selected default improved 12 of 16 images, regressed 3, and produced an overall
mean NFIQ2 change of +3.8125 with a worst change of -5. Every database had a
positive mean change: DB1 +1.25, DB2 +8.75, DB3 +3.50, and DB4 +1.75.

The higher-gain alternative averaged +5.00 but had a worse minimum change of
-7. Visual review of all 16 comparisons found no obvious seams, ringing,
clipping, or foreground spill in the selected default. These figures are pilot
tuning evidence, not the final 320-image Pipeline B result.

Reproduce the pilot with:

```powershell
python scripts/pipeline_b_p1_pilot.py
```

## References

1. C.-W. Hsieh, E. Lai, and Y.-C. Wang, "An effective algorithm for fingerprint
   image enhancement based on wavelet transform," *Pattern Recognition*,
   36(2), 2003. <https://doi.org/10.1016/S0031-3203(02)00032-8>
2. Y. Lei et al., "Fingerprint enhancement based on non-separable wavelet,"
   *2010 9th IEEE International Conference on Cognitive Informatics*, 2010.
   <https://doi.org/10.1109/COGINF.2010.5599722>
3. S. G. Mallat, "A theory for multiresolution signal decomposition: the
   wavelet representation," *IEEE Transactions on Pattern Analysis and Machine
   Intelligence*, 11(7), 1989. <https://doi.org/10.1109/34.192463>
4. PyWavelets developers, "2D Forward and Inverse Discrete Wavelet Transform."
   <https://pywavelets.readthedocs.io/en/latest/ref/2d-dwt-and-idwt.html>
