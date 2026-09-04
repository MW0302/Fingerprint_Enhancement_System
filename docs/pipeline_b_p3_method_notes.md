# Pipeline B Stage 3: orientation-steered morphology

## Scope and method

Pipeline B's P6 stage uses the shared `orientation_field()` result to steer
grayscale morphological closing. The shared function already converts the
structure tensor's gradient direction to ridge direction by adding 90
degrees, so P6 does not rotate it again.

Because FVC2002 ridges are dark, Stage 2 is inverted before closing so ridges
become the bright morphological foreground. Twelve short line structuring
elements cover orientations over 180 degrees. At every pixel, responses from
the two neighbouring orientation bins are linearly interpolated using a
doubled-angle orientation representation; this handles the equivalent
-90/+90 degree boundary and avoids hard block seams. The result is inverted
back and blended into the original Stage 2 grayscale image.

The default selected by the pilot is:

- line-kernel length: 7 pixels
- orientation bins: 12
- morphology strength: 0.50
- coherence floor/power: 0.20 / 1.0
- maximum per-pixel ridge darkening before strength scaling: 16 gray levels

Coherence and the shared foreground mask continuously gate the change. The
function returns only an original-size grayscale `uint8` image; no binary
image is returned or scored.

## Fixed 16-image cumulative pilot

The pilot reused the same four images from each FVC2002 Set B database and
kept P1/P2 defaults locked. It compared 30 bounded P6 configurations.

| Scope | Mean Stage 3 - Stage 2 | Mean Stage 3 - Raw |
|---|---:|---:|
| DB1_B | +2.00 | +4.00 |
| DB2_B | +0.75 | +9.50 |
| DB3_B | +4.00 | +8.75 |
| DB4_B | +3.00 | +5.25 |
| Overall | +2.4375 | +6.875 |

P6 improved 12 of 16 images and regressed 4, with a worst incremental change
of -3. The highest-mean candidate (+2.8125) was not selected because it
improved fewer images and had a worse minimum change of -5. These are pilot
tuning results, not the final 320-image Pipeline B evidence. Visual review of
all 16 images found no obvious cross-ridge merging, wrong-direction strokes,
block seams, background spill, or grayscale clipping.

Reproduce with:

```powershell
python scripts/pipeline_b_p3_pilot.py
```

## References

1. G. Milici, G. Raia, S. Vitabile, and F. Sorbello, "Fingerprint Image
   Enhancement Using Directional Morphological Filter," *EUROCON 2005*.
   <https://doi.org/10.1109/EURCON.2005.1630108>
2. L. Hong, Y. Wan, and A. Jain, "Fingerprint Image Enhancement: Algorithm
   and Performance Evaluation," *IEEE Transactions on Pattern Analysis and
   Machine Intelligence*, 20(8), 1998.
   <https://doi.org/10.1109/34.709565>
3. OpenCV developers, "Morphological Transformations."
   <https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html>
