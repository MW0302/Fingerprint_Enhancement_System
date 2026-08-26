"""
Pipeline C — Member C: Coherence-Preserving Denoising
(Coherence Diffusion + 2D Log-Gabor Filtering)

Targets (see Dataset_Problem_Analysis_and_Revised_Pipelines.docx, Section 6):
    P2 — high random noise (DB3)
    P6 — weak/inconsistent ridge orientation (DB4, DB3)
    P4 — blurred/soft ridges (secondary, cross-check against Pipeline D)

Citations (see Team_Member_Starter_Packets.docx for the full list):
    Perona & Malik (1990) — anisotropic / coherence-enhancing diffusion
    Field (1987) — Log-Gabor filtering
    Shams et al. (2023) — methodological inspiration (this is the group's own
                           Literature Review 2 — anchor your lit review here)
    Hong, Wan, & Jain (1998) — shared segmentation + orientation field steps

Steps 1 and 2 are implemented (shared). Steps 3 and 5 are TODOs: coherence
diffusion and 2D Log-Gabor filtering are the techniques you are responsible
for researching and implementing yourself. Step 4 (ridge-frequency
estimation) is a small supporting calculation needed before step 5.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from common import segment, orientation_field  # noqa: E402
from config import RAW_DIR  # noqa: E402

import cv2
import numpy as np


def _estimate_ridge_frequency(image, theta_field, block=16):
    """Very rough starting point for local ridge-frequency estimation:
    projects each block along its own orientation and looks for the spacing
    between intensity peaks. Replace or refine this — the accuracy of this
    estimate directly affects how well the Log-Gabor filter in step 5 performs."""
    h, w = image.shape
    freq_field = np.zeros_like(theta_field)
    # TODO: implement properly (e.g. following Hong et al., 1998's
    # frequency-estimation procedure). Left as a stub with a plausible
    # default so the pipeline still runs end-to-end while you work on it.
    freq_field[:] = 1.0 / 9.0  # placeholder: assumes ~9px ridge spacing everywhere
    return freq_field


def enhance(image, params=None):
    """
    image: 2D grayscale numpy array
    params: optional dict of tunable parameters
    returns: enhanced 2D grayscale numpy array, same shape as image
    """
    params = params or {}

    # Step 1: segmentation (shared)
    fg_mask_blocks, block_var = segment(image)

    # Step 2: ridge orientation field estimation (shared; Hong et al., 1998)
    theta_field, coherence_field = orientation_field(image)

    # Step 3: coherence-enhancing anisotropic diffusion (Perona & Malik, 1990),
    # steered by theta_field — TODO
    # Unlike an isotropic Gaussian blur, this should smooth MORE along the
    # ridge direction (theta_field) and LESS across it, which is what
    # preserves ridge structure while still reducing noise. Ask yourself:
    #   - how many diffusion iterations / what step size?
    #   - how do you stop diffusion at edges (the "conductance" function in
    #     Perona & Malik, 1990)?
    diffused = image.copy()  # placeholder — replace once diffusion is implemented

    # Step 4: local ridge-frequency estimation (supporting step for step 5)
    freq_field = _estimate_ridge_frequency(diffused, theta_field)

    # Step 5: 2D Log-Gabor filtering (Field, 1987; Shams et al., 2023) — TODO
    # Build filters in the frequency domain (np.fft.fft2 / fftshift) using
    # theta_field and freq_field to steer each region's filter, following
    # the general approach in Shams et al. (2023).
    enhanced = diffused  # placeholder — replace once Log-Gabor filtering is implemented

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
