"""Iterative compressed-sensing reconstruction.

Unifies reconstructCS.m (CINE) and CSreconstructor.m (self-gating) into a single
function parameterised by CSConfig.  Both MATLAB versions implement the same algorithm;
the Python version uses the cleaner convergence logic from CSreconstructor.m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from rawtoDICOM.reconstruction.kspace import fft2c, ifft2c


@dataclass(frozen=True)
class CSConfig:
    """Configuration for iterative CS reconstruction.

    Attributes:
        max_iterations:        Maximum number of CS iterations.
        percentile_threshold:  Percentile used to set the soft-threshold value on
                               the first iteration (0–100).  Default 50 matches both
                               MATLAB originals (CS_percentThresh = 50).
        convergence_threshold: Stop when |Δrms / Δrms_iter1| drops below this
                               fraction.  Default 0.01 (1 %) matches both originals.
    """

    max_iterations: int = 50
    percentile_threshold: float = 50.0
    convergence_threshold: float = 0.01


def reconstruct_cs(
    kspace: npt.NDArray[Any],
    config: CSConfig = CSConfig(),
) -> npt.NDArray[Any]:
    """Iterative compressed-sensing reconstruction.

    Translates reconstructCS.m / CSreconstructor.m.

    Algorithm per (slice, coil):
      1. IFFT to image space for all frames.
      2. Temporal FFT → soft-threshold → temporal IFFT.
      3. Restore originally acquired k-space lines.
      4. Repeat until convergence or max_iterations.

    Args:
        kspace: Complex array [x, y, slices, frames, coils].
                Zero at unsampled phase-encode lines.
        config: CS hyperparameters.

    Returns:
        Reconstructed k-space with the same shape as the input.
    """
    x_size, y_size, n_slices, n_frames, n_coils = kspace.shape

    us_mask = kspace != 0  # undersampling mask: True where data was acquired
    no_data_mask = ~us_mask
    kspace_cs: npt.NDArray[Any] = np.zeros_like(kspace)

    for s in range(n_slices):
        for c in range(n_coils):
            kspace_sc = kspace[:, :, s, :, c]           # [x, y, frames]
            mask_sc = us_mask[:, :, s, :, c]             # [x, y, frames]
            no_data_sc = no_data_mask[:, :, s, :, c]

            # Initialise image-space time series via IFFT
            im_temp: npt.NDArray[Any] = ifft2c(kspace_sc)  # [x, y, frames]

            thresh_val = 0.0
            diff_rms = np.zeros(config.max_iterations)

            for iteration in range(config.max_iterations):

                # Temporal FFT along frames axis (axis 2)
                kspace_temporal: npt.NDArray[Any] = np.fft.fft(im_temp, axis=2)

                # Set threshold from first-iteration temporal k-space magnitude
                if iteration == 0:
                    thresh_val = float(
                        np.percentile(np.abs(kspace_temporal), config.percentile_threshold)
                    )

                # Soft-threshold and transform back to image space
                kspace_temporal = _soft_thresh(kspace_temporal, thresh_val)
                im_temp = np.fft.ifft(kspace_temporal, axis=2)

                # Measure RMS error on acquired lines only
                kspace_after = fft2c(im_temp) * mask_sc
                diff = kspace_sc - kspace_after
                selected = diff[mask_sc]
                diff_rms[iteration] = float(np.sqrt(np.mean(np.abs(selected) ** 2)))

                # Restore originally acquired k-space lines
                adjusted = kspace_sc + fft2c(im_temp) * no_data_sc
                im_temp = ifft2c(adjusted)

                # Convergence check: stop when change is < 1 % of first-iteration change
                if iteration > 0:
                    diff_current = diff_rms[iteration] - diff_rms[iteration - 1]
                    diff_first = diff_rms[1] - diff_rms[0]
                    ratio = abs(diff_current / diff_first) if diff_first != 0 else 1.0
                    converged = ratio < config.convergence_threshold
                    if converged:
                        break

            kspace_cs[:, :, s, :, c] = fft2c(im_temp)

    return kspace_cs


def _soft_thresh(x: npt.NDArray[Any], threshold: float) -> npt.NDArray[Any]:
    """Complex soft thresholding.

    Reduces magnitude by threshold; preserves phase.  Values with magnitude
    ≤ threshold are zeroed.  Matches SoftThresh.m / softThresh in CSreconstructor.m.
    """
    magnitude = np.abs(x)
    # Avoid 0/0 by substituting 1.0 where the condition is False (scale will be 0 there anyway).
    safe_magnitude: npt.NDArray[Any] = np.where(magnitude > threshold, magnitude, 1.0)
    scale: npt.NDArray[Any] = np.where(
        magnitude > threshold, (magnitude - threshold) / safe_magnitude, 0.0
    )
    result: npt.NDArray[Any] = x * scale
    return result
