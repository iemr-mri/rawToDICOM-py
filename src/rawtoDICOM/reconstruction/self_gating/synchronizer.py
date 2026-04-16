"""Slice synchronizer for self-gated SAX stacks.

Translates sliceSynchronizer.m.

Self-gated slices from the same session are independently reconstructed and
therefore not phase-locked to each other.  This module aligns them to a
common cardiac phase (diastole) by:

  1. Detecting diastole automatically in the center slice using a minimum-air
     heuristic (replaces the interactive MATLAB ROI).
  2. Aligning each remaining slice to its already-aligned neighbour by finding
     the circular shift that minimises the RMS image difference.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def synchronize_slices(
    images: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Align a SAX image stack so that frame 0 is diastole.

    Translates sliceSynchronizer.m.  The interactive ROI selection is replaced
    by an automated centre-crop approach: the centre 50 % of the image (both
    axes) is used as the region of interest for diastole detection.

    For single-slice stacks the function returns the input unchanged (with
    diastole at frame 0).

    Args:
        images: Shape [x, y, slices, frames].

    Returns:
        Synchronized images, same shape as input, frame 0 ≈ diastole.
    """
    images = np.array(images, dtype=float)  # work on a copy
    x_pixels, y_pixels, n_slices, n_frames = images.shape

    # --- Automated diastole detection on the centre slice ---
    mid_idx = n_slices // 2
    mid_slice = images[:, :, mid_idx, :]  # [x, y, frames]

    # Centre-crop (inner 50 % of each spatial axis).
    x0, x1 = x_pixels // 4, 3 * x_pixels // 4
    y0, y1 = y_pixels // 4, 3 * y_pixels // 4
    center_region = mid_slice[x0:x1, y0:y1, :]  # [cx, cy, frames]

    # Pixels below 60 % of the mean intensity are "air".
    mean_intensity = float(np.mean(center_region))
    air_threshold = 0.6 * mean_intensity
    air_counts = np.sum(center_region < air_threshold, axis=(0, 1))  # [frames]

    # Smooth air-count trend with a rolling mean (window = floor(n_frames/10)
    # neighbours on each side), matching MATLAB.
    n_neighbours = max(1, n_frames // 10)
    kernel_size = 2 * n_neighbours + 1
    kernel = np.ones(kernel_size) / kernel_size
    # Pad circularly so smoothing wraps at the frame boundary.
    padded = np.concatenate([air_counts[-n_neighbours:], air_counts, air_counts[:n_neighbours]])
    smooth_air = np.convolve(padded, kernel, mode="valid")

    diastole_frame = int(np.argmin(smooth_air))
    # Shift center slice so diastole is at frame 0.
    images[:, :, mid_idx, :] = np.roll(images[:, :, mid_idx, :], -diastole_frame, axis=-1)

    if n_slices == 1:
        return images

    # --- Slice-to-slice synchronization ---
    # Process slices outward from the center: first upward (mid+1 … top),
    # then downward (mid-1 … bottom).  Each slice is aligned to its already-
    # synchronized neighbour.
    slice_sequence = list(range(mid_idx + 1, n_slices)) + list(range(mid_idx - 1, -1, -1))

    for slice_idx in slice_sequence:
        # Neighbour is the previously aligned adjacent slice.
        if slice_idx > mid_idx:
            neighbour_idx = slice_idx - 1
        else:
            neighbour_idx = slice_idx + 1

        cur_slice = images[:, :, slice_idx, :]   # [x, y, frames]
        neigh_slice = images[:, :, neighbour_idx, :]

        # Try all circular shifts and pick the one minimising RMS difference.
        rms_diffs = np.zeros(n_frames)
        for shift in range(n_frames):
            shifted = np.roll(cur_slice, shift, axis=-1)
            diff = neigh_slice - shifted
            rms_diffs[shift] = float(np.sqrt(np.mean(diff**2)))

        best_shift = int(np.argmin(rms_diffs))
        images[:, :, slice_idx, :] = np.roll(cur_slice, best_shift, axis=-1)

    result: npt.NDArray[np.floating] = images
    return result
