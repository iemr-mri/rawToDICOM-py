"""Image geometry corrections for DICOM output.

Translates imageCorrections.m and sliceShuffler.m.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.scan import BrukerScan


def apply_corrections(
    images: npt.NDArray[np.floating],
    scan: BrukerScan,
) -> npt.NDArray[np.int16]:
    """Shift phase offset and normalise to int16 with a 30 000-count ceiling.

    Translates imageCorrections.m.

    Computes the phase-direction pixel offset from PVM_Phase1Offset (mm) and
    the spatial resolution (PVM_FovCm / PVM_DefMatrix), then circularly shifts
    the image along the phase axis (axis 1) before normalising.

    Args:
        images: Float magnitude array, shape [x, y, slices, frames].
        scan:   BrukerScan supplying PVM_FovCm, PVM_DefMatrix, PVM_Phase1Offset.

    Returns:
        int16 array of the same shape, maximum value ≤ 30 000.
    """
    m = scan.method
    fov_cm = float(np.asarray(m["PVM_FovCm"]).ravel()[0])
    offset_mm = float(np.asarray(m["PVM_Phase1Offset"]).ravel()[0])

    # Use the actual image y-size so the offset is correct after any zero-padding.
    resolution_cm = fov_cm / images.shape[1]
    offset_pixels = (offset_mm / 10.0) / resolution_cm

    shifted = np.roll(images, -round(offset_pixels), axis=1)

    max_val = float(np.max(np.abs(shifted)))
    if max_val > 0:
        normalized = (30000.0 / max_val) * shifted
    else:
        normalized = shifted

    return normalized.astype(np.int16)


def shuffle_slices(
    images: npt.NDArray[np.generic],
    pack_slices: list[int] | None = None,
) -> npt.NDArray[np.generic]:
    """Reorder slices from Bruker interleaved acquisition to anatomical order.

    Translates sliceShuffler.m.

    Bruker acquires slices in two passes: first the odd-indexed positions
    (0, 2, 4, …), then the even-indexed positions (1, 3, 5, …).  This function
    inverts that permutation so slice 0 is the most inferior (or anterior)
    position and slice N-1 the most superior.

    For multi-pack acquisitions each pack is interleaved independently, so the
    shuffle is applied within each pack's slice range.  Pass pack_slices as the
    list of per-pack slice counts returned by bruker_to_lps().  When None, all
    slices are treated as a single pack (single-pack backward compatibility).

    Args:
        images:      Array of shape [x, y, slices, frames].
        pack_slices: Number of slices per pack.  Must sum to images.shape[2].

    Returns:
        Array with the same shape, slices in anatomical order within each pack.
    """
    n_total = images.shape[2]
    if n_total <= 1:
        return images

    if pack_slices is None:
        pack_slices = [n_total]

    result = images.copy()
    offset = 0
    for n in pack_slices:
        if n > 1:
            half = (n + 1) // 2  # matches MATLAB round(n/2), rounds .5 up
            order = np.zeros(n, dtype=np.intp)
            order[0::2] = np.arange(half)
            order[1::2] = np.arange(half, n)
            result[:, :, offset : offset + n, :] = images[:, :, offset + order, :]
        offset += n

    return result
