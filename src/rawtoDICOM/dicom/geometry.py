"""Image geometry corrections for DICOM output.

Translates imageCorrections.m, sliceShuffler.m, and orientRotation.m.
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
    matrix = int(np.asarray(m["PVM_DefMatrix"]).ravel()[0])
    offset_mm = float(np.asarray(m["PVM_Phase1Offset"]).ravel()[0])

    resolution_cm = fov_cm / matrix
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
) -> npt.NDArray[np.generic]:
    """Reorder slices from Bruker interleaved acquisition to anatomical order.

    Translates sliceShuffler.m.

    Bruker acquires slices in two passes: first the anatomically odd-indexed
    slices (0, 2, 4, …), then the even-indexed slices (1, 3, 5, …).  This
    function inverts that permutation so that slice 0 is the most inferior
    (or anterior) position and slice N-1 is the most superior.

    Args:
        images: Array of shape [x, y, slices, frames].

    Returns:
        Array with the same shape, slices in anatomical order.
    """
    n_slices = images.shape[2]
    if n_slices <= 1:
        return images

    half = (n_slices + 1) // 2  # matches MATLAB round(n/2), rounds .5 up
    order = np.zeros(n_slices, dtype=np.intp)
    order[0::2] = np.arange(half)
    order[1::2] = np.arange(half, n_slices)

    return images[:, :, order, :]


def orient_rotation(
    slice_images: npt.NDArray[np.generic],
    scan: BrukerScan,
) -> npt.NDArray[np.generic]:
    """Rotate and flip a single-slice image to standard display orientation.

    Translates orientRotation.m.

    The number of 90° CCW rotations (k) and whether to flip the y-axis are
    determined by the combination of PVM_SPackArrReadOrient ('H_F', 'A_P',
    'L_R') and PVM_SPackArrSliceOrient ('sagittal', 'coronal', 'axial').
    Cases not listed below leave the image unchanged (k=0, no flip).

    Args:
        slice_images: Shape [x, y, frames].
        scan:         BrukerScan supplying orientation parameters.

    Returns:
        Rotated/flipped array.  Shape may differ from input when k is odd.
    """
    m = scan.method
    read_orient = str(np.asarray(m["PVM_SPackArrReadOrient"]).ravel()[0]).upper()
    slice_orient = str(np.asarray(m["PVM_SPackArrSliceOrient"]).ravel()[0]).lower()

    k = 0
    flip_y = False

    if "sagittal" in slice_orient:
        if "H_F" in read_orient:
            k = 2
        elif "A_P" in read_orient:
            k = 1
            flip_y = True
    elif "coronal" in slice_orient:
        if "H_F" in read_orient:
            k = 2
        elif "L_R" in read_orient:
            k = 1
            flip_y = True
    elif "axial" in slice_orient:
        if "A_P" in read_orient:
            k = 2
        elif "L_R" in read_orient:
            k = 1
            flip_y = True

    result = np.rot90(slice_images, k, axes=(0, 1))
    if flip_y:
        result = np.flip(result, axis=1)

    return np.asarray(result)


# ---------------------------------------------------------------------------
# Coordinate system note for orient_rotation_from_visu
# ---------------------------------------------------------------------------
# VisuCoreOrientation rows are direction cosines in Bruker scanner space.
# From empirical analysis of test data (Foot_Prone, horizontal bore, PV360):
#   axis 0 = Left/Right  (coronal normal dominant = axis 1; sagittal normal dominant = axis 0)
#   axis 1 = Anterior/Posterior  (positive = Posterior, confirmed by coronal normal +0.941)
#   axis 2 = Head/Foot  (sign to be confirmed — further testing required)
#
# This matches the DICOM LPS patient coordinate system when the subject is
# Head_Prone.  For Foot_Prone subjects the bore z-axis and the patient
# superior axis are anti-parallel, so axis 2 sign may need to be negated.
# The target direction constants below are the best current estimate; adjust
# them if comparison with orient_rotation() reveals a systematic flip.

# Target display: patient Left increases along image x (radiological convention,
# patient right on viewer left).  The sign of axis 0 positive is assumed to be
# patient Right (read vector for L_R scan points mostly in -axis0 direction),
# so patient Left = [-1, 0, 0].
_TARGET_X: npt.NDArray[np.floating] = np.array([-1.0, 0.0, 0.0])

# Target display: head at top means image y increases toward foot.
# Assuming axis 2 positive = Head: foot = [0, 0, -1].
# If axis 2 positive = Foot, change to [0, 0, 1].
_TARGET_Y: npt.NDArray[np.floating] = np.array([0.0, 0.0, -1.0])


def orient_rotation_from_visu(
    slice_images: npt.NDArray[np.generic],
    scan: BrukerScan,
    slice_idx: int = 0,
    target_x: npt.NDArray[np.floating] | None = None,
    target_y: npt.NDArray[np.floating] | None = None,
) -> npt.NDArray[np.generic]:
    """Rotate and flip using VisuCoreOrientation instead of the parameter lookup.

    Alternative to orient_rotation().  Extracts the actual read and phase
    direction vectors from VisuCoreOrientation and scores all eight possible
    90°-increment 2D transforms against a target display convention, selecting
    the one that best aligns the image axes with the desired anatomical
    directions.

    This approach handles oblique planes (e.g. cardiac LAX views) more
    consistently than orient_rotation() because it operates on the actual
    geometry rather than the coarse Bruker plane label, which is assigned to
    the nearest standard plane and can be identical for two LAX views that
    differ by an arbitrary in-plane rotation.

    Coordinate system (see module comment above for derivation):
      axis 0 = L/R,  axis 1 = A/P (positive posterior),  axis 2 = H/F.
    Both the sign of axis 2 and the exact L/R convention need experimental
    confirmation; use target_x and target_y to adjust if needed.

    Transform direction-vector equations (verified against numpy.rot90):
      k=0        : new_x =  x_dir,  new_y =  y_dir
      k=0, flip  : new_x =  x_dir,  new_y = -y_dir
      k=1        : new_x = -y_dir,  new_y =  x_dir   (90 deg CCW)
      k=1, flip  : new_x = -y_dir,  new_y = -x_dir
      k=2        : new_x = -x_dir,  new_y = -y_dir   (180 deg)
      k=2, flip  : new_x = -x_dir,  new_y =  y_dir
      k=3        : new_x =  y_dir,  new_y = -x_dir   (90 deg CW)
      k=3, flip  : new_x =  y_dir,  new_y =  x_dir   (transpose)

    Args:
        slice_images: Shape [x, y, frames].
        scan:         BrukerScan supplying VisuCoreOrientation.
        slice_idx:    Which slice row to read from VisuCoreOrientation (0-based).
        target_x:     Desired image x-axis direction in scanner space.
                      Default: [-1, 0, 0] (patient Left, radiological convention).
        target_y:     Desired image y-axis direction in scanner space.
                      Default: [0, 0, -1] (foot direction, head at top).

    Returns:
        Rotated/flipped array.  Shape may differ from input when k is odd.
    """
    vp = scan.visu_pars
    orientation = np.asarray(vp["VisuCoreOrientation"]).reshape(-1, 9)
    idx = min(slice_idx, len(orientation) - 1)

    x_dir = orientation[idx, 0:3]  # read direction in scanner space
    y_dir = orientation[idx, 3:6]  # phase direction in scanner space

    tx = _TARGET_X if target_x is None else np.asarray(target_x, dtype=float)
    ty = _TARGET_Y if target_y is None else np.asarray(target_y, dtype=float)

    # All 8 possible (k, flip_y) transforms and what they do to image axes.
    candidates: list[tuple[int, bool, npt.NDArray[np.floating], npt.NDArray[np.floating]]] = [
        (0, False,  x_dir,   y_dir),
        (0, True,   x_dir,  -y_dir),
        (1, False, -y_dir,   x_dir),
        (1, True,  -y_dir,  -x_dir),
        (2, False, -x_dir,  -y_dir),
        (2, True,  -x_dir,   y_dir),
        (3, False,  y_dir,  -x_dir),
        (3, True,   y_dir,   x_dir),
    ]

    best_score = -np.inf
    best_k, best_flip = 0, False
    for k, flip, new_x, new_y in candidates:
        score = float(np.dot(new_x, tx) + np.dot(new_y, ty))
        if score > best_score:
            best_score = score
            best_k, best_flip = k, flip

    result = np.rot90(slice_images, best_k, axes=(0, 1))
    if best_flip:
        result = np.flip(result, axis=1)

    return np.asarray(result)
