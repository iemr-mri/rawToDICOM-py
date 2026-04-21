"""Bruker-to-DICOM affine math and LPS resolution.

Faithful translation of BrkRaw resolver/affine.py, adapted for our pipeline
I/O (BrukerScan dataclass instead of BrkRaw Scan/Reco nodes).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.scan import BrukerScan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive affine helpers  (mirrors BrkRaw 1:1)
# ---------------------------------------------------------------------------

def from_matvec(
    mat: npt.NDArray[np.floating],
    vec: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    if mat.shape == (3, 3) and vec.shape == (3,):
        affine: npt.NDArray[np.floating] = np.eye(4)
        affine[:3, :3] = mat
        affine[:3, 3] = vec
        return affine
    raise ValueError("Matrix must be 3×3 and vector must be length 3")


def to_matvec(
    affine: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    if np.asarray(affine).shape != (4, 4):
        raise ValueError("Affine matrix must be 4×4")
    return affine[:3, :3].copy(), affine[:3, 3].copy()


def rotate_affine(
    affine: npt.NDArray[np.floating],
    rad_x: float = 0.0,
    rad_y: float = 0.0,
    rad_z: float = 0.0,
    pivot: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.floating]:
    """Rotate a 4×4 affine; optionally around a world-space pivot point."""
    A = np.asarray(affine, dtype=float)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rad_x), -np.sin(rad_x)],
                   [0, np.sin(rad_x),  np.cos(rad_x)]])
    Ry = np.array([[ np.cos(rad_y), 0, np.sin(rad_y)],
                   [0,              1, 0],
                   [-np.sin(rad_y), 0, np.cos(rad_y)]])
    Rz = np.array([[np.cos(rad_z), -np.sin(rad_z), 0],
                   [np.sin(rad_z),  np.cos(rad_z), 0],
                   [0,              0,             1]])
    R = Rz @ Ry @ Rx
    M, t = to_matvec(A)
    if pivot is None:
        t_new = R @ t
    else:
        p = np.asarray(pivot, dtype=float).reshape(3)
        t_new = R @ t + (p - R @ p)
    return from_matvec(R @ M, t_new)


def flip_affine(
    affine: npt.NDArray[np.floating],
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    pivot: Optional[npt.NDArray[np.floating]] = None,
) -> npt.NDArray[np.floating]:
    """Flip world axes of an affine; optionally around a world-space pivot point."""
    A = np.asarray(affine, dtype=float)
    M, t = to_matvec(A)
    F = np.diag([-1.0 if flip_x else 1.0,
                 -1.0 if flip_y else 1.0,
                 -1.0 if flip_z else 1.0])
    if pivot is None:
        t_new = F @ t
    else:
        p = np.asarray(pivot, dtype=float).reshape(3)
        t_new = F @ t + (p - F @ p)
    return from_matvec(F @ M, t_new)


def flip_voxel_axis_affine(
    affine: npt.NDArray[np.floating],
    axis: int,
    shape: tuple[int, ...],
) -> npt.NDArray[np.floating]:
    """Flip a voxel axis, shifting origin by (n-1) voxels to preserve physical space."""
    A = np.asarray(affine, float)
    M = A[:3, :3].copy()
    t = A[:3, 3].copy()
    n = int(shape[axis])
    if n <= 1:
        return A.copy()
    col = M[:, axis].copy()
    M[:, axis] = -col
    t = t + col * (n - 1)
    out = np.eye(4)
    out[:3, :3] = M
    out[:3, 3] = t
    return out


def unwrap_to_scanner_xyz(
    affine: npt.NDArray[np.floating],
    subject_type: Optional[str],
    subject_pose: str,
) -> npt.NDArray[np.floating]:
    """Convert Bruker subject-frame affine to scanner XYZ (→ DICOM LPS).

    Only Biped and Quadruped receive pose-specific transforms; all other subject
    types pass through unchanged (matches BrkRaw behaviour).
    """
    _affine = np.asarray(affine, dtype=float)
    head_or_foot, gravity = subject_pose.split("_", 1)
    subject_type = subject_type or "Biped"  # PV5.1 backward compatibility

    if head_or_foot == "Foot":
        _affine = rotate_affine(_affine, rad_y=np.pi)

    if subject_type == "Biped":
        # PV stores affine in LPS+; scanner coord is LAS+ → flip Y
        _affine = flip_affine(_affine, flip_y=True)
        if gravity == "Prone":
            _affine = rotate_affine(_affine, rad_z=np.pi)
        elif gravity == "Left":
            _affine = rotate_affine(_affine, rad_z=-np.pi / 2)
        elif gravity == "Right":
            _affine = rotate_affine(_affine, rad_z=np.pi / 2)

    elif subject_type == "Quadruped":
        # PV uses LSA+; scanner is RSA+ → flip X
        _affine = flip_affine(_affine, flip_x=True)
        if gravity == "Supine":
            _affine = rotate_affine(_affine, rad_z=np.pi)
        elif gravity == "Left":
            _affine = rotate_affine(_affine, rad_z=np.pi / 2)
        elif gravity == "Right":
            _affine = rotate_affine(_affine, rad_z=-np.pi / 2)

    return _affine


# ---------------------------------------------------------------------------
# Slice-pack resolution  (mirrors BrkRaw resolve_slice_pack / resolve_matvec_and_shape)
# ---------------------------------------------------------------------------

def _resolve_slice_pack(
    method: dict,  # type: ignore[type-arg]
) -> tuple[int, list[int], list[float]]:
    """Parse per-pack slice counts and effective thicknesses from method params.

    Returns (n_packs, slices_per_pack, effective_thickness_per_pack).
    effective_thickness = PVM_SPackArrSliceDistance + PVM_SPackArrSliceGap.
    """
    n_packs = int(np.asarray(method.get("PVM_NSPacks", 1)).ravel()[0])
    n_slices = list(
        np.asarray(method.get("PVM_SPackArrNSlices", [1])).ravel()[:n_packs].astype(int)
    )
    thickness = list(
        np.asarray(method.get("PVM_SPackArrSliceDistance", [1.0])).ravel()[:n_packs].astype(float)
    )
    gap = list(
        np.asarray(method.get("PVM_SPackArrSliceGap", [0.0])).ravel()[:n_packs].astype(float)
    )
    effective = [thickness[i] + gap[i] for i in range(n_packs)]
    return n_packs, n_slices, effective


def _resolve_matvec_and_shape(
    visu_pars: dict,  # type: ignore[type-arg]
    spack_idx: int,
    num_slices: list[int],
    slice_thickness: list[float],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], tuple[int, ...]]:
    """Build (mat, vec, shape) for one slice pack.

    Direct translation of BrkRaw resolve_matvec_and_shape for dim==2.
    Raises ValueError when orientation/position arrays cannot unambiguously
    supply per-slice entries for a multi-slice pack.
    """
    rotate = np.asarray(visu_pars.get("VisuCoreOrientation", np.eye(3).ravel()), dtype=float)
    origin = np.asarray(visu_pars.get("VisuCorePosition", np.zeros(3)), dtype=float)
    extent = np.asarray(visu_pars.get("VisuCoreExtent", [1.0, 1.0]), dtype=float).ravel()
    shape  = np.asarray(visu_pars.get("VisuCoreSize", [1, 1]), dtype=float).ravel()

    num_slice_packs = len(num_slices)
    total_slices = int(np.sum(np.asarray(num_slices, dtype=int)))
    spack_slice_start = int(np.sum(np.asarray(num_slices[:spack_idx], dtype=int)))
    spack_slice_end = spack_slice_start + int(num_slices[spack_idx])

    def _select_slice_entries(
        arr: npt.NDArray[np.floating], width: int, name: str
    ) -> npt.NDArray[np.floating]:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            if arr.size == width:
                arr = arr.reshape((1, width))
            else:
                raise ValueError(f"{name} has shape {arr.shape}, expected (*, {width})")
        if arr.ndim != 2 or arr.shape[1] != width:
            raise ValueError(f"{name} has shape {arr.shape}, expected (*, {width})")

        if arr.shape[0] > total_slices:
            if not np.allclose(arr[:total_slices], arr[0], atol=0, rtol=0):
                logger.warning(
                    "%s has %d entries but expected %d; using the first %d.",
                    name, arr.shape[0], total_slices, total_slices,
                )
            arr = arr[:total_slices, :]

        if arr.shape[0] == total_slices:
            return arr[spack_slice_start:spack_slice_end, :]

        # Per-pack fallback: only valid for single-slice packs
        if arr.shape[0] == num_slice_packs:
            if int(num_slices[spack_idx]) != 1:
                raise ValueError(
                    f"{name} provides one entry per slice pack ({num_slice_packs}) "
                    f"but pack {spack_idx} has {num_slices[spack_idx]} slices; "
                    "per-slice entries are required."
                )
            return arr[spack_idx : spack_idx + 1, :]

        raise ValueError(
            f"{name} has {arr.shape[0]} entries, expected {total_slices} (per-slice) "
            f"or {num_slice_packs} (per-pack); num_slices={num_slices}."
        )

    _rotate = _select_slice_entries(rotate.reshape(-1, 9), 9, "VisuCoreOrientation")
    _origin = _select_slice_entries(origin.reshape(-1, 3), 3, "VisuCorePosition")
    _num_slices = num_slices[spack_idx]
    _slice_thickness = slice_thickness[spack_idx]

    if _rotate.shape[0] > 1 and not np.allclose(_rotate, _rotate[0], atol=0, rtol=0):
        logger.warning(
            "VisuCoreOrientation varies across slices in pack %d; using first slice.",
            spack_idx,
        )

    row = _rotate[0, 0:3]
    col = _rotate[0, 3:6]
    slc = _rotate[0, 6:9]

    n = slc / np.linalg.norm(slc)
    if _num_slices > 1:
        s = _origin @ n
        vec: npt.NDArray[np.floating] = _origin[int(np.argmin(s))]
    else:
        vec = _origin[0]

    shape_3d = np.append(shape, _num_slices)
    extent_3d = np.append(extent, _num_slices * _slice_thickness)
    resols = extent_3d / shape_3d
    rot = np.column_stack([row, col, slc])
    mat = rot * resols.reshape(1, 3)

    return mat, vec, tuple(shape_3d.astype(int).tolist())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bruker_to_lps(
    scan: BrukerScan,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    list[int],
    list[bool],
]:
    """Build DICOM position/orientation metadata from Bruker parameters.

    Mirrors BrkRaw resolve() with unwrap_pose=True, expanded to per-slice arrays
    for DICOM output.

    Steps per slice pack:
      1. _resolve_matvec_and_shape — orientation + minimum-projection origin.
      2. phase flip (ACQ_scaling_phase < 0) — flip_voxel_axis_affine(axis=1):
         shifts origin by (n_phase-1)*voxel_phase to keep it at the correct
         physical end of the phase-encode direction.
      3. coronal flip — flip_voxel_axis_affine(axis=2): reverses slice ordering
         within coronal packs for consistent anatomical ordering.
      4. unwrap_to_scanner_xyz — subject-type/position → scanner LPS frame.
      5. expand to per-slice ImagePositionPatient / ImageOrientationPatient.

    Returns:
        positions:       float64 [total_slices, 3] — ImagePositionPatient.
        orientations:    float64 [total_slices, 6] — ImageOrientationPatient
                         (row cosines then column cosines).
        pack_slices:     slices per pack.
        pack_is_coronal: whether each pack's orientation is coronal.
    """
    vp = scan.visu_pars
    m = scan.method

    n_packs, pack_n_slices, pack_thickness = _resolve_slice_pack(m)

    # BrkRaw contingency: when a single pack reports 1 slice but VisuCoreSize
    # encodes more (e.g. 3D-encoded acquisitions), trust VisuCoreSize.
    if n_packs == 1 and pack_n_slices[0] == 1:
        core_size = np.asarray(vp.get("VisuCoreSize", [1, 1])).ravel()
        if len(core_size) >= 3 and int(core_size[2]) != 1:
            pack_n_slices = [int(core_size[2])]

    phase_dir = float(np.asarray(scan.acqp.get("ACQ_scaling_phase", 1.0)).ravel()[0])
    flip_phase = phase_dir < 0

    raw_slice_orient = np.asarray(m.get("PVM_SPackArrSliceOrient", ["axial"])).ravel()

    subj_type_raw = vp.get("VisuSubjectType", None)
    subj_type = str(subj_type_raw) if subj_type_raw is not None else None
    subj_pos_raw = vp.get("VisuSubjectPosition", "Head_Prone")
    subj_pos = str(np.asarray(subj_pos_raw).ravel()[0])

    all_positions: list[npt.NDArray[np.floating]] = []
    all_orientations: list[npt.NDArray[np.floating]] = []
    pack_is_coronal: list[bool] = []

    for i in range(n_packs):
        if n_packs == 1:
            orient_name = str(raw_slice_orient[0]).lower()
        else:
            orient_name = str(raw_slice_orient[min(i, len(raw_slice_orient) - 1)]).lower()

        is_coronal = "coronal" in orient_name
        pack_is_coronal.append(is_coronal)

        mat, vec, shape = _resolve_matvec_and_shape(vp, i, pack_n_slices, pack_thickness)
        affine = from_matvec(mat, vec)

        if flip_phase:
            affine = flip_voxel_axis_affine(affine, axis=1, shape=shape)
        if is_coronal and shape[2] > 1:
            affine = flip_voxel_axis_affine(affine, axis=2, shape=shape)

        affine = unwrap_to_scanner_xyz(affine, subj_type, subj_pos)
        affine = np.round(affine, decimals=6)

        slice_step = affine[:3, 2]
        origin = affine[:3, 3]
        row_unit = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
        col_unit = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
        orientation_vec: npt.NDArray[np.floating] = np.concatenate([row_unit, col_unit])

        for j in range(pack_n_slices[i]):
            all_positions.append(origin + j * slice_step)
            all_orientations.append(orientation_vec)

    positions = np.array(all_positions, dtype=float)
    orientations = np.array(all_orientations, dtype=float)
    return positions, orientations, pack_n_slices, pack_is_coronal


def orient_correction_brkraw(
    images: npt.NDArray[np.generic],
    scan: BrukerScan,
    pack_slices: list[int] | None = None,
    pack_is_coronal: list[bool] | None = None,
) -> npt.NDArray[np.generic]:
    """Apply pixel-axis flips that match the affine flips in bruker_to_lps().

    bruker_to_lps() calls flip_voxel_axis_affine() for phase and coronal cases,
    which shifts the affine origin.  The pixel array must be reordered to match
    so that voxel (0,0,0) maps to the updated origin.

    unwrap_to_scanner_xyz() is a pure coordinate-frame transform and does not
    change which voxel is first, so no pixel flip is needed for that step.
    """
    result = np.array(images)

    phase_scaling = float(np.asarray(scan.acqp.get("ACQ_scaling_phase", 1.0)).ravel()[0])
    if phase_scaling < 0:
        result = np.flip(result, axis=1)

    if pack_slices is not None and pack_is_coronal is not None:
        offset = 0
        for n, is_coronal in zip(pack_slices, pack_is_coronal):
            if is_coronal and n > 1:
                slices = result[:, :, offset : offset + n, :].copy()
                result[:, :, offset : offset + n, :] = slices[:, :, ::-1, :]
            offset += n
    else:
        raw_so = np.asarray(scan.method.get("PVM_SPackArrSliceOrient", "")).ravel()[0]
        if "coronal" in str(raw_so).lower() and result.shape[2] > 1:
            result = np.flip(result, axis=2)

    return np.asarray(result)
