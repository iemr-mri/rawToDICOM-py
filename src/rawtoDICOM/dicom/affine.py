"""Bruker-to-DICOM affine math and LPS resolution.

Translated from BrkRaw resolver/affine.py.  Covers the full pipeline from
raw Bruker orientation/position parameters to per-slice ImagePositionPatient
and ImageOrientationPatient values, plus the companion pixel-axis corrections.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.scan import BrukerScan


# ---------------------------------------------------------------------------
# Primitive affine helpers
# ---------------------------------------------------------------------------

def from_matvec(
    mat: npt.NDArray[np.floating],
    vec: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    affine: npt.NDArray[np.floating] = np.eye(4)
    affine[:3, :3] = mat
    affine[:3, 3] = vec
    return affine


def to_matvec(
    affine: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    return affine[:3, :3].copy(), affine[:3, 3].copy()


def rotate_affine(
    affine: npt.NDArray[np.floating],
    rad_x: float = 0.0,
    rad_y: float = 0.0,
    rad_z: float = 0.0,
) -> npt.NDArray[np.floating]:
    A = np.asarray(affine, dtype=float)
    cx, sx = np.cos(rad_x), np.sin(rad_x)
    cy, sy = np.cos(rad_y), np.sin(rad_y)
    cz, sz = np.cos(rad_z), np.sin(rad_z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    M, t = to_matvec(A)
    return from_matvec(R @ M, R @ t)


def flip_affine(
    affine: npt.NDArray[np.floating],
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> npt.NDArray[np.floating]:
    A = np.asarray(affine, dtype=float)
    M, t = to_matvec(A)
    F = np.diag([
        -1.0 if flip_x else 1.0,
        -1.0 if flip_y else 1.0,
        -1.0 if flip_z else 1.0,
    ])
    return from_matvec(F @ M, F @ t)


def flip_voxel_axis_affine(
    affine: npt.NDArray[np.floating],
    axis: int,
    shape: tuple[int, ...],
) -> npt.NDArray[np.floating]:
    """Flip a voxel axis in an affine, shifting the origin to keep physical space correct.

    Negates column `axis` of the rotation and shifts the translation by (n-1)
    voxels along the original column direction so the flipped array still maps
    to the same physical volume.  Translated from BrkRaw flip_voxel_axis_affine.
    """
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
    """Convert a Bruker scanner-reference-frame affine to scanner XYZ (→ DICOM LPS).

    Translated from BrkRaw unwrap_to_scanner_xyz.

    Bruker stores VisuCorePosition / VisuCoreOrientation in a subject-relative
    frame.  The exact frame depends on species:
      Quadruped: PV uses LSA+; flip_x converts to RSA+ (scanner view).
      Biped:     PV uses LPS+; flip_y converts to LAS+ (scanner view).
    Foot-first entry additionally applies a 180° Y-rotation before the above.
    Gravity orientation (Prone / Supine / Left / Right) adds a Z-rotation.
    """
    _affine = np.asarray(affine, dtype=float)
    head_or_foot, gravity = subject_pose.split("_", 1)
    subject_type = subject_type or "Biped"

    if head_or_foot == "Foot":
        _affine = rotate_affine(_affine, rad_y=np.pi)

    if subject_type in ("Quadruped", "OtherAnimal"):
        _affine = flip_affine(_affine, flip_x=True)
        if gravity == "Supine":
            _affine = rotate_affine(_affine, rad_z=np.pi)
        elif gravity == "Left":
            _affine = rotate_affine(_affine, rad_z=np.pi / 2)
        elif gravity == "Right":
            _affine = rotate_affine(_affine, rad_z=-np.pi / 2)
    else:  # Biped / Phantom / Other / default
        _affine = flip_affine(_affine, flip_y=True)
        if gravity == "Prone":
            _affine = rotate_affine(_affine, rad_z=np.pi)
        elif gravity == "Left":
            _affine = rotate_affine(_affine, rad_z=-np.pi / 2)
        elif gravity == "Right":
            _affine = rotate_affine(_affine, rad_z=np.pi / 2)

    return _affine


# ---------------------------------------------------------------------------
# Slice-pack resolution
# ---------------------------------------------------------------------------

def _resolve_slice_pack(
    method: dict,  # type: ignore[type-arg]
) -> tuple[int, list[int], list[float], list[str]]:
    """Extract per-pack slice counts, thicknesses, and orientations from method params.

    Returns (n_packs, slices_per_pack, thickness_per_pack, orient_name_per_pack).
    Thickness is slice distance + gap, matching BrkRaw's convention.
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
    # BrkRaw: effective slice thickness = slice distance + gap
    thickness = [thickness[i] + gap[i] for i in range(n_packs)]

    raw_orients = np.asarray(method.get("PVM_SPackArrSliceOrient", ["axial"])).ravel()
    if n_packs == 1:
        orient_names = [str(raw_orients[0]).lower()]
    else:
        orient_names = [
            str(raw_orients[min(i, len(raw_orients) - 1)]).lower() for i in range(n_packs)
        ]

    return n_packs, n_slices, thickness, orient_names


def _build_pack_affine(
    visu_pars: dict,  # type: ignore[type-arg]
    spack_idx: int,
    pack_slice_start: int,
    pack_n_slices: int,
    pack_thickness: float,
    total_slices: int,
) -> npt.NDArray[np.floating]:
    """Build the 4x4 affine for one slice pack.

    Mirrors BrkRaw resolve_matvec_and_shape for the 2D multi-slice case:
    - Extracts the pack's orientation and position entries from the Visu arrays.
    - Uses minimum-projection origin selection for multi-slice packs so the
      origin is the slice with the smallest projection onto the slice normal,
      which gives a consistent anatomical ordering regardless of acquisition order.
    - Scales each rotation column by voxel resolution (extent / size).
    """
    raw_orient = np.asarray(
        visu_pars.get("VisuCoreOrientation", np.eye(3).ravel())
    ).reshape(-1, 9).astype(float)
    raw_pos = np.asarray(
        visu_pars.get("VisuCorePosition", np.zeros((1, 3)))
    ).reshape(-1, 3).astype(float)
    extent = np.asarray(visu_pars.get("VisuCoreExtent", [1.0, 1.0])).ravel().astype(float)
    core_size = np.asarray(visu_pars.get("VisuCoreSize", [1, 1])).ravel().astype(float)

    pack_end = pack_slice_start + pack_n_slices

    # Select orientation entries for this pack (prefer per-slice; fall back to per-pack)
    if raw_orient.shape[0] >= total_slices:
        pack_orient = raw_orient[pack_slice_start:pack_end]
    elif raw_orient.shape[0] == 1:
        pack_orient = np.tile(raw_orient[0], (pack_n_slices, 1))
    else:
        row = raw_orient[min(spack_idx, len(raw_orient) - 1)]
        pack_orient = np.tile(row, (pack_n_slices, 1))

    # Select position entries for this pack
    if raw_pos.shape[0] >= total_slices:
        pack_pos = raw_pos[pack_slice_start:pack_end]
    elif raw_pos.shape[0] == 1:
        pack_pos = np.tile(raw_pos[0], (pack_n_slices, 1))
    else:
        pack_pos = np.tile(raw_pos[min(spack_idx, len(raw_pos) - 1)], (pack_n_slices, 1))

    row_dir = pack_orient[0, 0:3]
    col_dir = pack_orient[0, 3:6]
    slc_dir = pack_orient[0, 6:9]

    # BrkRaw minimum-projection origin: the slice whose position projects least
    # along the slice normal is the anatomical "first" slice and becomes the origin.
    if pack_n_slices > 1:
        n = slc_dir / np.linalg.norm(slc_dir)
        s = pack_pos @ n
        origin = pack_pos[int(np.argmin(s))]
    else:
        origin = pack_pos[0]

    # Voxel resolutions: in-plane from VisuCoreExtent / VisuCoreSize; through-plane
    # from the effective slice thickness supplied by the caller.
    full_extent = np.array([extent[0], extent[1], pack_n_slices * pack_thickness])
    full_shape = np.array([core_size[0], core_size[1], float(pack_n_slices)])
    resols = full_extent / full_shape

    rot = np.column_stack([row_dir, col_dir, slc_dir])
    mat = rot * resols.reshape(1, 3)
    return from_matvec(mat, origin)


# ---------------------------------------------------------------------------
# Public API: affine-based LPS resolution and pixel corrections
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

    Follows BrkRaw resolve() approach for each slice pack:
    1. Parse PVM_NSPacks / PVM_SPackArrNSlices for per-pack slice counts.
    2. Build a 4x4 affine from VisuCoreOrientation / VisuCorePosition /
       VisuCoreExtent / VisuCoreSize, with minimum-projection origin selection.
    3. Phase flip (ACQ_scaling_phase < 0): flip_voxel_axis_affine(axis=1).
    4. Coronal flip (PVM_SPackArrSliceOrient == "coronal"): flip_voxel_axis_affine(axis=2).
    5. Subject-type/position transform via unwrap_to_scanner_xyz (quadruped-aware).
    6. Expand each pack's affine to per-slice ImagePositionPatient /
       ImageOrientationPatient by stepping along the slice column.

    Args:
        scan: BrukerScan supplying visu_pars, method, and acqp.

    Returns:
        positions:       float64 [total_slices, 3] — ImagePositionPatient per slice.
        orientations:    float64 [total_slices, 6] — ImageOrientationPatient per slice
                         (row-direction cosines then column-direction cosines).
        pack_slices:     slices per pack, e.g. [8] or [4, 4].
        pack_is_coronal: whether each pack's slice orientation is coronal.
    """
    vp = scan.visu_pars
    m = scan.method

    n_packs, pack_n_slices, pack_thickness, orient_names = _resolve_slice_pack(m)
    total_slices = sum(pack_n_slices)

    phase_dir = float(np.asarray(scan.acqp.get("ACQ_scaling_phase", 1.0)).ravel()[0])
    flip_phase = phase_dir < 0

    core_size = np.asarray(vp.get("VisuCoreSize", [1, 1])).ravel()
    n_read = int(core_size[0]) if len(core_size) > 0 else 1
    n_phase = int(core_size[1]) if len(core_size) > 1 else 1

    subj_type_raw = vp.get("VisuSubjectType", None)
    subj_type = str(subj_type_raw) if subj_type_raw is not None else None
    subj_pos_raw = vp.get("VisuSubjectPosition", "Head_Prone")
    subj_pos = str(np.asarray(subj_pos_raw).ravel()[0])

    all_positions: list[npt.NDArray[np.floating]] = []
    all_orientations: list[npt.NDArray[np.floating]] = []
    pack_is_coronal: list[bool] = []

    pack_slice_start = 0
    for i in range(n_packs):
        n_slices_i = pack_n_slices[i]
        is_coronal = "coronal" in orient_names[i]
        pack_is_coronal.append(is_coronal)

        affine = _build_pack_affine(
            vp,
            spack_idx=i,
            pack_slice_start=pack_slice_start,
            pack_n_slices=n_slices_i,
            pack_thickness=pack_thickness[i],
            total_slices=total_slices,
        )

        pack_shape = (n_read, n_phase, n_slices_i)

        if flip_phase:
            affine = flip_voxel_axis_affine(affine, axis=1, shape=pack_shape)
        if is_coronal and n_slices_i > 1:
            affine = flip_voxel_axis_affine(affine, axis=2, shape=pack_shape)

        affine = unwrap_to_scanner_xyz(affine, subj_type, subj_pos)
        affine = np.round(affine, decimals=6)

        # Expand to per-slice: position of slice j = origin + j * slice_step
        slice_step = affine[:3, 2]
        origin = affine[:3, 3]
        row_unit = affine[:3, 0] / np.linalg.norm(affine[:3, 0])
        col_unit = affine[:3, 1] / np.linalg.norm(affine[:3, 1])
        orientation_vec: npt.NDArray[np.floating] = np.concatenate([row_unit, col_unit])

        for j in range(n_slices_i):
            all_positions.append(origin + j * slice_step)
            all_orientations.append(orientation_vec)

        pack_slice_start += n_slices_i

    positions = np.array(all_positions, dtype=float)
    orientations = np.array(all_orientations, dtype=float)

    return positions, orientations, pack_n_slices, pack_is_coronal


def orient_correction_brkraw(
    images: npt.NDArray[np.generic],
    scan: BrukerScan,
    pack_slices: list[int] | None = None,
    pack_is_coronal: list[bool] | None = None,
) -> npt.NDArray[np.generic]:
    """Apply pixel-axis corrections that match the voxel flips encoded in bruker_to_lps().

    bruker_to_lps() calls flip_voxel_axis_affine() for two cases — phase and
    coronal — which redefine which voxel is "first" along those axes.  The
    pixel array must be reordered to stay consistent with the updated metadata.

    unwrap_to_scanner_xyz() (subject-position/type transform) is a pure
    coordinate-frame conversion and does NOT change voxel ordering, so no
    pixel flip is applied for it here.

    Args:
        images:          Float array [x, y, slices, frames].
        scan:            BrukerScan supplying acqp, method, visu_pars.
        pack_slices:     Per-pack slice counts from bruker_to_lps().
        pack_is_coronal: Per-pack coronal flag from bruker_to_lps().

    Returns:
        Array of the same shape and dtype with axes flipped as required.
    """
    result = np.array(images)

    # 1. Phase flip — mirrors flip_voxel_axis_affine(axis=1) in bruker_to_lps()
    phase_scaling = float(np.asarray(scan.acqp.get("ACQ_scaling_phase", 1.0)).ravel()[0])
    if phase_scaling < 0:
        result = np.flip(result, axis=1)

    # 2. Coronal flip — mirrors flip_voxel_axis_affine(axis=2) in bruker_to_lps()
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
