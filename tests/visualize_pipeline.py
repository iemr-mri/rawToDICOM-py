"""Pipeline visualization script.

Figure 1: Middle SAX slice reconstruction stages (undersampled k-space →
           CS-reconstructed k-space → final DICOM-ready image).

Figure 2: LAX4 image with lines marking basal, mid, and apical SAX slice
           positions for co-registration verification.

Subject: AGORA2_F1 (scan 10 = CINE_LAX4, scans 15-29 = segFLASH_CS SAX stack).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.dicom.geometry import apply_corrections, orient_rotation, shuffle_slices
from rawtoDICOM.reconstruction.coil_combination import combine_coils
from rawtoDICOM.reconstruction.compressed_sensing import reconstruct_cs
from rawtoDICOM.reconstruction.kspace import ifft2c, sort_kspace, zero_fill_kspace

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SUBJECT_DIR = (
    Path(__file__).parent
    / "raw-data/AGORA/cohort1/AGORA2_F1_s_2025121703_1_4_20251217_103442"
)
LAX4_SCAN_DIR = SUBJECT_DIR / "10"
# Scans 15-29 are individual SAX slices (one scan per slice, 15 total).
SAX_SCAN_DIRS = sorted(
    [SUBJECT_DIR / str(n) for n in range(15, 30)],
    key=lambda p: int(p.name),
)
N_SAX = len(SAX_SCAN_DIRS)  # 15


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


def reconstruct_scan(scan_dir: Path) -> npt.NDArray[np.floating]:
    """Run the full CINE pipeline and return [x, y, slices, frames]."""
    scan = load_scan(scan_dir)
    kspace = sort_kspace(scan)
    kspace_squeezed = kspace[:, :, :, :, 0, :]  # drop flow_enc axis
    if "CSPhaseEncList" in scan.method:
        kspace_squeezed = reconstruct_cs(kspace_squeezed)
    kspace_padded = zero_fill_kspace(kspace_squeezed)
    return combine_coils(kspace_padded)  # [x, y, slices, frames]


def to_display(images: npt.NDArray[np.floating], scan_dir: Path) -> npt.NDArray[np.int16]:
    """Apply phase correction, slice shuffle, and orientation rotation.

    Returns int16 array [x, y, slices, frames] in display orientation.
    """
    scan = load_scan(scan_dir, read_raw=False)
    corrected = apply_corrections(images, scan)
    shuffled = shuffle_slices(corrected)
    n_slices = shuffled.shape[2]
    rotated_slices = [
        orient_rotation(shuffled[:, :, s, :], scan) for s in range(n_slices)
    ]
    result = np.stack(rotated_slices, axis=2)
    return result  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Figure 1: CS reconstruction stages
# ---------------------------------------------------------------------------


def build_figure1() -> plt.Figure:
    """Show middle SAX slice at four reconstruction stages."""
    mid_idx = N_SAX // 2  # slice 7 (0-indexed) → scan directory 22
    scan_dir = SAX_SCAN_DIRS[mid_idx]
    scan = load_scan(scan_dir)

    kspace_raw = sort_kspace(scan)
    kspace_squeezed = kspace_raw[:, :, :, :, 0, :]  # [x, y, 1, frames, coils]

    # Use coil 0 for k-space display
    kspace_display = kspace_squeezed[:, :, 0, 0, 0]  # [x, y], first frame, first coil

    # CS-reconstructed k-space
    kspace_cs = reconstruct_cs(kspace_squeezed)
    kspace_cs_display = kspace_cs[:, :, 0, 0, 0]

    # Final image after zero-fill + coil combination (frame 0)
    kspace_padded = zero_fill_kspace(kspace_cs)
    images = combine_coils(kspace_padded)  # [x, y, 1, frames]
    image_raw = images[:, :, 0, 0]

    # DICOM-ready: corrections + orientation
    display = to_display(images, scan_dir)
    image_dicom = display[:, :, 0, 0].astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"SAX slice {mid_idx + 1}/{N_SAX} — reconstruction stages", fontsize=13)

    def log_kspace(k: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.floating]:
        return np.log1p(np.abs(k))

    axes[0].imshow(log_kspace(kspace_display).T, cmap="gray", origin="lower")
    axes[0].set_title("Undersampled k-space\n(log magnitude)")

    axes[1].imshow(log_kspace(kspace_cs_display).T, cmap="gray", origin="lower")
    axes[1].set_title("CS-reconstructed k-space\n(log magnitude)")

    axes[2].imshow(image_raw.T, cmap="gray", origin="lower")
    axes[2].set_title("Reconstructed image\n(zero-filled + coil combined)")

    axes[3].imshow(image_dicom.T, cmap="gray", origin="lower")
    axes[3].set_title("DICOM-ready\n(phase-corrected + rotated)")

    for ax in axes:
        ax.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Co-registration
# ---------------------------------------------------------------------------


def _get_geometry(scan_dir: Path) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Return (center_mm, row_dir, col_dir) in scanner space from visu_pars."""
    scan = load_scan(scan_dir, read_raw=False)
    vp = scan.visu_pars
    orientation = np.asarray(vp["VisuCoreOrientation"]).reshape(-1, 9)
    position = np.asarray(vp["VisuCorePosition"]).reshape(-1, 3)
    center = position[0]
    row_dir = orientation[0, 0:3]
    col_dir = orientation[0, 3:6]
    return center, row_dir, col_dir


def _sax_line_on_lax(
    lax_center: npt.NDArray[np.floating],
    lax_row: npt.NDArray[np.floating],
    lax_col: npt.NDArray[np.floating],
    lax_fov: tuple[float, float],
    lax_pixels: tuple[int, int],
    sax_center: npt.NDArray[np.floating],
    sax_row: npt.NDArray[np.floating],
    sax_col: npt.NDArray[np.floating],
) -> tuple[float, float, float, float] | None:
    """Compute the intersection of the SAX plane with the LAX4 image plane.

    Returns (x1, y1, x2, y2) in PRE-rotation LAX4 pixel coordinates, or None
    if the planes are parallel (no intersection).
    """
    sax_normal = np.cross(sax_row, sax_col)
    sax_normal = sax_normal / (np.linalg.norm(sax_normal) + 1e-12)

    # Intersection line: A*u + B*v + C = 0  (u, v in mm from lax_center)
    a_coef = float(np.dot(lax_row, sax_normal))
    b_coef = float(np.dot(lax_col, sax_normal))
    c_coef = float(np.dot(lax_center - sax_center, sax_normal))

    fov_x, fov_y = lax_fov
    nx, ny = lax_pixels
    half_x = fov_x / 2.0
    half_y = fov_y / 2.0

    # Collect candidate (u, v) pairs by intersecting with the four FOV edges.
    candidates: list[tuple[float, float]] = []

    if abs(a_coef) > 1e-8:
        for v in (-half_y, half_y):
            u = -(b_coef * v + c_coef) / a_coef
            if -half_x <= u <= half_x:
                candidates.append((u, v))

    if abs(b_coef) > 1e-8:
        for u in (-half_x, half_x):
            v = -(a_coef * u + c_coef) / b_coef
            if -half_y <= v <= half_y:
                candidates.append((u, v))

    # Deduplicate near-identical points
    unique: list[tuple[float, float]] = []
    for pt in candidates:
        if not any(abs(pt[0] - q[0]) < 0.01 and abs(pt[1] - q[1]) < 0.01 for q in unique):
            unique.append(pt)

    if len(unique) < 2:
        return None

    # Convert mm → pixel (pre-rotation image)
    def mm_to_pixel(u: float, v: float) -> tuple[float, float]:
        i = (u / fov_x) * nx + nx / 2.0
        j = (v / fov_y) * ny + ny / 2.0
        return i, j

    (i1, j1) = mm_to_pixel(*unique[0])
    (i2, j2) = mm_to_pixel(*unique[1])
    return i1, j1, i2, j2


def _apply_rotation_to_line(
    line: tuple[float, float, float, float],
    nx: int,
    ny: int,
    scan_dir: Path,
) -> tuple[float, float, float, float]:
    """Apply the same orient_rotation transform to line pixel coordinates.

    orient_rotation performs: rot90(k, axes=(0,1)) then optionally flip(axis=1).

    For a pixel (i, j) in an (nx, ny) image:
      k=0           : (i,  j)
      k=1           : (ny-1-j, i)     → result shape (ny, nx)
      k=2           : (nx-1-i, ny-1-j)
      k=3           : (j, nx-1-i)     → result shape (ny, nx)
    Followed by flip axis=1 on the rotated array.
    """
    scan = load_scan(scan_dir, read_raw=False)
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

    i1, j1, i2, j2 = line

    def transform(i: float, j: float) -> tuple[float, float]:
        if k == 0:
            ri, rj = i, j
            flip_size = ny  # rotated shape (nx, ny), axis-1 size = ny
        elif k == 1:
            ri, rj = ny - 1 - j, i
            flip_size = nx  # rotated shape (ny, nx), axis-1 size = nx
        elif k == 2:
            ri, rj = nx - 1 - i, ny - 1 - j
            flip_size = ny  # rotated shape (nx, ny), axis-1 size = ny
        else:  # k == 3
            ri, rj = j, nx - 1 - i
            flip_size = nx  # rotated shape (ny, nx), axis-1 size = nx
        if flip_y:
            rj = flip_size - 1 - rj
        return ri, rj

    r1 = transform(i1, j1)
    r2 = transform(i2, j2)
    return r1[0], r1[1], r2[0], r2[1]


def _select_visible_sax_slices(
    lax_center: npt.NDArray[np.floating],
    lax_row: npt.NDArray[np.floating],
    lax_col: npt.NDArray[np.floating],
    lax_fov: tuple[float, float],
    lax_pixels: tuple[int, int],
) -> list[Path]:
    """Return SAX scan dirs that intersect the LAX4 FOV, sorted by position."""
    visible = []
    for scan_dir in SAX_SCAN_DIRS:
        sax_center, sax_row, sax_col = _get_geometry(scan_dir)
        line = _sax_line_on_lax(
            lax_center, lax_row, lax_col, lax_fov, lax_pixels,
            sax_center, sax_row, sax_col,
        )
        if line is not None:
            visible.append(scan_dir)
    return visible


def build_figure2() -> plt.Figure:
    """LAX4 with basal, mid, and apical SAX slice lines overlaid.

    All SAX slices that intersect the LAX4 FOV are drawn as thin gray lines.
    The basal, mid, and apical selections (chosen from the visible slices) are
    highlighted with distinct colours and shown individually in the right panels.
    """
    # Reconstruct LAX4 and apply display corrections
    lax_images = reconstruct_scan(LAX4_SCAN_DIR)  # [x, y, 1, frames]
    lax_display = to_display(lax_images, LAX4_SCAN_DIR)
    lax_frame0 = lax_display[:, :, 0, 0].astype(float)

    # LAX4 geometry (pre-rotation)
    lax_scan = load_scan(LAX4_SCAN_DIR, read_raw=False)
    lax_center, lax_row, lax_col = _get_geometry(LAX4_SCAN_DIR)
    lax_vp = lax_scan.visu_pars
    lax_extent = np.asarray(lax_vp["VisuCoreExtent"]).ravel()
    lax_size_raw = np.asarray(lax_vp["VisuCoreSize"]).ravel()
    # Image is zero-filled to 2× in each direction
    lax_nx = int(lax_size_raw[0]) * 2
    lax_ny = int(lax_size_raw[1]) * 2
    lax_fov = (float(lax_extent[0]), float(lax_extent[1]))
    lax_pixels = (lax_nx, lax_ny)

    # Find which SAX slices lie within the LAX4 FOV.
    visible_sax = _select_visible_sax_slices(lax_center, lax_row, lax_col, lax_fov, lax_pixels)
    n_vis = len(visible_sax)
    if n_vis == 0:
        raise RuntimeError("No SAX slices intersect the LAX4 FOV — check geometry.")

    # Pick basal / mid / apical from the visible slices.
    selected: dict[str, Path] = {
        "basal": visible_sax[0],
        "mid": visible_sax[n_vis // 2],
        "apical": visible_sax[-1],
    }
    colors = {"basal": "#ff6600", "mid": "#ffdd00", "apical": "#00ccff"}

    # Reconstruct the three selected SAX slices
    sax_images_map: dict[str, npt.NDArray[np.floating]] = {}
    for label, scan_dir in selected.items():
        imgs = reconstruct_scan(scan_dir)
        disp = to_display(imgs, scan_dir)
        sax_images_map[label] = disp[:, :, 0, 0].astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"LAX4 with SAX slice positions  |  Basal / Mid / Apical"
        f"  ({n_vis}/{N_SAX} slices visible in LAX4 FOV)",
        fontsize=11,
    )

    # LAX4 panel — draw all visible SAX lines, highlight the three selected ones
    ax_lax = axes[0]
    ax_lax.imshow(lax_frame0.T, cmap="gray", origin="lower")
    ax_lax.set_title("LAX4")
    ax_lax.axis("off")

    selected_dirs = set(selected.values())
    for scan_dir in visible_sax:
        sax_center, sax_row, sax_col = _get_geometry(scan_dir)
        line = _sax_line_on_lax(
            lax_center, lax_row, lax_col, lax_fov, lax_pixels,
            sax_center, sax_row, sax_col,
        )
        if line is None:
            continue
        line_rot = _apply_rotation_to_line(line, lax_nx, lax_ny, LAX4_SCAN_DIR)
        i1, j1, i2, j2 = line_rot
        if scan_dir in selected_dirs:
            label = next(k for k, v in selected.items() if v == scan_dir)
            ax_lax.plot([i1, i2], [j1, j2], color=colors[label], linewidth=2.0,
                        label=label, zorder=3)
        else:
            ax_lax.plot([i1, i2], [j1, j2], color="white", linewidth=0.6,
                        alpha=0.4, zorder=2)

    ax_lax.legend(loc="upper right", fontsize=8, framealpha=0.7)

    # Three SAX panels
    for ax, (label, scan_dir) in zip(axes[1:], selected.items()):
        img = sax_images_map[label]
        ax.imshow(img.T, cmap="gray", origin="lower")
        ax.set_title(f"SAX — {label}\n(scan {scan_dir.name})")
        ax.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    examples_dir = Path(__file__).parent / "examples"
    examples_dir.mkdir(exist_ok=True)

    print("Building Figure 1 (CS reconstruction stages)…")
    fig1 = build_figure1()
    fig1.savefig(examples_dir / "figure1_reconstruction_stages.png", dpi=150, bbox_inches="tight")
    print("Saved figure1_reconstruction_stages.png")

    print("Building Figure 2 (co-registration)…")
    fig2 = build_figure2()
    fig2.savefig(examples_dir / "figure2_coreg_lax_sax.png", dpi=150, bbox_inches="tight")
    print("Saved figure2_coreg_lax_sax.png")

    plt.show()
