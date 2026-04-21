"""Phase 5 integration tests: full CINE pipeline on real Bruker data.

Tests run the complete pipeline (load → sort_kspace → reconstruct_cs →
zero_fill → combine_coils → write_dicom_series) against real scans and
verify that the DICOM output is valid and has the expected tags.

Visual diagnostic plots are saved to ``tests/examples/`` so that each
pipeline stage can be inspected visually.  They are generated once and
re-used across test runs (not re-generated if already present).

Fixtures
--------
- Scan 10 (CINE_LAX4)  : fully-sampled LAX, 1 slice — fast
- Scan 15 (segFLASH_CS): CS-undersampled SAX stack — exercises CS path and
  produces multi-stage plots
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pydicom
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt

matplotlib.rcParams["figure.max_open_warning"] = 0

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.pipeline.cine import process_cine_scan
from rawtoDICOM.reconstruction.coil_combination import combine_coils
from rawtoDICOM.reconstruction.compressed_sensing import reconstruct_cs
from rawtoDICOM.reconstruction.kspace import sort_kspace, zero_fill_kspace

_SUBJECT = (
    Path(__file__).parent
    / "raw-data"
    / "AGORA"
    / "cohort1"
    / "AGORA2_F1_s_2025121703_1_4_20251217_103442"
)
_LAX_SCAN_DIR = _SUBJECT / "10"   # CINE_LAX4, fully sampled
_CS_SCAN_DIR = _SUBJECT / "15"    # segFLASH_CS, CS-undersampled

_EXAMPLES_DIR = Path(__file__).parent / "examples"


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save a matplotlib figure to tests/examples/."""
    _EXAMPLES_DIR.mkdir(exist_ok=True)
    fig.savefig(_EXAMPLES_DIR / name, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lax_scan():
    return load_scan(_LAX_SCAN_DIR)


@pytest.fixture(scope="module")
def cs_scan():
    return load_scan(_CS_SCAN_DIR)


@pytest.fixture(scope="module")
def cs_kspace_sorted(cs_scan):
    """K-space after sort_kspace: [x, y, slices, frames, flow_enc, coils]."""
    return sort_kspace(cs_scan)


@pytest.fixture(scope="module")
def cs_kspace_squeezed(cs_kspace_sorted):
    """K-space with flow_enc squeezed out: [x, y, slices, frames, coils]."""
    return cs_kspace_sorted[:, :, :, :, 0, :]


@pytest.fixture(scope="module")
def cs_kspace_reconstructed(cs_kspace_squeezed):
    """K-space after CS reconstruction: [x, y, slices, frames, coils]."""
    return reconstruct_cs(cs_kspace_squeezed)


@pytest.fixture(scope="module")
def cs_kspace_padded(cs_kspace_reconstructed):
    """K-space after 2× zero-fill: [2x, 2y, slices, frames, coils]."""
    return zero_fill_kspace(cs_kspace_reconstructed)


@pytest.fixture(scope="module")
def cs_images(cs_kspace_padded):
    """Coil-combined magnitude images: [x, y, slices, frames]."""
    return combine_coils(cs_kspace_padded)


# ---------------------------------------------------------------------------
# Unit checks — shapes and types
# ---------------------------------------------------------------------------


def test_sort_kspace_shape_cs(cs_kspace_sorted):
    assert cs_kspace_sorted.ndim == 6, "Expected [x, y, slices, frames, flow_enc, coils]"


def test_reconstruct_cs_shape(cs_kspace_squeezed, cs_kspace_reconstructed):
    assert cs_kspace_reconstructed.shape == cs_kspace_squeezed.shape


def test_zero_fill_doubles_spatial(cs_kspace_squeezed, cs_kspace_padded):
    x0, y0 = cs_kspace_squeezed.shape[:2]
    xp, yp = cs_kspace_padded.shape[:2]
    assert xp == 2 * x0 and yp == 2 * y0


def test_combine_coils_removes_coil_axis(cs_kspace_padded, cs_images):
    assert cs_images.ndim == 4  # [x, y, slices, frames]
    assert cs_images.shape[:3] == cs_kspace_padded.shape[:3]


def test_images_non_negative(cs_images):
    assert np.all(cs_images >= 0)


# ---------------------------------------------------------------------------
# DICOM output — LAX scan (fast, fully sampled)
# ---------------------------------------------------------------------------


def test_process_cine_scan_writes_dicom(tmp_path, lax_scan):
    out_dir = tmp_path / "lax_dicom"
    written = process_cine_scan(_LAX_SCAN_DIR, out_dir)
    assert len(written) >= 1
    for p in written:
        assert p.suffix == ".dcm"
        assert p.exists()


def test_dicom_tags_lax(tmp_path, lax_scan):
    out_dir = tmp_path / "lax_tags"
    written = process_cine_scan(_LAX_SCAN_DIR, out_dir)
    ds = pydicom.dcmread(str(written[0]))
    assert ds.Modality == "MR"
    assert int(ds.NumberOfFrames) >= 1
    assert hasattr(ds, "PixelSpacing")


def test_process_cine_scan_skip_if_exists(tmp_path):
    out_dir = tmp_path / "skip_test"
    written_first = process_cine_scan(_LAX_SCAN_DIR, out_dir)
    written_second = process_cine_scan(_LAX_SCAN_DIR, out_dir)
    assert len(written_first) == len(written_second)


# ---------------------------------------------------------------------------
# DICOM output — CS scan
# ---------------------------------------------------------------------------


def test_process_cine_cs_writes_dicom(tmp_path):
    out_dir = tmp_path / "cs_dicom"
    written = process_cine_scan(_CS_SCAN_DIR, out_dir)
    assert len(written) >= 1
    for p in written:
        assert p.exists()


def test_dicom_cs_pixel_data_readable(tmp_path):
    out_dir = tmp_path / "cs_pixel"
    written = process_cine_scan(_CS_SCAN_DIR, out_dir)
    ds = pydicom.dcmread(str(written[0]))
    pixels = ds.pixel_array
    assert pixels.ndim == 3  # [frames, rows, cols]
    assert pixels.dtype == np.int16


# ---------------------------------------------------------------------------
# Visual pipeline plots (saved to tests/examples/)
# ---------------------------------------------------------------------------


def _save_gif(figures: list[plt.Figure], name: str, interval_ms: int = 200) -> None:
    """Render each figure to a PIL Image and save as an animated GIF."""
    from io import BytesIO

    from PIL import Image

    _EXAMPLES_DIR.mkdir(exist_ok=True)

    pil_frames: list[Image.Image] = []
    for fig in figures:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy().convert("RGB"))
        plt.close(fig)

    if not pil_frames:
        return

    pil_frames[0].save(
        _EXAMPLES_DIR / name,
        save_all=True,
        append_images=pil_frames[1:],
        duration=interval_ms,
        loop=0,
    )


def _make_kspace_ax(ax: plt.Axes, kspace_2d: np.ndarray, title: str) -> None:
    """Draw a log-magnitude k-space panel onto *ax* (in-place)."""
    mag = np.log1p(np.abs(kspace_2d))
    vmax = float(np.percentile(mag, 99.9))
    ax.imshow(mag.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("kx", fontsize=7)
    ax.set_ylabel("ky", fontsize=7)
    ax.tick_params(labelsize=6)


def _make_image_ax(
    ax: plt.Axes,
    image_2d: np.ndarray,
    title: str,
    vmax: float,
) -> None:
    """Draw a magnitude image panel onto *ax* (in-place)."""
    ax.imshow(image_2d.T, cmap="gray", vmin=0, vmax=vmax, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


@pytest.mark.slow
def test_plot_figure1_reconstruction_stages(
    cs_kspace_squeezed, cs_kspace_reconstructed, cs_images
) -> None:
    """Figure 1: SAX middle-slice reconstruction stages as an animated GIF.

    Three panels per animation frame:
      - Panel 1 (static): k-space before CS reconstruction (frame 0, coil 0, log|k|).
      - Panel 2 (static): k-space after CS reconstruction  (frame 0, coil 0, log|k|).
      - Panel 3 (animated): coil-combined image cycling through all cardiac frames.
    """
    sl = cs_kspace_squeezed.shape[2] // 2  # middle slice index

    ksp_before = cs_kspace_squeezed[:, :, sl, 0, 0]
    ksp_after = cs_kspace_reconstructed[:, :, sl, 0, 0]

    n_frames = cs_images.shape[3]
    imgs_sl = cs_images[:, :, sl, :]  # [x, y, frames]
    vmax_img = float(np.percentile(imgs_sl, 99))

    gif_frames: list[plt.Figure] = []
    for frame_idx in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"SAX reconstruction stages — slice {sl + 1} — frame {frame_idx + 1}/{n_frames}",
            fontsize=10,
        )

        _make_kspace_ax(axes[0], ksp_before, "K-space before CS (frame 1, log|k|)")
        _make_kspace_ax(axes[1], ksp_after, "K-space after CS (frame 1, log|k|)")
        _make_image_ax(
            axes[2], imgs_sl[:, :, frame_idx], f"Final image — frame {frame_idx + 1}", vmax_img
        )

        fig.tight_layout()
        gif_frames.append(fig)

    _save_gif(gif_frames, "figure1_reconstruction_stages.gif", interval_ms=150)


# ---------------------------------------------------------------------------
# Figure 2 helpers: co-registration line (SAX slice plane ∩ LAX image plane)
# ---------------------------------------------------------------------------


def _plane_normal(orientation_row: np.ndarray) -> np.ndarray:
    """Return the slice-normal vector from a 9-element VisuCoreOrientation row."""
    x_dir = orientation_row[0:3]
    y_dir = orientation_row[3:6]
    normal = np.cross(x_dir, y_dir)
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-9 else normal


def _sax_slice_line_in_lax(
    lax_scan: "BrukerScan",
    sax_scan: "BrukerScan",
    sax_slice_idx: int,
) -> tuple[float, float, float, float] | None:
    """Compute the intersection of a SAX slice plane with the LAX image plane.

    Returns (x1_px, y1_px, x2_px, y2_px) in LAX pixel coordinates, clipped to
    the image boundary, or None when the planes are parallel or the line misses
    the image.

    Pixel convention: (0, 0) = top-left corner of first pixel;
    (matrix_x, matrix_y) = bottom-right corner of last pixel.
    """
    lax_vp = lax_scan.visu_pars
    sax_vp = sax_scan.visu_pars

    # LAX plane geometry (single slice).
    lax_orient = np.asarray(lax_vp["VisuCoreOrientation"]).reshape(-1, 9)[0]
    lax_pos = np.asarray(lax_vp["VisuCorePosition"]).reshape(-1, 3)[0]
    lax_x_dir = lax_orient[0:3]  # read direction unit vector
    lax_y_dir = lax_orient[3:6]  # phase direction unit vector
    lax_normal = _plane_normal(lax_orient)
    lax_extent = np.asarray(lax_vp["VisuCoreExtent"]).ravel()[:2]  # mm
    lax_matrix = np.asarray(lax_vp["VisuCoreSize"]).ravel()[:2].astype(float)  # pixels

    # SAX slice plane geometry.
    sax_orient_all = np.asarray(sax_vp["VisuCoreOrientation"]).reshape(-1, 9)
    sax_pos_all = np.asarray(sax_vp["VisuCorePosition"]).reshape(-1, 3)
    idx = min(sax_slice_idx, len(sax_orient_all) - 1)
    sax_normal = _plane_normal(sax_orient_all[idx])
    sax_pos = sax_pos_all[idx]

    # Direction of the intersection line = cross(lax_normal, sax_normal).
    line_dir = np.cross(lax_normal, sax_normal)
    line_dir_norm = float(np.linalg.norm(line_dir))
    if line_dir_norm < 1e-6:
        return None  # planes are parallel
    line_dir = line_dir / line_dir_norm

    # Find one point on the intersection line by solving the 3×3 system:
    #   dot(lax_normal, P) = dot(lax_normal, lax_pos)
    #   dot(sax_normal, P) = dot(sax_normal, sax_pos)
    #   dot(line_dir,   P) = dot(line_dir,   lax_pos)   (arbitrary anchor)
    A = np.array([lax_normal, sax_normal, line_dir])
    rhs = np.array(
        [
            np.dot(lax_normal, lax_pos),
            np.dot(sax_normal, sax_pos),
            np.dot(line_dir, lax_pos),
        ]
    )
    try:
        point_on_line = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        return None

    # Express the line relative to the LAX image centre in the image plane.
    p0_rel = point_on_line - lax_pos
    p0_u = float(np.dot(p0_rel, lax_x_dir))  # mm along read direction
    p0_v = float(np.dot(p0_rel, lax_y_dir))  # mm along phase direction
    d_u = float(np.dot(line_dir, lax_x_dir))
    d_v = float(np.dot(line_dir, lax_y_dir))

    # Clip the line to the LAX image bounding box in mm.
    half_u = lax_extent[0] / 2.0
    half_v = lax_extent[1] / 2.0
    t_intervals: list[tuple[float, float]] = []
    for d_comp, p0_comp, half in [(d_u, p0_u, half_u), (d_v, p0_v, half_v)]:
        if abs(d_comp) > 1e-9:
            t_a = (-half - p0_comp) / d_comp
            t_b = (half - p0_comp) / d_comp
            t_intervals.append((min(t_a, t_b), max(t_a, t_b)))
        elif abs(p0_comp) > half:
            return None  # line is outside image along this axis

    t_min = max(interval[0] for interval in t_intervals) if t_intervals else -1e9
    t_max = min(interval[1] for interval in t_intervals) if t_intervals else 1e9
    if t_min > t_max:
        return None  # line segment misses the image

    def _to_pixel(t: float) -> tuple[float, float]:
        u_mm = p0_u + t * d_u
        v_mm = p0_v + t * d_v
        px = (u_mm / lax_extent[0] + 0.5) * lax_matrix[0]
        py = (v_mm / lax_extent[1] + 0.5) * lax_matrix[1]
        return px, py

    x1, y1 = _to_pixel(t_min)
    x2, y2 = _to_pixel(t_max)
    return x1, y1, x2, y2


@pytest.mark.slow
def test_plot_figure2_coreg(lax_scan, cs_scan, cs_images) -> None:
    """Figure 2: LAX4 (frame 0) + SAX (frame 0) looping through slices as a GIF.

    Left panel: LAX4 first-frame image with a yellow line showing the position
    of the current SAX slice (updated each GIF frame).
    Right panel: SAX first-frame image for the current slice (animated).
    """
    from rawtoDICOM.reconstruction.coil_combination import combine_coils
    from rawtoDICOM.reconstruction.kspace import sort_kspace, zero_fill_kspace

    # Reconstruct LAX4 images — [x, y, slices=1, frames].
    lax_kspace = sort_kspace(lax_scan)
    lax_kspace_sq = lax_kspace[:, :, :, :, 0, :]
    lax_kspace_pad = zero_fill_kspace(lax_kspace_sq)
    lax_images = combine_coils(lax_kspace_pad)  # [x, y, 1, frames]

    lax_frame0 = lax_images[:, :, 0, 0]  # [x, y]
    vmax_lax = float(np.percentile(lax_frame0, 99.5))

    n_slices = cs_images.shape[2]
    sax_frame0_all = cs_images[:, :, :, 0]  # [x, y, slices]
    vmax_sax = float(np.percentile(sax_frame0_all, 99.5))

    gif_frames: list[plt.Figure] = []
    for sax_sl in range(n_slices):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(
            f"LAX4 & SAX co-registration — SAX slice {sax_sl + 1}/{n_slices}",
            fontsize=10,
        )

        # Left: LAX4 frame 0 (static background).
        axes[0].imshow(
            lax_frame0.T, cmap="gray", vmin=0, vmax=vmax_lax, origin="lower", aspect="auto"
        )
        axes[0].set_title("LAX4 — frame 1", fontsize=9)
        axes[0].axis("off")

        # Overlay yellow co-registration line for this SAX slice.
        line = _sax_slice_line_in_lax(lax_scan, cs_scan, sax_sl)
        if line is not None:
            x1, y1, x2, y2 = line
            axes[0].plot([x1, x2], [y1, y2], color="yellow", linewidth=2)

        # Right: SAX frame 0 for this slice.
        axes[1].imshow(
            sax_frame0_all[:, :, sax_sl].T,
            cmap="gray",
            vmin=0,
            vmax=vmax_sax,
            origin="lower",
            aspect="auto",
        )
        axes[1].set_title(f"SAX — frame 1, slice {sax_sl + 1}", fontsize=9)
        axes[1].axis("off")

        fig.tight_layout()
        gif_frames.append(fig)

    _save_gif(gif_frames, "figure2_coreg_lax_sax.gif", interval_ms=400)
