"""Tests for Phase 4: DICOM geometry corrections and file writing.

Geometry functions (apply_corrections, shuffle_slices) are tested with
synthetic arrays and a real BrukerScan for metadata.  The write_dicom_series
integration test writes to a tmp_path and reads the result back with pydicom
to verify key DICOM tags.

Primary real-data fixture: AGORA2_F1 scan 10 (CINE_LAX4, coronal, L_R read).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pytest

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.dicom.geometry import apply_corrections, shuffle_slices
from rawtoDICOM.dicom.writer import write_dicom_series

_SUBJECT = (
    Path(__file__).parent
    / "raw-data"
    / "AGORA"
    / "cohort1"
    / "AGORA2_F1_s_2025121703_1_4_20251217_103442"
)
_CINE_SCAN_DIR = _SUBJECT / "10"


@pytest.fixture(scope="session")
def cine_scan() -> BrukerScan:
    return load_scan(_CINE_SCAN_DIR, read_raw=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scan_with_method(extra_method: dict) -> BrukerScan:
    """Minimal BrukerScan stub for geometry tests."""
    method = {
        "PVM_FovCm": np.array([4.5]),
        "PVM_DefMatrix": np.array([128, 128, 1]),
        "PVM_Phase1Offset": np.array([0.0]),
        "PVM_SPackArrReadOrient": "L_R",
        "PVM_SPackArrSliceOrient": "coronal",
        **extra_method,
    }
    return BrukerScan(
        scan_dir=Path("."),
        acqp={"ACQ_repetition_time": np.array([7.0])},
        method=method,
        visu_pars={
            "VisuCoreExtent": np.array([45.0, 45.0]),
            "VisuCorePosition": np.zeros((1, 3)),
            "VisuCoreOrientation": np.ones((1, 9)),
            "VisuSubjectId": "test_patient",
            "VisuAcquisitionProtocol": "test_protocol",
        },
        data=[],
    )


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------


def test_apply_corrections_dtype() -> None:
    scan = _make_scan_with_method({})
    images = np.ones((32, 32, 1, 10), dtype=float) * 500.0
    result = apply_corrections(images, scan)
    assert result.dtype == np.int16


def test_apply_corrections_max_near_30000() -> None:
    scan = _make_scan_with_method({})
    rng = np.random.default_rng(0)
    images = rng.random((32, 32, 1, 10)) * 1000.0
    result = apply_corrections(images, scan)
    assert int(np.max(result)) == pytest.approx(30000, abs=1)


def test_apply_corrections_zero_offset_no_roll() -> None:
    scan = _make_scan_with_method({"PVM_Phase1Offset": np.array([0.0])})
    images = np.arange(16, dtype=float).reshape(4, 4, 1, 1)
    result = apply_corrections(images, scan)
    # With zero offset, only normalisation applies — relative order preserved.
    flat_in = images[:, :, 0, 0].ravel()
    flat_out = result[:, :, 0, 0].ravel().astype(float)
    np.testing.assert_array_equal(
        np.argsort(flat_in), np.argsort(flat_out)
    )


def test_apply_corrections_nonzero_offset_shifts() -> None:
    # Offset = 4.5 mm, FOV = 4.5 cm, matrix = 128 → res = 0.035 cm/px
    # shift = (4.5/10) / 0.035 = 12.86 → round = 13 pixels
    scan = _make_scan_with_method({
        "PVM_Phase1Offset": np.array([4.5]),
        "PVM_FovCm": np.array([4.5]),
        "PVM_DefMatrix": np.array([128]),
    })
    images = np.zeros((32, 32, 1, 1), dtype=float)
    images[0, 0, 0, 0] = 100.0  # sentinel at y=0
    result = apply_corrections(images, scan)
    # Sentinel should no longer be at y=0 after shift.
    assert int(result[0, 0, 0, 0]) != 30000


# ---------------------------------------------------------------------------
# shuffle_slices
# ---------------------------------------------------------------------------


def test_shuffle_slices_single_slice_unchanged() -> None:
    images = np.arange(8, dtype=float).reshape(2, 2, 1, 2)
    result = shuffle_slices(images)
    np.testing.assert_array_equal(result, images)


def test_shuffle_slices_four_slices_order() -> None:
    # Four slices: half = 2
    # order[0::2] = [0, 1], order[1::2] = [2, 3]
    # → order = [0, 2, 1, 3]
    images = np.zeros((2, 2, 4, 1))
    for s in range(4):
        images[:, :, s, :] = float(s)
    result = shuffle_slices(images)
    assert float(result[0, 0, 0, 0]) == 0.0
    assert float(result[0, 0, 1, 0]) == 2.0
    assert float(result[0, 0, 2, 0]) == 1.0
    assert float(result[0, 0, 3, 0]) == 3.0


def test_shuffle_slices_preserves_shape() -> None:
    images = np.zeros((16, 16, 6, 20))
    result = shuffle_slices(images)
    assert result.shape == images.shape


# ---------------------------------------------------------------------------
# write_dicom_series
# ---------------------------------------------------------------------------


def test_write_dicom_creates_files(tmp_path: Path, cine_scan: BrukerScan) -> None:
    images = np.abs(np.random.randn(32, 32, 1, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    assert len(paths) == 1
    assert paths[0].exists()
    assert paths[0].suffix == ".dcm"


def test_write_dicom_readable(tmp_path: Path, cine_scan: BrukerScan) -> None:
    images = np.abs(np.random.randn(32, 32, 1, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    assert ds is not None


def test_write_dicom_modality(tmp_path: Path, cine_scan: BrukerScan) -> None:
    images = np.abs(np.random.randn(32, 32, 1, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    assert ds.Modality == "MR"


def test_write_dicom_number_of_frames(tmp_path: Path, cine_scan: BrukerScan) -> None:
    n_frames = 7
    images = np.abs(np.random.randn(32, 32, 1, n_frames)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    assert int(ds.NumberOfFrames) == n_frames


def test_write_dicom_pixel_spacing_set(tmp_path: Path, cine_scan: BrukerScan) -> None:
    images = np.abs(np.random.randn(32, 32, 1, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    assert hasattr(ds, "PixelSpacing")
    assert len(ds.PixelSpacing) == 2


def test_write_dicom_patient_id(tmp_path: Path, cine_scan: BrukerScan) -> None:
    images = np.abs(np.random.randn(32, 32, 1, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    # PatientID comes from VisuSubjectId in visu_pars
    assert ds.PatientID != ""


def test_write_dicom_multiple_slices(tmp_path: Path, cine_scan: BrukerScan) -> None:
    n_slices = 3
    images = np.abs(np.random.randn(32, 32, n_slices, 5)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    assert len(paths) == n_slices
    for p in paths:
        assert p.exists()


def test_write_dicom_pixel_data_correct_size(tmp_path: Path, cine_scan: BrukerScan) -> None:
    x, y, n_frames = 32, 32, 5
    images = np.abs(np.random.randn(x, y, 1, n_frames)) * 1000.0
    paths = write_dicom_series(images, cine_scan, tmp_path / "out")
    ds = pydicom.dcmread(str(paths[0]))
    expected_bytes = x * y * n_frames * 2  # int16 = 2 bytes
    assert len(ds.PixelData) == expected_bytes
