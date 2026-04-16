"""Tests for parse_bruker_params — verifies the JCAMP-DX parser against real data."""

from pathlib import Path

import numpy as np

from rawtoDICOM.bruker.reader import parse_bruker_params

# ---------------------------------------------------------------------------
# Scalar parsing
# ---------------------------------------------------------------------------


def test_scan_name_is_string(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    assert isinstance(params["ACQ_scan_name"], str)
    assert len(params["ACQ_scan_name"]) > 0


def test_ni_is_int(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    assert isinstance(params["NI"], int)
    assert params["NI"] >= 1


def test_nr_is_int(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    assert isinstance(params["NR"], int)
    assert params["NR"] >= 1


def test_bytorda_is_string(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    assert params["BYTORDA"] in ("little", "big")


# ---------------------------------------------------------------------------
# Array parsing
# ---------------------------------------------------------------------------


def test_enc_matrix_is_ndarray(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "method")
    enc = params["PVM_EncMatrix"]
    assert isinstance(enc, np.ndarray)
    assert enc.shape == (2,)
    assert enc.dtype.kind in ("i", "u", "f")  # integer or float


def test_acq_jobs_size_is_scalar(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    jobs_size = params["ACQ_jobs_size"]
    assert isinstance(jobs_size, (int, float, np.integer, np.floating))
    assert int(jobs_size) >= 1


# ---------------------------------------------------------------------------
# Struct array parsing
# ---------------------------------------------------------------------------


def test_acq_jobs_is_2d_object_array(scan_dir: Path) -> None:
    """ACQ_jobs must parse as a 2-D object array with shape (n_fields, n_jobs)."""
    params = parse_bruker_params(scan_dir / "acqp")
    jobs = params["ACQ_jobs"]
    assert isinstance(jobs, np.ndarray)
    assert jobs.ndim == 2
    n_jobs = int(params["ACQ_jobs_size"])
    assert jobs.shape[1] == n_jobs


def test_acq_jobs_scan_size_accessible(scan_dir: Path) -> None:
    """Job scan size at row 0 must be a positive integer."""
    params = parse_bruker_params(scan_dir / "acqp")
    jobs = params["ACQ_jobs"]
    scan_size = int(jobs[0, 0])
    assert scan_size > 0


def test_acq_jobs_name_is_string(scan_dir: Path) -> None:
    """Job name at row 8 must be a string like 'job0'."""
    params = parse_bruker_params(scan_dir / "acqp")
    jobs = params["ACQ_jobs"]
    name = str(jobs[8, 0])
    assert name.startswith("job")


def test_acq_scan_pipe_settings_is_2d(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    settings = params["ACQ_ScanPipeJobSettings"]
    assert isinstance(settings, np.ndarray)
    assert settings.ndim == 2


def test_acq_scan_pipe_data_format_accessible(scan_dir: Path) -> None:
    """Data format field (row 1) must be a non-empty string."""
    params = parse_bruker_params(scan_dir / "acqp")
    settings = params["ACQ_ScanPipeJobSettings"]
    fmt = str(settings[1, 0])
    assert len(fmt) > 0


# ---------------------------------------------------------------------------
# Receiver select parsing
# ---------------------------------------------------------------------------


def test_receiver_select_parsed(scan_dir: Path) -> None:
    """ACQ_ReceiverSelectPerChan must be parseable and contain Yes/No tokens."""
    params = parse_bruker_params(scan_dir / "acqp")
    recv = params["ACQ_ReceiverSelectPerChan"]
    assert isinstance(recv, np.ndarray)
    values = [str(v) for v in recv.ravel()]
    assert any(v.startswith("Yes") or v == "No" for v in values)


def test_active_receiver_count_positive(scan_dir: Path) -> None:
    params = parse_bruker_params(scan_dir / "acqp")
    recv = params["ACQ_ReceiverSelectPerChan"]
    n_active = sum(1 for v in recv.ravel() if str(v).startswith("Yes"))
    assert n_active > 0


# ---------------------------------------------------------------------------
# Cross-scan robustness
# ---------------------------------------------------------------------------


def test_all_scans_parse_acqp(all_scan_dirs: list[Path]) -> None:
    """Every scan directory must parse its acqp without raising."""
    for d in all_scan_dirs:
        params = parse_bruker_params(d / "acqp")
        assert "ACQ_scan_name" in params, f"Missing ACQ_scan_name in {d}"


def test_all_scans_parse_method(all_scan_dirs: list[Path]) -> None:
    for d in all_scan_dirs:
        params = parse_bruker_params(d / "method")
        assert isinstance(params, dict)
        assert len(params) > 0


def test_visu_pars_parses(scan_dir: Path) -> None:
    visu = parse_bruker_params(scan_dir / "pdata" / "1" / "visu_pars")
    assert isinstance(visu, dict)
    assert len(visu) > 0
