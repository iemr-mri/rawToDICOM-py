"""Tests for load_scan and BrukerScan — the contract consumed by downstream modules.

The shape test is the most critical: it validates that raw binary parsing
produces the exact dimensions that sort_kspace will expect.
"""

from pathlib import Path

import numpy as np
import pytest

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.bruker.scan import BrukerScan

# ---------------------------------------------------------------------------
# BrukerScan structure
# ---------------------------------------------------------------------------


def test_scan_is_frozen(bruker_scan: BrukerScan) -> None:
    with pytest.raises((AttributeError, TypeError)):
        bruker_scan.acqp = {}  # type: ignore[misc]


def test_scan_dir_is_path(bruker_scan: BrukerScan) -> None:
    assert isinstance(bruker_scan.scan_dir, Path)
    assert bruker_scan.scan_dir.exists()


def test_acqp_not_empty(bruker_scan: BrukerScan) -> None:
    assert len(bruker_scan.acqp) > 0
    assert "ACQ_scan_name" in bruker_scan.acqp


def test_method_not_empty(bruker_scan: BrukerScan) -> None:
    assert len(bruker_scan.method) > 0


def test_visu_pars_not_empty(bruker_scan: BrukerScan) -> None:
    assert len(bruker_scan.visu_pars) > 0


# ---------------------------------------------------------------------------
# Raw data — the critical contract
# ---------------------------------------------------------------------------


def test_data_list_not_empty(bruker_scan: BrukerScan) -> None:
    assert len(bruker_scan.data) >= 1


def test_data_is_complex(bruker_scan: BrukerScan) -> None:
    assert np.iscomplexobj(bruker_scan.data[0])


def test_data_is_3d(bruker_scan: BrukerScan) -> None:
    """data[0] must be 3-D: [coils, x_points, acquisitions]."""
    assert bruker_scan.data[0].ndim == 3


def test_data_shape_matches_method(bruker_scan: BrukerScan) -> None:
    """data[0].shape[0] must equal the number of active receive channels.

    This is the contract consumed by sort_kspace.  Keep it green.
    """
    acqp = bruker_scan.acqp
    recv = acqp["ACQ_ReceiverSelectPerChan"]
    n_active = sum(1 for v in recv.ravel() if str(v).startswith("Yes"))
    assert bruker_scan.data[0].shape[0] == n_active


def test_data_no_nan(bruker_scan: BrukerScan) -> None:
    assert not np.any(np.isnan(bruker_scan.data[0]))


def test_data_not_all_zeros(bruker_scan: BrukerScan) -> None:
    assert np.any(bruker_scan.data[0] != 0)


# ---------------------------------------------------------------------------
# No-read mode
# ---------------------------------------------------------------------------


def test_no_raw_read_returns_empty_data(bruker_scan_params_only: BrukerScan) -> None:
    assert bruker_scan_params_only.data == []


def test_no_raw_read_has_params(bruker_scan_params_only: BrukerScan) -> None:
    assert "ACQ_scan_name" in bruker_scan_params_only.acqp


# ---------------------------------------------------------------------------
# All scans load
# ---------------------------------------------------------------------------


def test_all_scans_load(all_scan_dirs: list[Path]) -> None:
    """Every scan directory in the test set must load without raising."""
    for d in all_scan_dirs:
        scan = load_scan(d)
        assert len(scan.data) >= 1, f"No data loaded for {d}"
        assert "ACQ_scan_name" in scan.acqp, f"Missing ACQ_scan_name for {d}"
        assert scan.data[0].ndim == 3, f"Expected 3-D data for {d}"
