"""Tests for fft2c, ifft2c, and sort_kspace."""

from pathlib import Path

import numpy as np

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.reconstruction.kspace import fft2c, ifft2c, sort_kspace

# ---------------------------------------------------------------------------
# FFT utilities
# ---------------------------------------------------------------------------


def test_fft2c_round_trip() -> None:
    """ifft2c(fft2c(x)) must recover x to numerical precision."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((32, 32, 4)) + 1j * rng.standard_normal((32, 32, 4))
    np.testing.assert_allclose(ifft2c(fft2c(x)), x, rtol=1e-10)


def test_ifft2c_round_trip() -> None:
    """fft2c(ifft2c(x)) must recover x to numerical precision."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((16, 16, 3)) + 1j * rng.standard_normal((16, 16, 3))
    np.testing.assert_allclose(fft2c(ifft2c(x)), x, rtol=1e-10)


def test_fft2c_dc_normalization() -> None:
    """fft2c of ones(N,N) must have DC = N (orthonormal DFT convention: 1/sqrt(N²) * N² = N)."""
    n = 8
    x = np.ones((n, n, 1), dtype=complex)
    result = fft2c(x)
    cx, cy = n // 2, n // 2  # DC is at center after fftshift
    assert abs(result[cx, cy, 0] - float(n)) < 1e-10


def test_fft2c_works_on_5d_input() -> None:
    """fft2c must handle arrays with more than 3 dimensions."""
    x = np.ones((8, 8, 2, 3, 4), dtype=complex)
    result = fft2c(x)
    assert result.shape == x.shape


# ---------------------------------------------------------------------------
# sort_kspace — fully sampled scan (scan 1, localizer)
# ---------------------------------------------------------------------------


def test_sort_kspace_is_6d(bruker_scan: BrukerScan) -> None:
    kspace = sort_kspace(bruker_scan)
    assert kspace.ndim == 6


def test_sort_kspace_is_complex(bruker_scan: BrukerScan) -> None:
    kspace = sort_kspace(bruker_scan)
    assert np.iscomplexobj(kspace)


def test_sort_kspace_output_shape(bruker_scan: BrukerScan) -> None:
    """Output shape must match [x, y, slices, frames, 1, coils] from method params."""
    kspace = sort_kspace(bruker_scan)
    m = bruker_scan.method
    x = int(np.asarray(m["PVM_EncMatrix"]).ravel()[0])
    y = int(np.asarray(m["PVM_EncMatrix"]).ravel()[1])
    slices = int(np.asarray(m["PVM_SPackArrNSlices"]).sum())  # sum across packages
    frames = int(np.asarray(m["PVM_NMovieFrames"]).ravel()[0])
    assert kspace.shape == (x, y, slices, frames, 1, bruker_scan.data[0].shape[0])


def test_sort_kspace_not_all_zeros(bruker_scan: BrukerScan) -> None:
    kspace = sort_kspace(bruker_scan)
    assert np.any(kspace != 0)


# ---------------------------------------------------------------------------
# sort_kspace — CS undersampled scan (scan 15, segFLASH_CS)
# ---------------------------------------------------------------------------


def test_sort_kspace_cs_y_size(cs_bruker_scan: BrukerScan) -> None:
    """CS kspace y-dimension must equal max(cs_indices) + 1 (full phase-encode range)."""
    cs_raw = np.asarray(cs_bruker_scan.method["CSPhaseEncList"]).ravel()
    cs_indices = ((cs_raw + 4) * 16).astype(int) - 1
    expected_y = int(np.max(cs_indices)) + 1
    kspace = sort_kspace(cs_bruker_scan)
    assert kspace.shape[1] == expected_y


def test_sort_kspace_cs_has_zeros(cs_bruker_scan: BrukerScan) -> None:
    """CS k-space must have unsampled lines (zeros at non-acquired positions)."""
    kspace = sort_kspace(cs_bruker_scan)
    assert np.any(kspace == 0)


def test_sort_kspace_cs_has_data(cs_bruker_scan: BrukerScan) -> None:
    """CS k-space must have non-zero data at acquired positions."""
    kspace = sort_kspace(cs_bruker_scan)
    assert np.any(kspace != 0)


# ---------------------------------------------------------------------------
# Cross-scan robustness
# ---------------------------------------------------------------------------


def test_all_scans_sort_kspace(all_scan_dirs: list[Path]) -> None:
    """Every scan must sort without raising and produce 6-D complex output."""
    for d in all_scan_dirs:
        scan = load_scan(d)
        kspace = sort_kspace(scan)
        assert kspace.ndim == 6, f"Expected 6-D for {d}, got {kspace.ndim}"
        assert np.iscomplexobj(kspace), f"Expected complex output for {d}"
