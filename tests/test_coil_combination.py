"""Tests for combine_coils."""

import numpy as np

from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.reconstruction.coil_combination import combine_coils
from rawtoDICOM.reconstruction.kspace import sort_kspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sorted_kspace(scan: BrukerScan) -> np.ndarray:  # type: ignore[type-arg]
    """Return sort_kspace output with flow_enc_dir squeezed out → [x, y, slices, frames, coils]."""
    kspace = sort_kspace(scan)
    return kspace[:, :, :, :, 0, :]  # squeeze flow_enc_dir axis (always 1)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_combine_coils_drops_coil_dim(bruker_scan: BrukerScan) -> None:
    """Output must have one fewer dimension than input (coil axis removed)."""
    kspace = _sorted_kspace(bruker_scan)
    images = combine_coils(kspace)
    assert images.ndim == kspace.ndim - 1


def test_combine_coils_output_shape(bruker_scan: BrukerScan) -> None:
    """Output shape must match input shape minus the last (coils) axis."""
    kspace = _sorted_kspace(bruker_scan)
    images = combine_coils(kspace)
    assert images.shape == kspace.shape[:-1]


# ---------------------------------------------------------------------------
# Output values
# ---------------------------------------------------------------------------


def test_combine_coils_output_nonneg(bruker_scan: BrukerScan) -> None:
    """Sum-of-squares magnitude is always non-negative."""
    kspace = _sorted_kspace(bruker_scan)
    images = combine_coils(kspace)
    assert np.all(images >= 0)


def test_combine_coils_output_real(bruker_scan: BrukerScan) -> None:
    """Output must be real-valued (sqrt of sum of squared magnitudes)."""
    kspace = _sorted_kspace(bruker_scan)
    images = combine_coils(kspace)
    assert np.isrealobj(images)


def test_combine_coils_not_all_zeros(bruker_scan: BrukerScan) -> None:
    kspace = _sorted_kspace(bruker_scan)
    images = combine_coils(kspace)
    assert np.any(images > 0)


# ---------------------------------------------------------------------------
# Synthetic correctness
# ---------------------------------------------------------------------------


def test_combine_coils_single_coil_matches_ifft() -> None:
    """With one coil, combine_coils must equal the absolute IFFT magnitude."""
    from rawtoDICOM.reconstruction.kspace import ifft2c

    rng = np.random.default_rng(7)
    kspace = rng.standard_normal((8, 8, 1, 2, 1)) + 1j * rng.standard_normal((8, 8, 1, 2, 1))
    images = combine_coils(kspace)
    expected = np.abs(ifft2c(kspace[:, :, :, :, 0]))
    np.testing.assert_allclose(images, expected, rtol=1e-10)
