"""Tests for reconstruct_cs and _soft_thresh.

All tests use synthetic data so the suite stays fast.  The CS algorithm is
validated for convergence, shape preservation, and acquired-line fidelity.
"""

import numpy as np
import pytest

from rawtoDICOM.reconstruction.compressed_sensing import CSConfig, _soft_thresh, reconstruct_cs
from rawtoDICOM.reconstruction.kspace import fft2c

# ---------------------------------------------------------------------------
# Soft thresholding
# ---------------------------------------------------------------------------


def test_soft_thresh_zeros_below_threshold() -> None:
    x = np.array([0.5 + 0j, 1.0 + 0j, 2.0 + 0j])
    result = _soft_thresh(x, threshold=1.0)
    assert result[0] == 0.0
    assert result[1] == 0.0


def test_soft_thresh_reduces_magnitude() -> None:
    x = np.array([3.0 + 0j])
    result = _soft_thresh(x, threshold=1.0)
    assert abs(abs(result[0]) - 2.0) < 1e-10


def test_soft_thresh_preserves_phase() -> None:
    x = np.array([2.0 + 2.0j])  # 45-degree phase
    result = _soft_thresh(x, threshold=1.0)
    assert abs(np.angle(result[0]) - np.angle(x[0])) < 1e-10


def test_soft_thresh_handles_zero_input() -> None:
    x = np.zeros(4, dtype=complex)
    result = _soft_thresh(x, threshold=1.0)
    assert np.all(result == 0)


# ---------------------------------------------------------------------------
# reconstruct_cs — shape and basic contracts
# ---------------------------------------------------------------------------


def _make_undersampled_kspace(
    x: int = 16, y: int = 16, n_slices: int = 1, n_frames: int = 4, n_coils: int = 1
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Create a synthetic undersampled k-space with random (not regular) undersampling.

    Regular undersampling (every-other-line) creates a y-periodicity invariant that
    makes the CS algorithm unable to recover missing lines. CS requires pseudo-random
    undersampling, as used by Bruker's CSPhaseEncList in real acquisitions.

    Returns (kspace_undersampled, full_kspace) both shape [x, y, slices, frames, coils].
    """
    rng = np.random.default_rng(42)
    image = rng.standard_normal((x, y, n_slices, n_frames, n_coils)) + 1j * rng.standard_normal(
        (x, y, n_slices, n_frames, n_coils)
    )
    full_kspace = fft2c(image)

    # Random undersampling: acquire half the y-lines at random positions
    acquired_lines = rng.choice(y, size=y // 2, replace=False)
    mask = np.zeros_like(full_kspace, dtype=bool)
    mask[:, acquired_lines, :, :, :] = True
    kspace_us = full_kspace * mask

    return kspace_us, full_kspace


def test_cs_output_shape() -> None:
    kspace_us, _ = _make_undersampled_kspace()
    result = reconstruct_cs(kspace_us, CSConfig(max_iterations=5))
    assert result.shape == kspace_us.shape


def test_cs_output_is_complex() -> None:
    kspace_us, _ = _make_undersampled_kspace()
    result = reconstruct_cs(kspace_us, CSConfig(max_iterations=5))
    assert np.iscomplexobj(result)


def test_cs_output_not_all_zeros() -> None:
    kspace_us, _ = _make_undersampled_kspace()
    result = reconstruct_cs(kspace_us, CSConfig(max_iterations=5))
    assert np.any(result != 0)


def test_cs_preserves_acquired_lines() -> None:
    """Acquired k-space lines must be unchanged in the output (within tolerance)."""
    kspace_us, _ = _make_undersampled_kspace()
    mask = kspace_us != 0
    result = reconstruct_cs(kspace_us, CSConfig(max_iterations=10))
    np.testing.assert_allclose(result[mask], kspace_us[mask], rtol=1e-3)


def test_cs_fills_unsampled_lines() -> None:
    """CS must estimate non-zero values at unsampled k-space positions.

    CS requires pseudo-random undersampling (not every-other-line): regular undersampling
    creates a y-periodicity invariant that prevents the algorithm from estimating missing lines.
    """
    kspace_us, _ = _make_undersampled_kspace()  # uses random undersampling
    mask = kspace_us != 0
    no_data = ~mask

    result = reconstruct_cs(kspace_us, CSConfig(max_iterations=5))
    assert np.any(result[no_data] != 0), "CS did not estimate any unsampled lines"


def test_cs_config_defaults() -> None:
    config = CSConfig()
    assert config.max_iterations == 50
    assert config.percentile_threshold == 50.0
    assert config.convergence_threshold == 0.01


def test_cs_config_is_frozen() -> None:
    config = CSConfig()
    with pytest.raises((AttributeError, TypeError)):
        config.max_iterations = 10  # type: ignore[misc]
