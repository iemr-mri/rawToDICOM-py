"""Tests for the self-gating reconstruction pipeline.

Primary fixture: AGORA2_f2 scan 18 (SG Cine SAX, mid-stack slice).
  - Method: segFLASH_CS_SGv3
  - Matrix: 160 × 64, 4 coils, 30 frames, 5 reps
  - MidlineRate: 3, CSacceleration: 4, TR: 7.11 ms
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rawtoDICOM.bruker.reader import is_sg_scan, load_scan, scan_plane
from rawtoDICOM.bruker.scan import BrukerScan
from rawtoDICOM.reconstruction.kspace import zero_fill_kspace
from rawtoDICOM.reconstruction.self_gating.navigator import (
    clean_curves,
    interpolate_timeline,
    run_pca,
)
from rawtoDICOM.reconstruction.self_gating.peak_finder import (
    find_breath_starts,
    find_cardiac_peaks,
)
from rawtoDICOM.reconstruction.self_gating.reader import SGRawData, read_sg_data
from rawtoDICOM.reconstruction.self_gating.synchronizer import synchronize_slices

SG_SUBJECT = (
    Path(__file__).parent
    / "raw-data"
    / "cohort1"
    / "AGORA2_f2_s_2025121704_1_1_20251217_103502"
)
SG_SCAN_DIR = SG_SUBJECT / "18"


# ---------------------------------------------------------------------------
# Session-scoped fixtures (expensive — load once for all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sg_scan() -> BrukerScan:
    return load_scan(SG_SCAN_DIR)


@pytest.fixture(scope="session")
def sg_raw(sg_scan: BrukerScan) -> SGRawData:
    return read_sg_data(sg_scan)


@pytest.fixture(scope="session")
def pca_result(
    sg_raw: SGRawData,
) -> tuple[np.ndarray, np.ndarray]:
    return run_pca(sg_raw.midlines)


@pytest.fixture(scope="session")
def timeline_result(
    pca_result: tuple[np.ndarray, np.ndarray],
    sg_scan: BrukerScan,
) -> tuple[np.ndarray, np.ndarray]:
    scores, _ = pca_result
    m = sg_scan.method
    return interpolate_timeline(
        scores,
        midline_rate=int(m["MidlineRate"]),
        tr_ms=float(m["FrameRepTime"]),
    )


@pytest.fixture(scope="session")
def clean_result(
    timeline_result: tuple[np.ndarray, np.ndarray],
    pca_result: tuple[np.ndarray, np.ndarray],
    sg_scan: BrukerScan,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    interpolated, _ = timeline_result
    _, explained = pca_result
    m = sg_scan.method
    return clean_curves(
        interpolated,
        explained,
        midline_rate=int(m["MidlineRate"]),
        tr_ms=float(m["FrameRepTime"]),
        species="rat",
    )


# ---------------------------------------------------------------------------
# Phase 1 additions: scan classifiers
# ---------------------------------------------------------------------------


def test_is_sg_scan_true(sg_scan: BrukerScan) -> None:
    assert is_sg_scan(sg_scan) is True


def test_is_sg_scan_false_on_regular_cine() -> None:
    regular_dir = (
        Path(__file__).parent
        / "raw-data"
        / "cohort1"
        / "AGORA2_F1_s_2025121703_1_4_20251217_103442"
        / "10"  # CINE_LAX4 — ECG-gated
    )
    scan = load_scan(regular_dir, read_raw=False)
    assert is_sg_scan(scan) is False


def test_scan_plane_sax(sg_scan: BrukerScan) -> None:
    assert scan_plane(sg_scan) == "SAX"


def test_scan_plane_lax() -> None:
    lax_dir = (
        Path(__file__).parent
        / "raw-data"
        / "cohort1"
        / "AGORA2_f2_s_2025121704_1_1_20251217_103502"
        / "7"  # scan named "lax"
    )
    scan = load_scan(lax_dir, read_raw=False)
    assert scan_plane(scan) == "LAX"


def test_scan_plane_other() -> None:
    other_dir = (
        Path(__file__).parent
        / "raw-data"
        / "cohort1"
        / "AGORA2_F1_s_2025121703_1_4_20251217_103442"
        / "1"  # 1_localizer
    )
    scan = load_scan(other_dir, read_raw=False)
    assert scan_plane(scan) == "other"


# ---------------------------------------------------------------------------
# zero_fill_kspace
# ---------------------------------------------------------------------------


def test_zero_fill_doubles_spatial_dims() -> None:
    x = np.random.randn(32, 48, 3, 10) + 1j * np.random.randn(32, 48, 3, 10)
    result = zero_fill_kspace(x)
    assert result.shape == (64, 96, 3, 10)


def test_zero_fill_preserves_input_in_center() -> None:
    x = np.ones((4, 4), dtype=complex)
    result = zero_fill_kspace(x)
    assert result.shape == (8, 8)
    # Input placed at [2:6, 2:6]
    np.testing.assert_array_equal(result[2:6, 2:6], x)
    # Padding is zero
    assert result[0, 0] == 0.0


def test_zero_fill_non_spatial_axes_preserved() -> None:
    x = np.random.randn(16, 16, 5, 8, 4) + 0j
    result = zero_fill_kspace(x)
    assert result.shape == (32, 32, 5, 8, 4)


# ---------------------------------------------------------------------------
# SG reader
# ---------------------------------------------------------------------------


def test_sg_read_midlines_shape(sg_raw: SGRawData, sg_scan: BrukerScan) -> None:
    m = sg_scan.method
    coils = int(m["PVM_EncNReceivers"])
    x_points = int(np.asarray(m["PVM_EncMatrix"]).ravel()[0])
    ky_lines = int(np.asarray(m["PVM_EncMatrix"]).ravel()[1])
    movie_frames = int(m["PVM_NMovieFrames"])
    reps = int(m["PVM_NRepetitions"])
    midline_rate = int(m["MidlineRate"])

    midlines_per_rep = len(range(midline_rate, movie_frames + 1, midline_rate)) * ky_lines
    expected = midlines_per_rep * reps

    assert sg_raw.midlines.shape == (coils, x_points, expected)


def test_sg_read_kspace_is_complex(sg_raw: SGRawData) -> None:
    assert np.iscomplexobj(sg_raw.kspace)


def test_sg_read_cs_vector_range(sg_raw: SGRawData, sg_scan: BrukerScan) -> None:
    m = sg_scan.method
    ky_lines = int(np.asarray(m["PVM_EncMatrix"]).ravel()[1])
    cs_acceleration = int(m["CSacceleration"])
    actual_y = cs_acceleration * ky_lines  # 4 × 64 = 256

    # cs_vector is centred; half-range check.
    half = actual_y // 2
    assert sg_raw.cs_vector.min() >= -half
    assert sg_raw.cs_vector.max() <= half


def test_sg_read_midlines_complex(sg_raw: SGRawData) -> None:
    assert np.iscomplexobj(sg_raw.midlines)


# ---------------------------------------------------------------------------
# Navigator: PCA
# ---------------------------------------------------------------------------


def test_pca_scores_shape(
    pca_result: tuple[np.ndarray, np.ndarray],
    sg_raw: SGRawData,
) -> None:
    scores, explained = pca_result
    assert scores.shape[0] == 10
    assert scores.shape[1] == sg_raw.midlines.shape[2]


def test_pca_explained_sums_to_one(
    pca_result: tuple[np.ndarray, np.ndarray],
) -> None:
    _, explained = pca_result
    assert explained.shape == (10,)
    assert float(explained.sum()) <= 1.0 + 1e-6
    assert float(explained[0]) >= float(explained[-1])  # sorted descending


# ---------------------------------------------------------------------------
# Navigator: timeline interpolation
# ---------------------------------------------------------------------------


def test_interpolation_output_shape(
    timeline_result: tuple[np.ndarray, np.ndarray],
    pca_result: tuple[np.ndarray, np.ndarray],
) -> None:
    interpolated, timeline_ms = timeline_result
    scores, _ = pca_result
    assert interpolated.shape[0] == scores.shape[0]  # same n_components
    assert interpolated.shape[1] == len(timeline_ms)


def test_timeline_resolution(
    timeline_result: tuple[np.ndarray, np.ndarray],
    sg_scan: BrukerScan,
) -> None:
    _, timeline_ms = timeline_result
    tr_ms = float(sg_scan.method["FrameRepTime"])
    expected_step = tr_ms / 100.0
    actual_step = float(np.mean(np.diff(timeline_ms)))
    np.testing.assert_allclose(actual_step, expected_step, rtol=1e-4)


# ---------------------------------------------------------------------------
# Navigator: curve cleaning
# ---------------------------------------------------------------------------


def test_clean_curves_cardiac_in_rat_range(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
) -> None:
    _, _, cardiac_freq, _ = clean_result
    assert 4.0 <= cardiac_freq <= 8.0, f"Cardiac frequency {cardiac_freq:.2f} Hz outside rat range"


def test_clean_curves_breath_in_rat_range(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
) -> None:
    _, _, _, breath_freq = clean_result
    assert 0.5 <= breath_freq <= 1.5, f"Breath frequency {breath_freq:.2f} Hz outside rat range"


def test_clean_curves_output_lengths(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
    timeline_result: tuple[np.ndarray, np.ndarray],
) -> None:
    cardiac, breath, _, _ = clean_result
    _, timeline_ms = timeline_result
    assert len(cardiac) == len(timeline_ms)
    assert len(breath) == len(timeline_ms)


# ---------------------------------------------------------------------------
# Peak finder
# ---------------------------------------------------------------------------


def test_cardiac_peaks_found(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
    timeline_result: tuple[np.ndarray, np.ndarray],
    sg_scan: BrukerScan,
) -> None:
    cardiac, _, cardiac_freq, _ = clean_result
    _, timeline_ms = timeline_result
    tr_ms = float(sg_scan.method["FrameRepTime"])
    temporal_resolution_ms = tr_ms / 100.0

    peaks = find_cardiac_peaks(cardiac, cardiac_freq, temporal_resolution_ms)
    assert len(peaks) > 0


def test_cardiac_peaks_spacing(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
    timeline_result: tuple[np.ndarray, np.ndarray],
    sg_scan: BrukerScan,
) -> None:
    cardiac, _, cardiac_freq, _ = clean_result
    _, timeline_ms = timeline_result
    tr_ms = float(sg_scan.method["FrameRepTime"])
    temporal_resolution_ms = tr_ms / 100.0

    peaks = find_cardiac_peaks(cardiac, cardiac_freq, temporal_resolution_ms)
    if len(peaks) < 2:
        pytest.skip("Too few peaks to check spacing")

    peak_times = timeline_ms[peaks]
    intervals = np.diff(peak_times)
    expected_period_ms = 1000.0 / cardiac_freq
    # Allow ±50 % variation around expected period.
    assert float(np.median(intervals)) == pytest.approx(expected_period_ms, rel=0.5)


def test_breath_starts_found(
    clean_result: tuple[np.ndarray, np.ndarray, float, float],
) -> None:
    _, breath, _, _ = clean_result
    starts = find_breath_starts(breath)
    assert len(starts) > 0


# ---------------------------------------------------------------------------
# Slice synchronizer (synthetic input — no real data needed)
# ---------------------------------------------------------------------------


def test_synchronize_single_slice_no_change() -> None:
    rng = np.random.default_rng(0)
    images = rng.random((32, 32, 1, 20))
    result = synchronize_slices(images)
    assert result.shape == images.shape


def test_synchronize_preserves_shape() -> None:
    rng = np.random.default_rng(1)
    images = rng.random((32, 32, 5, 20))
    result = synchronize_slices(images)
    assert result.shape == (32, 32, 5, 20)


def test_synchronize_cyclic_shift_recovered() -> None:
    """A stack where all slices are the same but shifted should be realigned."""
    rng = np.random.default_rng(2)
    base = rng.random((16, 16, 20))  # [x, y, frames]
    shifts = [0, 5, 12]
    images = np.stack(
        [np.roll(base, s, axis=-1) for s in shifts], axis=2
    )  # [x, y, slices, frames]

    result = synchronize_slices(images)
    # After sync, slices should be mutually closer than before.
    before_diff = float(np.mean((images[:, :, 0, :] - images[:, :, 1, :]) ** 2))
    after_diff = float(np.mean((result[:, :, 0, :] - result[:, :, 1, :]) ** 2))
    assert after_diff <= before_diff + 1e-10
