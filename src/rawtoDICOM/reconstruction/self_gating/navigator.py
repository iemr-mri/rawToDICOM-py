"""Navigator signal processing: PCA, timeline interpolation, curve cleaning.

Translates PCArunner.m, timeCorrecter.m, and curveCleaner.m.

Pipeline
--------
1. run_pca          — PCA on z-scored navigator midlines.
2. interpolate_timeline — place sparse midline timestamps onto a fine grid.
3. clean_curves     — bandpass-select cardiac and breathing principal components.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# Species-dependent physiological frequency bands (Hz).
_CARDIAC_BANDS: dict[str, tuple[float, float]] = {
    "rat": (4.0, 8.0),
    "mouse": (4.0, 10.0),
}
_BREATH_BANDS: dict[str, tuple[float, float]] = {
    "rat": (0.5, 1.5),
    "mouse": (0.5, 2.0),
}
_NEAR_ZERO_CUTOFF_HZ = 0.3  # frequencies below this are treated as signal drift


def run_pca(
    midlines: npt.NDArray[np.complexfloating],
    n_components: int = 10,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Run PCA on navigator midlines to extract motion principal components.

    Translates PCArunner.m.

    Real and imaginary parts are split and concatenated across coils so that
    each row of the input matrix represents one midline acquisition (timepoint)
    and each column is one spatial feature.

    Args:
        midlines:     Shape [coils, x_points, n_midlines].
        n_components: Number of PCs to return (default 10, matches MATLAB).

    Returns:
        scores:    Shape [n_components, n_midlines].  PC time-series.
        explained: Shape [n_components].  Fraction of variance explained (0–1).
    """
    coils, x_points, n_midlines = midlines.shape

    # Build [n_midlines, coils * 2 * x_points] feature matrix.
    # For each coil concatenate real columns then imaginary columns, matching
    # the MATLAB [realMidlines', imagMidlines'] horizontal concatenation.
    parts = []
    for coil in range(coils):
        parts.append(midlines[coil].real.T)  # [n_midlines, x_points]
        parts.append(midlines[coil].imag.T)
    feature_matrix: npt.NDArray[np.floating] = np.concatenate(parts, axis=1)

    # Z-score each feature (column) across timepoints — matches MATLAB zscore(X, 0, 2)
    # on the transposed matrix (rows = timepoints).
    col_std = feature_matrix.std(axis=0, ddof=1)
    col_std[col_std == 0] = 1.0  # avoid division by zero for constant features
    standardized = (feature_matrix - feature_matrix.mean(axis=0)) / col_std

    pca = PCA(n_components=n_components)
    scores_T = pca.fit_transform(standardized)  # [n_midlines, n_components]

    scores: npt.NDArray[np.floating] = scores_T.T
    explained: npt.NDArray[np.floating] = pca.explained_variance_ratio_

    return scores, explained


def interpolate_timeline(
    pca_scores: npt.NDArray[np.floating],
    midline_rate: int,
    tr_ms: float,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Resample sparse midline PCA scores onto a fine temporal grid.

    Translates timeCorrecter.m.

    Midlines are acquired at intervals of midline_rate × TR.  This function
    places each PC score at its true acquisition timestamp and linearly
    interpolates between them at a resolution of TR/100 ms.

    Args:
        pca_scores:   Shape [n_components, n_midlines].
        midline_rate: MidlineRate parameter — one midline per this many frames.
        tr_ms:        FrameRepTime in milliseconds.

    Returns:
        interpolated: Shape [n_components, n_fine_timepoints].
        timeline_ms:  Fine time axis in milliseconds.
    """
    n_components, n_midlines = pca_scores.shape

    midline_period_ms = midline_rate * tr_ms
    # Timestamps of each midline acquisition (milliseconds).
    midline_timestamps = np.arange(n_midlines) * midline_period_ms

    new_temp_res = tr_ms / 100.0  # fine temporal resolution in ms
    total_time = midline_timestamps[-1]
    timeline_ms: npt.NDArray[np.floating] = np.arange(0.0, total_time + 1.0, new_temp_res)

    interpolated = np.zeros((n_components, len(timeline_ms)), dtype=np.float64)
    for pc in range(n_components):
        interp_fn = interp1d(
            midline_timestamps,
            pca_scores[pc],
            kind="linear",
            bounds_error=False,
            fill_value=(pca_scores[pc, 0], pca_scores[pc, -1]),
        )
        interpolated[pc] = interp_fn(timeline_ms)

    return interpolated, timeline_ms


def clean_curves(
    interpolated: npt.NDArray[np.floating],
    explained: npt.NDArray[np.floating],
    midline_rate: int,
    tr_ms: float,
    species: str,
) -> tuple[
    npt.NDArray[np.floating],
    npt.NDArray[np.floating],
    float,
    float,
]:
    """Select and bandpass-filter cardiac and breathing principal components.

    Translates curveCleaner.m.

    For each PC, computes a one-sided PSD (periodogram), nulls near-zero
    frequencies (< 0.3 Hz drift), and scores the PC by the fraction of its
    spectral energy at the dominant peak.  The PC with the highest score
    within the cardiac band is selected as the cardiac signal; likewise for
    breathing.

    Butterworth bandpass filters are applied:
    - Cardiac:   [0.5 × cardiac_freq, 2 × cardiac_freq]
    - Breathing: low-pass at 4 × breath_freq with hard cutoff below 0.2 × breath_freq

    Args:
        interpolated: Shape [n_components, n_timepoints].
        explained:    Explained-variance fractions, shape [n_components].
        midline_rate: MidlineRate parameter (used to compute original Nyquist).
        tr_ms:        FrameRepTime in milliseconds.
        species:      "rat" or "mouse" (determines physiological frequency bands).

    Returns:
        cardiac_curve:  Bandpass-filtered cardiac PC, shape [n_timepoints].
        breath_curve:   Bandpass-filtered breathing PC, shape [n_timepoints].
        cardiac_freq_hz: Dominant cardiac frequency in Hz.
        breath_freq_hz:  Dominant breathing frequency in Hz.
    """
    if species not in _CARDIAC_BANDS:
        raise ValueError(f"Unknown species '{species}'. Expected 'rat' or 'mouse'.")

    cardiac_band = _CARDIAC_BANDS[species]
    breath_band = _BREATH_BANDS[species]

    n_components, n_timepoints = interpolated.shape
    sampling_freq_hz = 1000.0 / (tr_ms / 100.0)  # fine grid: TR/100 ms step
    freq_step = sampling_freq_hz / n_timepoints

    # Score each PC by the fraction of its PSD energy at its dominant peak,
    # after nulling near-zero frequencies to suppress drift.
    near_zero_bins = int(np.ceil(_NEAR_ZERO_CUTOFF_HZ / freq_step))

    psd_peak_hz = np.zeros(n_components)
    psd_energy = np.zeros(n_components)

    for pc_idx in range(n_components):
        freqs, psd = signal.periodogram(interpolated[pc_idx], fs=sampling_freq_hz)
        psd = psd.copy()
        psd[:near_zero_bins] = 0.0  # null drift
        if psd.sum() == 0.0:
            continue
        peak_bin = int(np.argmax(psd))
        psd_peak_hz[pc_idx] = freqs[peak_bin]
        psd_energy[pc_idx] = psd[peak_bin] / psd.sum()

    # Select the best PC within each physiological band.
    cardiac_mask = (psd_peak_hz >= cardiac_band[0]) & (psd_peak_hz <= cardiac_band[1])
    breath_mask = (psd_peak_hz >= breath_band[0]) & (psd_peak_hz <= breath_band[1])

    if not cardiac_mask.any():
        raise RuntimeError(
            f"No PC found with dominant frequency in cardiac band {cardiac_band} Hz. "
            "Check species parameter or scan quality."
        )
    if not breath_mask.any():
        raise RuntimeError(
            f"No PC found with dominant frequency in breathing band {breath_band} Hz. "
            "Check species parameter or scan quality."
        )

    cardiac_pc_idx = int(np.argmax(psd_energy * cardiac_mask))
    breath_pc_idx = int(np.argmax(psd_energy * breath_mask))

    cardiac_freq_hz = float(psd_peak_hz[cardiac_pc_idx])
    breath_freq_hz = float(psd_peak_hz[breath_pc_idx])

    cardiac_raw = interpolated[cardiac_pc_idx]
    breath_raw = interpolated[breath_pc_idx]

    # Butterworth bandpass for cardiac.
    cardiac_low = 0.5 * cardiac_freq_hz
    cardiac_high = min(2.0 * cardiac_freq_hz, sampling_freq_hz / 2.0 * 0.99)
    nyq = sampling_freq_hz / 2.0
    b_card, a_card = signal.butter(
        2, [cardiac_low / nyq, cardiac_high / nyq], btype="bandpass"
    )
    cardiac_curve: npt.NDArray[np.floating] = signal.filtfilt(b_card, a_card, cardiac_raw)

    # Butterworth low-pass for breathing (hard cutoff below 0.2 × breath_freq).
    breath_low_cutoff = min(4.0 * breath_freq_hz, nyq * 0.99)
    b_br, a_br = signal.butter(3, breath_low_cutoff / nyq, btype="low")
    breath_filtered: npt.NDArray[np.floating] = signal.filtfilt(b_br, a_br, breath_raw)
    # Hard high-pass: zero out frequencies below 0.2 × breath_freq in Fourier domain.
    hard_cutoff_bins = int(np.ceil(0.2 * breath_freq_hz / freq_step))
    breath_fft = np.fft.fft(breath_filtered)
    breath_fft[:hard_cutoff_bins] = 0.0
    breath_fft[-hard_cutoff_bins + 1 :] = 0.0
    breath_curve: npt.NDArray[np.floating] = np.fft.ifft(breath_fft).real

    # Normalize to [-50, 50] range (matches MATLAB).
    cardiac_max = np.max(np.abs(cardiac_curve))
    if cardiac_max > 0:
        cardiac_curve = 50.0 * cardiac_curve / cardiac_max
    breath_max = np.max(np.abs(breath_curve))
    if breath_max > 0:
        breath_curve = 50.0 * breath_curve / breath_max

    return cardiac_curve, breath_curve, cardiac_freq_hz, breath_freq_hz
