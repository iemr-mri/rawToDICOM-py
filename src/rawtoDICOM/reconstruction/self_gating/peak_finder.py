"""Cardiac and breathing peak detection for self-gated MRI.

Translates PCApeakFinder.m (cardiac rise/fall search and widthBreathHandler).

The MATLAB pipeline defaults to:
  pm.performCurveFit = false  → rough peaks only, no skewed-Gaussian fine fit
  pm.diffOrWidth = "width"    → breath starts via peak/valley width comparison
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import signal


def find_cardiac_peaks(
    cardiac_curve: npt.NDArray[np.floating],
    cardiac_freq_hz: float,
    temporal_resolution_ms: float,
    min_height: float = 5.0,
) -> npt.NDArray[np.intp]:
    """Detect cardiac peaks using a rise-then-fall sliding window.

    Translates the rough-peak section of PCApeakFinder.m.
    Fine curve fitting (skewed Gaussian) is not implemented — the MATLAB
    pipeline disables it with pm.performCurveFit = false.

    Algorithm:
    - Compute rise_time and fall_time as 45 % of the expected points per
      heartbeat.
    - Slide through the curve.  At each position check:
        curve[t + rise] - curve[t]           > min_height  (rising)
        curve[t + rise] - curve[t + rise + fall] > min_height  (falling)
    - Take the argmax within the detected window as the peak.
    - Advance by rise + fall steps to avoid double-counting.

    Args:
        cardiac_curve:         Filtered cardiac PC, shape [n_timepoints].
        cardiac_freq_hz:       Dominant cardiac frequency in Hz.
        temporal_resolution_ms: Time step of the fine timeline (TR / 100).
        min_height:            Minimum required rise/fall amplitude.

    Returns:
        peak_indices: Integer indices into cardiac_curve, shape [n_peaks].
    """
    cardiac_period_ms = 1000.0 / cardiac_freq_hz
    points_per_hb = cardiac_period_ms / temporal_resolution_ms

    rise_time = int(np.floor(0.45 * points_per_hb))
    fall_time = int(np.floor(0.45 * points_per_hb))
    window = rise_time + fall_time

    if window == 0:
        raise ValueError(
            "Rise+fall window is zero — cardiac_freq_hz or temporal_resolution_ms "
            "are likely wrong."
        )

    n = len(cardiac_curve)
    peak_indices: list[int] = []
    t = 0

    while t <= n - window - 1:
        peak_candidate = t + rise_time
        after_fall = t + window

        rises = cardiac_curve[peak_candidate] - cardiac_curve[t] > min_height
        falls = cardiac_curve[peak_candidate] - cardiac_curve[after_fall] > min_height

        if rises and falls:
            chunk_end = min(after_fall + 1, n)
            chunk = cardiac_curve[t:chunk_end]
            local_peak = int(np.argmax(chunk))
            peak_indices.append(t + local_peak)
            t += window
        else:
            t += 1

    return np.array(peak_indices, dtype=np.intp)


def find_breath_starts(
    breath_curve: npt.NDArray[np.floating],
    flip: bool = False,
) -> npt.NDArray[np.intp]:
    """Detect the starts of breath cycles using peak/valley width comparison.

    Translates widthBreathHandler in PCApeakFinder.m.
    (MATLAB defaults to pm.diffOrWidth = "width".)

    Finds all peaks and valleys of the breathing curve via scipy.signal.find_peaks.
    Compares mean widths: the narrower set (peaks vs valleys) marks the breath
    boundary.  Optionally flips the selection if the curve polarity is inverted.

    Args:
        breath_curve: Filtered breathing PC, shape [n_timepoints].
        flip:         If True, swap the peak/valley selection (matches
                      pm.breathFlipBool = true in MATLAB).

    Returns:
        breath_start_indices: Indices of breath-cycle starts, shape [n_starts].
    """
    prominence = float(np.std(breath_curve))

    peak_locs, peak_props = signal.find_peaks(breath_curve, prominence=prominence)
    valley_locs, valley_props = signal.find_peaks(-breath_curve, prominence=prominence)

    if len(peak_locs) == 0 and len(valley_locs) == 0:
        return np.array([], dtype=np.intp)

    peak_widths: npt.NDArray[np.floating] = (
        signal.peak_widths(breath_curve, peak_locs)[0]
        if len(peak_locs) > 0
        else np.array([np.inf])
    )
    valley_widths: npt.NDArray[np.floating] = (
        signal.peak_widths(-breath_curve, valley_locs)[0]
        if len(valley_locs) > 0
        else np.array([np.inf])
    )

    mean_peak_width = float(np.mean(peak_widths))
    mean_valley_width = float(np.mean(valley_widths))

    # Narrower set = breath boundary (sharper transition).
    peaks_are_narrower = mean_peak_width < mean_valley_width

    if peaks_are_narrower != flip:
        return np.asarray(peak_locs, dtype=np.intp)
    return np.asarray(valley_locs, dtype=np.intp)
