"""Data shuffler: bin SG acquisitions into cardiac frames.

Translates dataShuffler.m.

Each acquisition is tested against:
  1. Breath gate — must fall in [breath_range[0], breath_range[1]] of the
     surrounding breath cycle.
  2. Cardiac gate — must follow a 'good' cardiac peak (beat duration within
     beat_tolerance of the median).
  3. Frame bin — absolute time since previous peak, quantised to frame_period.

Accepted acquisitions are placed into the k-space matrix at the ky position
given by cs_vector.  Duplicate lines are averaged.  Navigator midlines are
averaged into the center ky line of each frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rawtoDICOM.reconstruction.self_gating.reader import SGRawData


@dataclass(frozen=True)
class ShuffleConfig:
    """Configuration for the data shuffler.

    Attributes:
        new_frame_count: Number of cardiac frames in the output.
        breath_range:    Accepted window of the breath cycle [start, end] as
                         fractions of the full breath period.  Default [0.25, 0.75]
                         discards the inhalation and exhalation extremes.
        beat_tolerance:  Maximum allowed deviation from median beat duration,
                         as a fraction (e.g. 0.05 = ±5 %).
        bin_tolerance:   Acceptance window around each frame bin, as a fraction
                         of the frame period.  1.0 = accept the whole period.
        abs_or_rel:      "abs" — frame bin relative to previous peak only.
                         "rel" — frame bin relative to the interval between
                         previous and next peak.
    """

    new_frame_count: int = 40
    breath_range: tuple[float, float] = (0.25, 0.75)
    beat_tolerance: float = 0.05
    bin_tolerance: float = 1.0
    abs_or_rel: str = "abs"


def shuffle_data(
    sg_raw: SGRawData,
    cardiac_peak_indices: npt.NDArray[np.intp],
    breath_start_indices: npt.NDArray[np.intp],
    timeline_ms: npt.NDArray[np.floating],
    config: ShuffleConfig = ShuffleConfig(),
) -> npt.NDArray[np.complexfloating]:
    """Sort self-gated k-space acquisitions into cardiac frames.

    Translates dataShuffler.m.

    Args:
        sg_raw:               Output of read_sg_data().
        cardiac_peak_indices: Indices into timeline_ms for each cardiac peak.
        breath_start_indices: Indices into timeline_ms for each breath start.
        timeline_ms:          Fine temporal axis (ms), length matches the
                              interpolated PCA curves.
        config:               Shuffling parameters.

    Returns:
        averaged_kspace: Shape [x, actual_y, new_frame_count, coils].
                         Zero where no acquisition landed; CS reconstruction
                         fills the gaps.
    """
    method = sg_raw.scan.method
    coils: int = int(method["PVM_EncNReceivers"])
    x_points: int = int(np.asarray(method["PVM_EncMatrix"]).ravel()[0])
    ky_lines: int = int(np.asarray(method["PVM_EncMatrix"]).ravel()[1])
    cs_acceleration: int = int(method["CSacceleration"])
    actual_y: int = cs_acceleration * ky_lines
    movie_frames: int = int(method["PVM_NMovieFrames"])
    repetitions: int = int(method["PVM_NRepetitions"])
    midline_rate: int = int(method["MidlineRate"])
    tr_ms: float = float(method["FrameRepTime"])
    new_frame_count = config.new_frame_count

    cardiac_freq_hz = 1000.0 / (
        float(np.median(np.diff(timeline_ms[cardiac_peak_indices]))) + 1e-12
    )
    cardiac_period_ms = 1000.0 / cardiac_freq_hz
    frame_period_ms = cardiac_period_ms / new_frame_count

    peak_times_ms = timeline_ms[cardiac_peak_indices]
    breath_times_ms = timeline_ms[breath_start_indices]

    # Mark good peaks: beats whose duration is within beat_tolerance of median.
    beat_durations = np.diff(peak_times_ms)
    median_dur = float(np.median(beat_durations))
    good_beats = np.abs(beat_durations - median_dur) <= config.beat_tolerance * median_dur
    # good_beats[i] = True means the beat from peak[i] to peak[i+1] is good.

    # Allocate output and a count array for averaging.
    kspace_out = np.zeros((coils, x_points, actual_y, new_frame_count), dtype=sg_raw.kspace.dtype)
    count_out = np.zeros((x_points, actual_y, new_frame_count), dtype=np.int32)

    # Navigator midline buffer: accumulate midlines per frame, average later.
    midline_out = np.zeros((coils, x_points, new_frame_count), dtype=sg_raw.kspace.dtype)
    midline_count = np.zeros(new_frame_count, dtype=np.int32)

    # Build acquisition timestamps.  Each acquisition occupies one slot in the
    # raw buffer, in the order: rep × ky_line × movie_frame.
    total_acqs = repetitions * ky_lines * movie_frames
    acq_times_ms = np.arange(total_acqs, dtype=float) * tr_ms

    # cs_vector for k-space acqs (not midlines) from SGRawData.
    # We need to rebuild the full acquisition sequence (including midline slots)
    # and map to cs_vector for the non-midline ones.
    midline_frames_set = set(range(midline_rate, movie_frames + 1, midline_rate))

    ks_acq_idx = 0  # index into sg_raw.kspace and sg_raw.cs_vector
    mid_acq_idx = 0  # index into sg_raw.midlines

    for acq in range(total_acqs):
        rem = acq % (ky_lines * movie_frames)
        frame_1idx = (rem % movie_frames) + 1  # 1-indexed frame

        acq_time = acq_times_ms[acq]
        is_midline = frame_1idx in midline_frames_set

        # --- Breath gate ---
        prev_breath_mask = breath_times_ms < acq_time
        if not prev_breath_mask.any() or np.all(breath_times_ms <= acq_time):
            # Before first or after last breath start
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        prev_breath_idx = int(np.where(prev_breath_mask)[0][-1])
        if prev_breath_idx + 1 >= len(breath_times_ms):
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        next_breath_time = breath_times_ms[prev_breath_idx + 1]
        prev_breath_time = breath_times_ms[prev_breath_idx]
        breath_pos = (acq_time - prev_breath_time) / (next_breath_time - prev_breath_time + 1e-12)

        if breath_pos < config.breath_range[0] or breath_pos > config.breath_range[1]:
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        # --- Cardiac gate ---
        earlier_peaks = np.where(peak_times_ms < acq_time)[0]
        if len(earlier_peaks) == 0:
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        last_peak_idx = int(earlier_peaks[-1])
        if last_peak_idx >= len(good_beats) or not good_beats[last_peak_idx]:
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        last_peak_time = peak_times_ms[last_peak_idx]

        # --- Frame bin ---
        if config.abs_or_rel == "abs":
            abs_acq_time = acq_time - last_peak_time
            bin_diff = abs_acq_time % frame_period_ms
            if bin_diff > frame_period_ms * config.bin_tolerance:
                if is_midline:
                    mid_acq_idx += 1
                else:
                    ks_acq_idx += 1
                continue
            new_frame = int(np.ceil(abs_acq_time / frame_period_ms))
        else:
            if last_peak_idx + 1 >= len(peak_times_ms):
                if is_midline:
                    mid_acq_idx += 1
                else:
                    ks_acq_idx += 1
                continue
            peak_distance = peak_times_ms[last_peak_idx + 1] - last_peak_time
            rel_acq = (acq_time - last_peak_time) / (peak_distance + 1e-12)
            bin_size = 1.0 / new_frame_count
            bin_diff = rel_acq % bin_size
            if bin_diff > bin_size * config.bin_tolerance:
                if is_midline:
                    mid_acq_idx += 1
                else:
                    ks_acq_idx += 1
                continue
            new_frame = int(np.ceil(rel_acq / bin_size))

        if new_frame < 1 or new_frame > new_frame_count:
            if is_midline:
                mid_acq_idx += 1
            else:
                ks_acq_idx += 1
            continue

        frame_idx = new_frame - 1  # 0-indexed

        if is_midline:
            acq_data = sg_raw.midlines[:, :, mid_acq_idx]  # [coils, x]
            midline_out[:, :, frame_idx] += acq_data
            midline_count[frame_idx] += 1
            mid_acq_idx += 1
        else:
            acq_data = sg_raw.kspace[:, :, ks_acq_idx]  # [coils, x]
            ky_pos = int(sg_raw.cs_vector[ks_acq_idx]) + actual_y // 2 - 1  # centre to 0-indexed
            ky_pos = int(np.clip(ky_pos, 0, actual_y - 1))

            existing = kspace_out[:, :, ky_pos, frame_idx]
            if np.any(existing != 0):
                kspace_out[:, :, ky_pos, frame_idx] = (existing + acq_data) / 2.0
            else:
                kspace_out[:, :, ky_pos, frame_idx] = acq_data
                count_out[:, ky_pos, frame_idx] += 1
            ks_acq_idx += 1

    # Insert averaged midlines at center ky line.
    center_ky = actual_y // 2
    for frame_idx in range(new_frame_count):
        if midline_count[frame_idx] > 0:
            kspace_out[:, :, center_ky, frame_idx] = (
                midline_out[:, :, frame_idx] / midline_count[frame_idx]
            )

    # Permute to [x, actual_y, new_frame_count, coils] for reconstruct_cs compatibility.
    result: npt.NDArray[np.complexfloating] = kspace_out.transpose(1, 2, 3, 0)
    return result
