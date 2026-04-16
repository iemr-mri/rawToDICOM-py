"""SG raw-data reader — separates navigator midlines from k-space acquisitions.

Translates CSSGcineReader.m.

The raw acquisition buffer from BrukerScan.data[0] has shape
[coils, x_points, total_acquisitions].  The acquisition loop runs as:

    for rep in repetitions:
        for ky_line in kyLines:
            for frame in movieFrames:
                acquire one line

Every frame index that is a multiple of MidlineRate is a navigator midline
(acquired at ky = 0, the k-space centerline).  All other frames are regular
k-space lines whose ky position is given by cs_vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.scan import BrukerScan


@dataclass(frozen=True)
class SGRawData:
    """Container for the separated SG acquisition.

    Attributes:
        kspace:    Regular k-space acquisitions.
                   Shape [coils, x, n_kspace_acqs].  Each acquisition is one
                   ky line; the corresponding ky position is in cs_vector.
        midlines:  Navigator (centerline) acquisitions.
                   Shape [coils, x, n_midlines].
        cs_vector: Integer ky-position for each k-space acquisition (0-indexed,
                   centred so 0 is DC).  Shape [n_kspace_acqs].
        scan:      The originating BrukerScan (params access).
    """

    kspace: npt.NDArray[np.complexfloating]
    midlines: npt.NDArray[np.complexfloating]
    cs_vector: npt.NDArray[np.intp]
    scan: BrukerScan


def read_sg_data(scan: BrukerScan) -> SGRawData:
    """Separate navigator midlines from k-space lines.

    Translates CSSGcineReader.m (the data-sorting section).

    Args:
        scan: A BrukerScan loaded with read_raw=True.

    Returns:
        SGRawData with kspace, midlines, and cs_vector populated.
    """
    method = scan.method
    raw = scan.data[0]  # [coils, x, total_acquisitions]

    coils = raw.shape[0]
    x_points = raw.shape[1]

    movie_frames: int = int(method["PVM_NMovieFrames"])
    ky_lines: int = int(np.asarray(method["PVM_EncMatrix"]).ravel()[1])
    repetitions: int = int(method["PVM_NRepetitions"])
    midline_rate: int = int(method["MidlineRate"])
    cs_acceleration: int = int(method["CSacceleration"])

    # Midline positions within each frame-cycle (1-indexed frame numbers that
    # are multiples of midline_rate, matching MATLAB's ismember logic).
    midline_frames = set(range(midline_rate, movie_frames + 1, midline_rate))

    # Pre-count to allocate arrays up front.
    midlines_per_rep = len(midline_frames) * ky_lines
    kspace_per_rep = (movie_frames - len(midline_frames)) * ky_lines
    total_midlines = midlines_per_rep * repetitions
    total_kspace = kspace_per_rep * repetitions

    midlines = np.zeros((coils, x_points, total_midlines), dtype=raw.dtype)
    kspace_lines = np.zeros((coils, x_points, total_kspace), dtype=raw.dtype)

    # CS vector: maps each raw acquisition (in order) to a ky index.
    # Formula from CSSGcineReader.m:
    #   cs_vector = round(CSPhaseEncList * actual_y / (2 * CSacceleration))
    # actual_y = CSacceleration * kyLines; result is centred around 0.
    actual_y = cs_acceleration * ky_lines
    cs_raw = np.asarray(method["CSPhaseEncList"]).ravel()
    cs_single_rep = np.round(cs_raw * actual_y / (2 * cs_acceleration)).astype(np.intp)
    cs_full = np.tile(cs_single_rep, repetitions)  # replicated across reps

    # Reshape raw to [coils, x, movie_frames, ky_lines, reps] using Fortran
    # (column-major) order to match the MATLAB reshape convention.
    # MATLAB reshape(rawWithMid, [coils, x, movieFrames, kyLines, reps])
    # then permute([1,2,4,3,5]) → [coils, x, kyLines, movieFrames, reps]
    raw_reshaped = raw.reshape(coils, x_points, movie_frames, ky_lines, repetitions, order="F")
    raw_reshaped = raw_reshaped.transpose(0, 1, 3, 2, 4)
    # Now shape: [coils, x, ky_lines, movie_frames, reps]

    mid_idx = 0
    ks_idx = 0
    cs_idx = 0  # tracks position into cs_full

    for rep in range(repetitions):
        for ky in range(ky_lines):
            for frame_1idx in range(1, movie_frames + 1):
                acq = raw_reshaped[:, :, ky, frame_1idx - 1, rep]  # [coils, x]
                if frame_1idx in midline_frames:
                    midlines[:, :, mid_idx] = acq
                    mid_idx += 1
                else:
                    kspace_lines[:, :, ks_idx] = acq
                    ks_idx += 1
                cs_idx += 1

    cs_vector = cs_full[:total_kspace]

    return SGRawData(
        kspace=kspace_lines,
        midlines=midlines,
        cs_vector=cs_vector,
        scan=scan,
    )
