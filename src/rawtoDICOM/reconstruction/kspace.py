"""K-space utilities: normalized FFT pair and raw-to-sorted k-space reshaping.

Translates kspaceSort.m, fft2c.m, and ifft2c.m.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from rawtoDICOM.bruker.scan import BrukerScan


def fft2c(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Normalized centered 2-D FFT over axes 0 and 1.

    Matches MATLAB: 1/sqrt(Nx*Ny) * fftshift(fft2(ifftshift(x)))
    Works on arrays of any rank — axes 0 and 1 are always the spatial axes.
    """
    nx, ny = x.shape[0], x.shape[1]
    shifted = np.fft.ifftshift(x, axes=(0, 1))
    transformed = np.fft.fft2(shifted, axes=(0, 1))
    result: npt.NDArray[Any] = np.fft.fftshift(transformed, axes=(0, 1)) / np.sqrt(nx * ny)
    return result


def ifft2c(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Normalized centered 2-D IFFT over axes 0 and 1.

    Matches MATLAB: sqrt(Nx*Ny) * fftshift(ifft2(ifftshift(x)))
    Works on arrays of any rank — axes 0 and 1 are always the spatial axes.
    """
    nx, ny = x.shape[0], x.shape[1]
    shifted = np.fft.ifftshift(x, axes=(0, 1))
    transformed = np.fft.ifft2(shifted, axes=(0, 1))
    result: npt.NDArray[Any] = np.fft.fftshift(transformed, axes=(0, 1)) * np.sqrt(nx * ny)
    return result


def sort_kspace(scan: BrukerScan) -> npt.NDArray[Any]:
    """Reshape raw k-space to [x, y, slices, frames, flow_enc_dir, coils].

    Input from BrukerScan: data[0] shape [coils, x, acquisitions].

    Handles three cases from kspaceSort.m:
      1. Fully sampled — reshape + permute only.
      2. CS undersampled — place each acquisition into its correct y-position in a
         full (x × x) matrix using CSPhaseEncList.
      3. Partial echo — zero-fill the readout leading edge.

    Translates kspaceSort.m.
    """
    data = scan.data[0]  # [coils, x_data, acquisitions]
    method = scan.method

    coils = data.shape[0]
    x_data = data.shape[1]
    total_acq = data.shape[2]

    movie_frames = int(np.asarray(method.get("PVM_NMovieFrames", 1)).ravel()[0])
    # PVM_SPackArrNSlices is per-package; sum to get total slice count.
    slices = int(np.asarray(method.get("PVM_SPackArrNSlices", 1)).sum())
    flow_enc_dir = 1

    # When NR > 1, the raw file stores NR repetitions of the full acquisition
    # (NR is the outermost loop in Bruker: NR → y-lines → frames → readout).
    # Average across NR repetitions before sorting to improve SNR.
    #
    # For CS scans, CSPhaseEncList enumerates all y-line acquisitions for one NR,
    # so acq_per_nr comes directly from that list.  For fully-sampled scans, read
    # NR from acqp (the authoritative source) to avoid misidentifying y-lines as
    # repetitions when nr would otherwise be computed as total_acq / frames.
    if "CSPhaseEncList" in method:
        nr_lines = len(np.asarray(method["CSPhaseEncList"]).ravel())
        acq_per_nr = nr_lines  # CSPhaseEncList covers exactly one NR
        nr = total_acq // acq_per_nr
    else:
        nr = int(np.asarray(scan.acqp.get("NR", 1)).ravel()[0])
        acq_per_nr = total_acq // nr
    if nr > 1:
        data = data.reshape(coils, x_data, nr, acq_per_nr).mean(axis=2)
        total_acq = acq_per_nr

    y_data = total_acq // (movie_frames * slices * flow_enc_dir)

    # Step 1: reshape + permute (MATLAB column-major → Fortran order in NumPy)
    # MATLAB: reshape(raw, [coils, x, frames, y, slices, flowEncDir])
    # MATLAB: permute(..., [2 4 5 3 6 1]) → [x, y, slices, frames, flowEncDir, coils]
    kspace = data.reshape(coils, x_data, movie_frames, y_data, slices, flow_enc_dir, order="F")
    kspace = kspace.transpose(1, 3, 4, 2, 5, 0)

    # Step 2: CS undersampled — scatter acquisitions to their correct y-positions
    if "CSPhaseEncList" in method:
        kspace = _place_cs_lines(
            data, method, x_data, y_data, slices, movie_frames, flow_enc_dir, coils
        )

    # Step 3: partial echo zero-fill
    pft = float(np.asarray(method.get("PVM_EncPft", [1.0])).ravel()[0])
    if pft > 1.0:
        kspace = _zero_fill_partial_echo(kspace, x_data, pft)

    return kspace


def zero_fill_kspace(kspace: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """2× zero-fill: embed acquired k-space in center of a doubled spatial matrix.

    Doubles the size of axes 0 (x) and 1 (y) by placing the acquired data in
    the center of a zero-padded matrix. All other axes (slices, frames, coils,
    etc.) are preserved unchanged.

    Translates zipper.m (the zero-filling step; IFFT is left to the caller).
    Applies to both SG and non-SG pipelines.

    Args:
        kspace: Array of any rank. Axes 0 and 1 are the spatial (x, y) axes.

    Returns:
        Zero-padded array with shape (2*x, 2*y, ...).
    """
    nx, ny = kspace.shape[0], kspace.shape[1]
    out_shape = (2 * nx, 2 * ny) + kspace.shape[2:]
    padded: npt.NDArray[Any] = np.zeros(out_shape, dtype=kspace.dtype)

    start_x = nx // 2
    start_y = ny // 2
    padded[start_x : start_x + nx, start_y : start_y + ny] = kspace
    return padded


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _place_cs_lines(
    data: npt.NDArray[Any],
    method: dict[str, Any],
    x_data: int,
    y_data: int,
    slices: int,
    movie_frames: int,
    flow_enc_dir: int,
    coils: int,
) -> npt.NDArray[Any]:
    """Scatter CS acquisitions into a full (x × x) k-space matrix.

    CSPhaseEncList is 1-indexed in MATLAB; the transform (list + 4) * 16
    maps encoded values to k-space column positions. Subtract 1 for 0-indexing.

    MATLAB loop order (outermost first): slices → y_lines → flow_enc_dir → frames.
    count increments once per (slice, line, flow_enc, frame) tuple.
    """
    cs_raw = np.asarray(method["CSPhaseEncList"]).ravel()
    cs_indices = ((cs_raw.astype(float) + 4) * 16).astype(int) - 1  # 0-indexed

    # Full y-size is determined by the CS index range, not x_data.
    # Indices span 0 to (max_raw + 4)*16 - 1; add 1 for total size.
    full_y = int(np.max(cs_indices)) + 1
    kspace_us: npt.NDArray[Any] = np.zeros(
        (x_data, full_y, slices, movie_frames, flow_enc_dir, coils), dtype=data.dtype
    )

    count = 0
    for k in range(slices):
        for _line in range(y_data):
            for v in range(flow_enc_dir):
                for t in range(movie_frames):
                    ky = cs_indices[count]
                    # data[:, :, count] = [coils, x_data]; transpose → [x_data, coils]
                    kspace_us[:, ky, k, t, v, :] = data[:, :, count].T
                    count += 1

    return kspace_us


def _zero_fill_partial_echo(
    kspace: npt.NDArray[Any],
    x_data: int,
    pft: float,
) -> npt.NDArray[Any]:
    """Zero-fill the leading readout edge for partial-echo acquisitions.

    Matches MATLAB kspaceSort.m section 3:
      partialStart = round(xData * (PVM_EncPft - 1))   (0-indexed)
      full_x       = round(xData * PVM_EncPft)
    The acquired data is placed from partialStart onward; leading rows are zero.
    """
    full_x = round(x_data * pft)
    partial_start = round(x_data * (pft - 1))
    shape = (full_x,) + kspace.shape[1:]
    kspace_zero: npt.NDArray[Any] = np.zeros(shape, dtype=kspace.dtype)
    kspace_zero[partial_start:] = kspace
    return kspace_zero
