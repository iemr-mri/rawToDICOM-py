"""Binary k-space reader for Bruker ParaVision scan directories.

Translates readBrukerRaw.m — reads rawdata.job* (PV360+) and fid (pre-PV360)
binary files into complex numpy arrays of shape [coils, x_points, acquisitions].
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt


def read_raw_data(
    acqp: dict[str, Any],
    scan_dir: Path,
    precision: str = "double",
    specified_nrs: list[int] | None = None,
    specified_jobs: list[int] | None = None,
) -> list[npt.NDArray[Any]]:
    """Read binary k-space data from a Bruker scan directory.

    Routes to the PV360+ path when ACQ_ScanPipeJobSettings is present,
    otherwise uses the pre-PV360 fid path.

    Returns:
        List of complex numpy arrays, one per job (or one fid for pre-PV360).
        Each array has shape [coils, x_points, acquisitions].
    """
    if "ACQ_ScanPipeJobSettings" in acqp:
        return _read_jobs_pv360(acqp, scan_dir, precision, specified_jobs)
    return _read_fid_pre_pv360(acqp, scan_dir, precision, specified_nrs, specified_jobs)


# ---------------------------------------------------------------------------
# PV360+ reader
# ---------------------------------------------------------------------------


def _read_jobs_pv360(
    acqp: dict[str, Any],
    scan_dir: Path,
    precision: str,
    specified_jobs: list[int] | None,
) -> list[npt.NDArray[Any]]:
    """Read rawdata.job* files for PV360+ acquisitions.

    PV360 always stores complex (real/imag interleaved along x-axis).
    """
    jobs_size: int = int(acqp["ACQ_jobs_size"])
    job_indices = list(range(jobs_size)) if specified_jobs is None else specified_jobs

    pipe_settings: npt.NDArray[Any] = acqp["ACQ_ScanPipeJobSettings"]
    acq_jobs: npt.NDArray[Any] = acqp["ACQ_jobs"]
    bytorda: str = str(acqp["BYTORDA"])
    endian: Literal["<", ">"] = "<" if bytorda.lower() == "little" else ">"

    output: list[npt.NDArray[Any]] = [np.array([])] * jobs_size

    for job_idx in job_indices:
        store_mode = str(pipe_settings[0, job_idx])
        if "discard" in store_mode.lower():
            continue

        data_format = str(pipe_settings[1, job_idx])
        dtype = _pv360_dtype(data_format, endian, precision)

        scan_size = int(acq_jobs[0, job_idx])
        job_name = str(acq_jobs[8, job_idx])
        n_receivers = _count_receivers_pv360(acqp, job_idx)

        job_path = scan_dir / f"rawdata.{job_name}"
        if not job_path.exists():
            raise FileNotFoundError(f"Job file not found: {job_path}")

        output[job_idx] = _read_job_file(job_path, dtype, scan_size, n_receivers)

    return [arr for arr in output if arr.size > 0]


def _pv360_dtype(
    data_format: str,
    endian: Literal["<", ">"],
    precision: str,
) -> np.dtype[Any]:
    """Map ACQ_ScanPipeJobSettings format string to a numpy dtype."""
    fmt = data_format.upper()
    base: type[np.generic]
    if "32BIT_SIGNED" in fmt or "32BIT_SGN" in fmt:
        base = np.int32
    elif "64BIT_FLOAT" in fmt:
        base = np.float64
    else:
        base = np.int32  # safe default

    if precision == "single":
        base = np.float32

    return np.dtype(base).newbyteorder(endian)


def _count_receivers_pv360(acqp: dict[str, Any], job_idx: int) -> int:
    """Count active receive channels for a given job (PV360+ format)."""
    receiver_select: npt.NDArray[Any] = acqp["ACQ_ReceiverSelectPerChan"]
    if receiver_select.ndim == 1:
        row = receiver_select
    else:
        row = receiver_select[job_idx]
    return int(sum(1 for r in row if str(r).startswith("Yes")))


def _read_job_file(
    path: Path,
    dtype: np.dtype[Any],
    scan_size: int,
    n_receivers: int,
) -> npt.NDArray[Any]:
    """Read one rawdata.job* file and return complex [coils, x_points, acquisitions]."""
    raw: npt.NDArray[Any] = np.fromfile(path, dtype=dtype)
    n_total = n_receivers * scan_size
    if n_total == 0 or len(raw) < n_total:
        raise ValueError(f"File {path} too small: expected ≥{n_total}, got {len(raw)}")

    n_acq = len(raw) // n_total
    raw = raw[: n_acq * n_total]

    # MATLAB: fread([nRecs*scanSize, inf]) → reshape([scanSize, nRecs, n_acq]) → permute([2,1,3])
    # Numpy Fortran-order reshape replicates MATLAB column-major reshape.
    raw_3d: npt.NDArray[Any] = raw.reshape(scan_size, n_receivers, n_acq, order="F")
    raw_3d = raw_3d.transpose(1, 0, 2)  # [n_receivers, scan_size, n_acq]

    # Convert interleaved real/imag to complex (PV360 always complex)
    return raw_3d[:, 0::2, :].astype(np.float64) + 1j * raw_3d[:, 1::2, :]


# ---------------------------------------------------------------------------
# Pre-PV360 fid reader
# ---------------------------------------------------------------------------


def _read_fid_pre_pv360(
    acqp: dict[str, Any],
    scan_dir: Path,
    precision: str,
    specified_nrs: list[int] | None,
    specified_jobs: list[int] | None,
) -> list[npt.NDArray[Any]]:
    """Read fid / job files for pre-PV360 acquisitions."""
    bytorda = str(acqp["BYTORDA"])
    endian: Literal["<", ">"] = "<" if bytorda.lower() == "little" else ">"
    fmt_str = str(acqp.get("GO_raw_data_format", "GO_32BIT_SGN_INT"))
    dtype, bytes_per = _pre_pv360_dtype(fmt_str, endian, precision)

    aq_mod = str(acqp.get("AQ_mod", "qseq"))
    is_complex = aq_mod != "qf"

    n_receivers = _count_receivers_pre_pv360(acqp)

    acq_size: npt.NDArray[Any] = np.asarray(acqp["ACQ_size"]).ravel()
    n_high = int(np.prod(acq_size[1:])) if len(acq_size) > 1 else 1
    ni = int(acqp["NI"])
    nr = int(acqp["NR"])

    block_fmt = str(acqp.get("GO_block_size", ""))
    if block_fmt == "Standard_KBlock_Format":
        block_size = (
            math.ceil(int(acq_size[0]) * n_receivers * bytes_per / 1024) * 1024 // bytes_per
        )
    else:
        block_size = int(acq_size[0]) * n_receivers

    fid_path = scan_dir / "fid"
    data = _read_fid_file(
        fid_path,
        dtype,
        block_size,
        n_high,
        ni,
        nr,
        int(acq_size[0]),
        n_receivers,
        is_complex,
        specified_nrs,
    )

    result: list[npt.NDArray[Any]] = [data]

    # Additional job files (pre-PV360 with ACQ_jobs_size > 0)
    jobs_size = int(acqp.get("ACQ_jobs_size", 0))
    if jobs_size > 0:
        job_indices = (
            list(range(jobs_size))
            if specified_jobs is None
            else [j for j in specified_jobs if j != -1]
        )
        acq_jobs: npt.NDArray[Any] = acqp["ACQ_jobs"]
        for j_idx in job_indices:
            job_scan_size = int(acq_jobs[0, j_idx])
            job_path = scan_dir / f"rawdata.job{j_idx}"
            if job_path.exists() and job_scan_size > 0:
                result.append(_read_job_file(job_path, dtype, job_scan_size, n_receivers))

    return result


def _pre_pv360_dtype(
    fmt_str: str,
    endian: Literal["<", ">"],
    precision: str,
) -> tuple[np.dtype[Any], int]:
    """Map GO_raw_data_format to (numpy dtype, bytes_per_value)."""
    fmt = fmt_str.upper()
    base: type[np.generic]
    if "32BIT_SGN_INT" in fmt or "32BIT_SGN" in fmt:
        base, bpv = np.int32, 4
    elif "16BIT_SGN_INT" in fmt:
        base, bpv = np.int16, 2
    elif "32BIT_FLOAT" in fmt:
        base, bpv = np.float32, 4
    else:
        base, bpv = np.int32, 4

    if precision == "single":
        base = np.float32

    return np.dtype(base).newbyteorder(endian), bpv


def _count_receivers_pre_pv360(acqp: dict[str, Any]) -> int:
    """Count active receive channels for pre-PV360 acquisitions."""
    for key in ("GO_selected_receivers", "ACQ_ReceiverSelect"):
        if key in acqp:
            receivers = acqp[key]
            if isinstance(receivers, np.ndarray):
                return int(sum(1 for r in receivers.ravel() if str(r).startswith("Yes")))
            if isinstance(receivers, list):
                return sum(1 for t in receivers if str(t).startswith("Yes"))
            if isinstance(receivers, str):
                return sum(1 for t in receivers.split() if t.startswith("Yes"))
    raise KeyError(
        "Cannot determine receiver count: no GO_selected_receivers or ACQ_ReceiverSelect in acqp"
    )


def _read_fid_file(
    path: Path,
    dtype: np.dtype[Any],
    block_size: int,
    n_high: int,
    ni: int,
    nr: int,
    x_size: int,
    n_receivers: int,
    is_complex: bool,
    specified_nrs: list[int] | None,
) -> npt.NDArray[Any]:
    """Read a pre-PV360 fid file and return [coils, x_points, acquisitions]."""
    total_acq = n_high * ni * nr
    raw: npt.NDArray[Any] = np.fromfile(path, dtype=dtype, count=block_size * total_acq)

    # MATLAB: fread([blockSize, total_acq]) → F-order matrix
    raw = raw.reshape(block_size, total_acq, order="F")  # [block_size, total_acq]
    raw = raw.T  # [total_acq, block_size]

    # Strip block padding
    raw = np.ascontiguousarray(raw[:, : x_size * n_receivers])  # [total_acq, x_size*nRecs]

    # MATLAB: reshape([total_acq, x_size, n_receivers]) → permute([3,2,1])
    raw = raw.reshape(total_acq, x_size, n_receivers, order="F")
    raw = raw.transpose(2, 1, 0)  # [n_receivers, x_size, total_acq]

    if is_complex:
        return raw[:, 0::2, :].astype(np.float64) + 1j * raw[:, 1::2, :]

    return raw.astype(np.float64)
