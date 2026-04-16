"""BrukerScan dataclass — pure data container for one Bruker ParaVision scan directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy.typing as npt


@dataclass(frozen=True)
class BrukerScan:
    """Immutable container for one Bruker ParaVision scan.

    Replaces RawDataObject / BrukerDataSuperclass from the MATLAB pipeline.
    All paths are derived from scan_dir at load time; nothing is mutated after
    construction.

    Attributes:
        acqp:      Acquisition parameters (replaces obj.Acqp).
        method:    Sequence / method parameters (replaces obj.Method).
        visu_pars: Visualization / DICOM metadata parameters.
        data:      Raw k-space arrays. data[0] = fid (pre-PV360) or job0
                   (PV360+); data[1+] = additional job files.
                   Each array has shape [coils, x_points, acquisitions].
        scan_dir:  Root ParaVision scan directory (replaces obj.Filespath).
    """

    acqp: dict[str, Any]
    method: dict[str, Any]
    visu_pars: dict[str, Any]
    data: list[npt.NDArray[Any]] = field(default_factory=list)
    scan_dir: Path = field(default_factory=Path)
