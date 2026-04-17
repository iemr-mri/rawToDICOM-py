"""Top-level pipeline configuration.

Replaces the mutable ``pm`` struct threaded through the MATLAB pipeline.
All fields are immutable after construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Paths and override flags for a full pipeline run.

    Attributes:
        raw_root:     Root directory containing raw Bruker session folders
                      (equivalent to MATLAB ``pm.rawRoot``).
        sorted_root:  Root directory where scans are sorted into keyword
                      subdirectories (equivalent to ``pm.sortedRoot``).
        dicom_root:   Root directory where DICOM output is written
                      (equivalent to ``pm.DICOMRoot``).
        project:      Project name — used as the first level of subdirectory
                      under ``sorted_root`` and ``dicom_root``.
        cohort:       Cohort path fragment appended below ``project``.
        skip_sort:    When True, ``sort_raw_data`` is not called.  Use when
                      the sorted folder already contains the expected layout.
        force_recon:  When True, CS reconstruction runs even if output already
                      exists.
        force_dicom:  When True, DICOM files are overwritten if they exist.
        force_sg:     When True, the SG pipeline runs even if DICOM files
                      already exist for a subject.
    """

    raw_root: Path
    sorted_root: Path
    dicom_root: Path
    project: str = ""
    cohort: str = ""
    skip_sort: bool = False
    force_recon: bool = False
    force_dicom: bool = False
    force_sg: bool = False

    def cine_dir(self) -> Path:
        """Return the CINE sorted folder for this project / cohort."""
        return self.sorted_root / self.project / "CINE" / self.cohort

    def dicom_out_dir(self, subject_name: str) -> Path:
        """Return the DICOM output folder for a given subject."""
        return self.dicom_root / self.project / self.cohort / "CINE_DICOM" / subject_name
