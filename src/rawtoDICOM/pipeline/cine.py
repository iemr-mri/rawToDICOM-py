"""CINE pipeline orchestration.

Translates createDICOMCine.m.

Pipeline per scan:
  1. load_scan          — read Bruker files into BrukerScan
  2. sort_kspace        — reshape raw data → [x, y, slices, frames, flow_enc, coils]
  3. reconstruct_cs     — (CS scans only) fill in missing k-space lines
  4. zero_fill_kspace   — 2× spatial zero-padding
  5. combine_coils      — sum-of-squares over coil axis → [x, y, slices, frames]
  6. write_dicom_series — geometry corrections + DICOM write
"""

from __future__ import annotations

from pathlib import Path

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.dicom.writer import write_dicom_series
from rawtoDICOM.reconstruction.coil_combination import combine_coils
from rawtoDICOM.reconstruction.compressed_sensing import reconstruct_cs
from rawtoDICOM.reconstruction.kspace import sort_kspace, zero_fill_kspace


def process_cine_scan(
    scan_dir: Path,
    output_dir: Path,
    *,
    force_dicom: bool = False,
) -> list[Path]:
    """Run the full CINE reconstruction pipeline for one scan directory.

    Translates createDICOMCine.m.

    Args:
        scan_dir:    Path to a single Bruker numeric scan directory containing
                     ``acqp``, ``method``, and ``rawdata.job0`` (or ``fid``).
        output_dir:  Destination directory for DICOM files.  Created if absent.
        force_dicom: When True, existing DICOM files are overwritten.

    Returns:
        List of written DICOM file paths, one per slice.
    """
    output_dir = Path(output_dir)

    if not force_dicom and output_dir.exists() and any(output_dir.glob("*.dcm")):
        existing = sorted(output_dir.glob("*.dcm"))
        return existing

    scan = load_scan(scan_dir)

    # [x, y, slices, frames, flow_enc_dir, coils]
    kspace = sort_kspace(scan)

    # squeeze flow_enc_dir axis (always 1 for CINE) → [x, y, slices, frames, coils]
    kspace_squeezed = kspace[:, :, :, :, 0, :]

    if "CSPhaseEncList" in scan.method:
        kspace_squeezed = reconstruct_cs(kspace_squeezed)

    kspace_padded = zero_fill_kspace(kspace_squeezed)

    # combine_coils expects [x, y, slices, frames, coils]
    images = combine_coils(kspace_padded)  # [x, y, slices, frames]

    return write_dicom_series(images, scan, output_dir)
