"""Command-line entry point for the rawToDICOM pipeline.

Translates rawToDICOM.m.

The pipeline automatically:
  - Sorts raw scans that have not been sorted yet (skips existing destinations).
  - Converts sorted scans to DICOM where output does not already exist.
  - Detects self-gating (SG) vs. CINE scans and routes them accordingly.

Use ``--skip-sort`` if data is already sorted, ``--force-dicom`` to overwrite
existing DICOM files.

Usage example::

    python -m rawtoDICOM \\
        --raw-root /data/raw \\
        --sorted-root /data/sorted \\
        --dicom-root /data/dicom \\
        --project AGORA \\
        --cohort cohort1

    # Skip sorting and force DICOM overwrite:
    python -m rawtoDICOM \\
        --raw-root /data/raw \\
        --sorted-root /data/sorted \\
        --dicom-root /data/dicom \\
        --project AGORA \\
        --cohort cohort1 \\
        --skip-sort \\
        --force-dicom
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rawtoDICOM.bruker.reader import is_sg_scan, load_scan
from rawtoDICOM.config import PipelineConfig
from rawtoDICOM.pipeline.cine import process_cine_scan
from rawtoDICOM.pipeline.self_gating import process_sg_scan
from rawtoDICOM.pipeline.sort import sort_raw_data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rawtoDICOM",
        description=(
            "Convert raw Bruker MRI data to DICOM files. "
            "Sorts unsorted scans and converts to DICOM where output is missing."
        ),
    )
    parser.add_argument("--raw-root", required=True, type=Path, help="Root of raw Bruker data.")
    parser.add_argument(
        "--sorted-root", required=True, type=Path, help="Root for sorted output."
    )
    parser.add_argument("--dicom-root", required=True, type=Path, help="Root for DICOM output.")
    parser.add_argument("--project", default="", help="Project name (subdirectory level).")
    parser.add_argument("--cohort", default="", help="Cohort path fragment.")
    parser.add_argument(
        "--species", default="rat", choices=["rat", "mouse"], help="Species (used by SG pipeline)."
    )
    parser.add_argument("--skip-sort", action="store_true", help="Skip sort step.")
    parser.add_argument("--force-recon", action="store_true", help="Force CS reconstruction.")
    parser.add_argument(
        "--force-dicom", action="store_true", help="Overwrite existing DICOM files."
    )
    return parser


def _process_scan(
    scan_dir: Path,
    output_dir: Path,
    *,
    species: str,
    force_dicom: bool,
) -> list[Path]:
    """Detect scan type and run the appropriate pipeline."""
    meta = load_scan(scan_dir, read_raw=False)
    if is_sg_scan(meta):
        return process_sg_scan(scan_dir, output_dir, species=species, force_dicom=force_dicom)
    return process_cine_scan(scan_dir, output_dir, force_dicom=force_dicom)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``rawtoDICOM`` CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        raw_root=args.raw_root,
        sorted_root=args.sorted_root,
        dicom_root=args.dicom_root,
        project=args.project,
        cohort=args.cohort,
        skip_sort=args.skip_sort,
        force_recon=args.force_recon,
        force_dicom=args.force_dicom,
    )

    if not config.skip_sort:
        print("Sorting raw data …")
        copied = sort_raw_data(config)
        total = sum(len(v) for v in copied.values())
        print(f"  {total} new scan(s) sorted.")

    cine_dir = config.cine_dir()
    if not cine_dir.exists():
        print(f"No CINE subjects found under {cine_dir}.", file=sys.stderr)
        sys.exit(1)

    subjects = sorted(d for d in cine_dir.iterdir() if d.is_dir())
    if not subjects:
        print(f"No subjects in {cine_dir}.", file=sys.stderr)
        sys.exit(1)

    for subject_dir in subjects:
        print(f"Processing {subject_dir.name} …")
        scan_dirs = sorted(
            (d for d in subject_dir.iterdir() if d.is_dir() and d.name.isdigit()),
            key=lambda p: int(p.name),
        )
        for scan_dir in scan_dirs:
            out_dir = config.dicom_out_dir(subject_dir.name) / scan_dir.name
            written = _process_scan(
                scan_dir,
                out_dir,
                species=args.species,
                force_dicom=config.force_dicom,
            )
            status = "skipped (exists)" if not written else f"{len(written)} file(s) written"
            print(f"  scan {scan_dir.name}: {status}")

    print("Done.")


if __name__ == "__main__":
    main()
