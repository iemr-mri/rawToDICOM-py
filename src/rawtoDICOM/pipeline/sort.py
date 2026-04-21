"""Raw data sorter.

Translates sortRawData.m.

Walks the raw directory tree and copies each numeric scan directory into a
keyword-named subdirectory under ``sorted_root``.  The keyword is derived from
the ``ACQ_scan_name`` field of the scan's ``acqp`` file.  Scans that do not
match any known keyword are skipped.

The MATLAB original uses ``copyfile`` per scan; this implementation copies with
``shutil.copytree`` so the full Bruker directory structure (pdata, acqp, method,
rawdata.job*, …) is preserved verbatim.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from rawtoDICOM.bruker.params import parse_bruker_params
from rawtoDICOM.config import PipelineConfig

# Maps match pattern (case-insensitive substring of ACQ_scan_name) → destination
# folder name under sorted_root.  The first matching pattern wins.
# "segFLASH_CS" is listed before "CINE" so it takes priority over the shorter match.
_KEYWORD_FOLDER: dict[str, str] = {
    "segFLASH_CS": "CINE",
    "CINE": "CINE",
    "TPM": "TPM",
    "t1": "t1",
    "MRE": "MRE",
    "LGE": "LGE",
    "tagged": "tagged",
}


def sort_raw_data(config: PipelineConfig) -> dict[str, list[Path]]:
    """Walk raw directories and copy scans into keyword subdirectories.

    Translates sortRawData.m.

    For each subject directory under ``config.raw_root``, numeric scan
    sub-directories are inspected.  The ``ACQ_scan_name`` field determines which
    keyword folder the scan is copied into.  If the destination already exists
    the copy is skipped (no overwrite).

    Args:
        config: Pipeline configuration supplying ``raw_root`` and ``sorted_root``.

    Returns:
        Dictionary mapping keyword → list of destination paths that were copied.
    """
    unique_folders = dict.fromkeys(_KEYWORD_FOLDER.values())
    copied: dict[str, list[Path]] = {folder: [] for folder in unique_folders}

    raw_root = Path(config.raw_root)
    subject_root = raw_root / config.project / config.cohort
    if not subject_root.exists():
        raise FileNotFoundError(f"raw subject directory does not exist: {subject_root}")

    for subject_dir in sorted(subject_root.iterdir()):
        if not subject_dir.is_dir():
            continue

        scan_dirs = sorted(
            subject_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1
        )
        for scan_dir in scan_dirs:
            if not scan_dir.name.isdigit():
                continue

            acqp_path = scan_dir / "acqp"
            if not acqp_path.exists():
                continue

            try:
                acqp = parse_bruker_params(acqp_path)
            except Exception:
                continue

            scan_name = str(acqp.get("ACQ_scan_name", "")).strip("<>")

            keyword = _match_keyword(scan_name)
            if keyword is None:
                continue

            dest = (
                config.sorted_root
                / config.project
                / keyword
                / config.cohort
                / subject_dir.name
                / scan_name
            )

            if dest.exists():
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(str(scan_dir), str(dest))
            copied[keyword].append(dest)

    return copied


def _match_keyword(scan_name: str) -> str | None:
    """Return the destination folder for the first matching pattern, or None."""
    lower = scan_name.lower()
    for pattern, folder in _KEYWORD_FOLDER.items():
        if pattern.lower() in lower:
            return folder
    return None
