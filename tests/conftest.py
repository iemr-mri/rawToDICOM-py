"""Shared pytest fixtures for the rawToDICOM test suite.

All fixtures are session-scoped so the Bruker binary files are read once per
test session.  The raw-data directory contains real Bruker ParaVision 360 data
from cohort1 of the AGORA project.
"""

from pathlib import Path

import pytest

from rawtoDICOM.bruker.reader import load_scan
from rawtoDICOM.bruker.scan import BrukerScan

# Subject used for single-scan fixtures — first subject, scan 1 (localizer)
_SUBJECT_ROOT = (
    Path(__file__).parent
    / "raw-data"
    / "AGORA"
    / "cohort1"
    / "AGORA2_F1_s_2025121703_1_4_20251217_103442"
)


@pytest.fixture(scope="session")
def scan_dir() -> Path:
    """Path to scan directory 1 of the first test subject."""
    return _SUBJECT_ROOT / "1"


@pytest.fixture(scope="session")
def bruker_scan(scan_dir: Path) -> BrukerScan:
    """Fully loaded BrukerScan for scan 1 of the first test subject."""
    return load_scan(scan_dir)


@pytest.fixture(scope="session")
def bruker_scan_params_only(scan_dir: Path) -> BrukerScan:
    """BrukerScan loaded without reading raw binary data."""
    return load_scan(scan_dir, read_raw=False)


@pytest.fixture(scope="session")
def all_scan_dirs() -> list[Path]:
    """All numeric scan directories in the first test subject, sorted numerically."""
    return sorted(
        (p for p in _SUBJECT_ROOT.iterdir() if p.name.isdigit()),
        key=lambda p: int(p.name),
    )


@pytest.fixture(scope="session")
def cs_scan_dir() -> Path:
    """Path to scan 15 (segFLASH_CS) — used to exercise the CS undersampled path."""
    return _SUBJECT_ROOT / "15"


@pytest.fixture(scope="session")
def cs_bruker_scan(cs_scan_dir: Path) -> BrukerScan:
    """Fully loaded BrukerScan for a CS-undersampled scan."""
    return load_scan(cs_scan_dir)


@pytest.fixture(scope="session")
def all_subjects() -> list[Path]:
    """All subject directories in cohort1."""
    cohort1 = Path(__file__).parent / "raw-data" / "AGORA" / "cohort1"
    return sorted(cohort1.iterdir())
