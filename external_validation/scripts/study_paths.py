"""Locations of study releases supplied locally by the researcher."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
COMMUNITY_ROOT = Path(os.environ.get("AGING_CLOCK_COHORTS", PROJECT / "data" / "cohorts")).expanduser().resolve()
MIMIC_ARCHIVE = Path(os.environ.get("AGING_CLOCK_MIMIC_ARCHIVE", PROJECT / "data" / "mimic-iv-3.1.zip")).expanduser().resolve()


def community_dir(name):
    """Return a cohort directory under the configured parent."""
    return COMMUNITY_ROOT / name.upper()
