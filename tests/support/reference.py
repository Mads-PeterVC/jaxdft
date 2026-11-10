"""Loaders for reference data in reference/."""

import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


def cb1966_si_table1() -> dict:
    """Cohen & Bergstresser (1966) Table I, Si row (see the README next to the file)."""
    return json.loads((REFERENCE / "cohen_bergstresser_1966" / "si_table1.json").read_text())


def dftk_h2() -> dict:
    """DFTK reference for Milestone 2a (see reference/dftk_h2/README.md)."""
    return json.loads((REFERENCE / "dftk_h2" / "dftk_h2.json").read_text())


def libxc_lda() -> dict:
    """libxc lda_x and lda_c_pw on a density grid (see reference/libxc_lda/README.md)."""
    return json.loads((REFERENCE / "libxc_lda" / "libxc_lda.json").read_text())
