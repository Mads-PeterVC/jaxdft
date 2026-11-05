"""Loaders for reference data in reference/."""

import json
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[2] / "reference"


def cb1966_si_table1() -> dict:
    """Cohen & Bergstresser (1966) Table I, Si row (see the README next to the file)."""
    return json.loads((REFERENCE / "cohen_bergstresser_1966" / "si_table1.json").read_text())
