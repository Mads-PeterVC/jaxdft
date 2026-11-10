"""The example scripts run as documented (the 2a validation post shows the H2 one)."""

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def run_example(name, *args):
    return subprocess.run(
        [sys.executable, str(EXAMPLES / name), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )


@pytest.mark.slow
def test_h2_single_point_writes_a_log(tmp_path):
    log = tmp_path / "h2.log"
    run = run_example("h2_single_point.py", log)
    assert run.stdout.startswith("E = -1.10488")
    text = log.read_text()
    assert "converged in" in text and "== Result" in text


@pytest.mark.slow
def test_h3_single_point_half_fills_the_second_level():
    """Three electrons, spin-unpolarised: occupations (2, 1, 0) and a converged SCF.
    No reference energy: this checks that an odd electron count runs end to end."""
    run = run_example("h3_single_point.py")
    out = run.stdout
    assert "occupations = [2. 1. 0.]" in out
    assert "== SCF iterations" in out and "converged in" in out
    assert "int rho dr = 3.000000000000" in out
