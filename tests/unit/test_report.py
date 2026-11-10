"""Log formatting and machine details (engineering log, 'SCF observability'): pure
functions from data to text, checked without running an SCF."""

import functools
import math

import jax.numpy as jnp

from jaxdft import Anderson, lobpcg
from jaxdft.output import machine_info
from jaxdft.output.report import describe, iteration_line, machine_lines
from jaxdft.pseudo.gth import GTH, hgh_lda


def test_describe_never_prints_a_memory_address():
    for obj in (lobpcg, functools.partial(lobpcg, max_iter=5), Anderson(), hgh_lda("H"), object()):
        assert " at 0x" not in describe(obj)


def test_describe_examples():
    assert describe(Anderson(depth=4, beta=0.5)) == "Anderson(depth=4, beta=0.5)"
    assert describe(functools.partial(lobpcg, max_iter=5)) == "jaxdft.solvers.lobpcg.lobpcg(max_iter=5)"
    assert describe(jnp.zeros((2, 3))) == "array(2, 3) float64"
    assert describe(jnp.asarray(1.5)) == "1.5"
    assert "source='h-q1.hgh'" in describe(hgh_lda("H"))  # file name, not the full path


def test_gth_source_is_provenance_only():
    """Parsed and hand-built pseudopotentials with the same data are equal and hash
    alike, so the source never changes a jit cache key."""
    parsed = hgh_lda("H")
    built = GTH(parsed.zion, parsed.rloc, parsed.c)
    assert parsed.source is not None and built.source is None
    assert parsed == built and hash(parsed) == hash(built)


def test_iteration_line_marks_missing_values():
    values = dict(
        energy=-1.1,
        delta_energy=math.nan,
        residual=0.1,
        projection=0.0,
        homo=-0.3,
        gap=math.nan,
        eig_residual_max=1e-11,
        eig_iters=12,
    )
    line = iteration_line(0, values, elapsed=None)
    fields = line.split()
    assert fields[0] == "0" and float(fields[1]) == -1.1
    assert fields[2] == "-" and fields[6] == "-" and fields[-1] == "-"
    assert fields[8] == "12"


def test_machine_info(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    info = machine_info()
    for key in ("time", "jaxdft", "host", "python", "jax", "jaxlib", "backend", "devices", "x64"):
        assert key in info
    assert info["x64"] is True
    assert sum(info["devices"].values()) >= 1
    assert info["environment"]["SLURM_JOB_ID"] == "12345"
    assert any("$SLURM_JOB_ID" in line and "12345" in line for line in machine_lines(info))
