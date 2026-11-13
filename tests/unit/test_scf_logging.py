"""SCF observability (engineering log, 'SCF observability'): the per-iteration history
in SCFResult and SCFError, and the log written when the ``jaxdft`` logger is enabled.
Logging must not change the calculation."""

import logging
import re

import numpy as np
import pytest

import jaxdft
from jaxdft import SCFError, SCFOptions, scf
from support.h2 import converged_small, h2_terms
from support.tolerances import SCF_TOL

ITERATION = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d{10})\s")
HISTORY_KEYS = {
    "energy",
    "delta_energy",
    "residual",
    "projection",
    "homo",
    "lumo",
    "gap",
    "eig_residual_max",
    "eig_iters",
    # Milestone 2c (engineering log, '2026-11-12: Milestone 2c public API')
    "eig_iters_mean",
    "eig_tol",
    "refined",
    "eig_drops",
    "eig_stagnated",
}


def logged_run(caplog, options):
    s, b, _ = converged_small()
    with caplog.at_level(logging.INFO, logger=jaxdft.output.LOGGER_NAME):
        res = scf(s, b, h2_terms(), options)
    return res, [r.getMessage() for r in caplog.records]


def test_history_is_consistent_with_the_result():
    _, _, res = converged_small()
    h = {k: np.asarray(v) for k, v in res.history.items()}
    n = int(res.n_iter)
    assert set(h) == HISTORY_KEYS
    assert all(v.shape == (n + 1,) for v in h.values())
    assert h["residual"][-1] == float(res.residual)
    assert h["residual"][-1] < SCF_TOL <= h["residual"][-2]
    # The last recorded energy is the direct functional of the returned orbitals.
    np.testing.assert_allclose(h["energy"][-1], float(res.total_energy), rtol=0, atol=1e-12)
    assert np.isnan(h["delta_energy"][0])
    np.testing.assert_allclose(h["delta_energy"][1:], np.diff(h["energy"]), rtol=0, atol=1e-15)
    eps = np.asarray(res.eigenvalues[0])
    np.testing.assert_allclose([h["homo"][-1], h["lumo"][-1]], eps, rtol=0, atol=1e-14)
    assert np.all(h["eig_iters"] >= 0) and np.all(h["eig_residual_max"] >= 0)


def test_the_log_covers_setup_iterations_and_result(caplog):
    res, lines = logged_run(caplog, SCFOptions(tol=SCF_TOL))
    text = "\n".join(lines)
    for title in ("Machine", "Structure", "Model", "Basis", "Electrons and SCF settings", "SCF iterations"):
        assert f"== {title}" in text
    for setting in ("Anderson(depth=6, beta=0.7)", "lobpcg", "AtomicGaussians(width=0.8)", "Ecut", "1.0e-09"):
        assert setting in text
    rows = [ITERATION.match(line) for line in lines]
    rows = [(int(m[1]), float(m[2])) for m in rows if m]
    n = int(res.n_iter)
    assert [i for i, _ in rows] == list(range(n + 1))
    np.testing.assert_allclose([e for _, e in rows], np.asarray(res.history["energy"]), rtol=0, atol=6e-11)
    assert f"converged in {n} iterations" in text
    total = next(line for line in lines if line.strip().startswith("total ") and "Ha" not in line)
    assert float(total.split()[-1]) == pytest.approx(float(res.total_energy), abs=1e-12)
    assert "== Timing" in text and "compile" in text


def test_logging_does_not_change_the_calculation(caplog):
    _, _, quiet = converged_small()
    loud, _ = logged_run(caplog, SCFOptions(tol=SCF_TOL))
    assert int(loud.n_iter) == int(quiet.n_iter)
    assert float(loud.total_energy) == pytest.approx(float(quiet.total_energy), abs=1e-14)
    np.testing.assert_allclose(loud.density, quiet.density, rtol=0, atol=1e-14)


def test_nothing_is_logged_unless_enabled(caplog):
    s, b, _ = converged_small()
    with caplog.at_level(logging.WARNING, logger=jaxdft.output.LOGGER_NAME):
        scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL))
    assert not [r for r in caplog.records if r.name.startswith("jaxdft")]


def test_a_failed_scf_keeps_its_history_and_logs_why(caplog):
    s, b, _ = converged_small()
    with caplog.at_level(logging.INFO, logger=jaxdft.output.LOGGER_NAME):
        with pytest.raises(SCFError) as err:
            scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL, max_iter=2))
    assert err.value.n_iter == 2
    assert {k: np.asarray(v).shape for k, v in err.value.history.items()} == {k: (3,) for k in HISTORY_KEYS}
    assert any("FAILED: SCF not converged after 2 iterations" in r.getMessage() for r in caplog.records)
