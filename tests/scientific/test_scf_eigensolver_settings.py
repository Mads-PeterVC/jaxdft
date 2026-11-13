"""The SCF's use of the eigensolver (theory post 02c, acceptance criteria 2 and 3;
engineering log, '2026-11-12: Milestone 2c public API').

- The converged state does not depend on buffer bands, on the adaptive tolerance
  ratio c, or on k-point chunking: these change how the eigenproblems are solved,
  not which problem is solved.
- The eigensolver tolerance is non-increasing, starts at eigensolver_tol_max, is
  refined at most once and then stays tight (the latch), and the step that
  converges was evaluated tightly.
- The SCF asks the eigensolver to converge only the needed bands.
"""

import functools

import numpy as np
import pytest

from jaxdft import SCFOptions, lobpcg, scf
from jaxdft.scf.scf import EIGEN_TOL_FRACTION
from support import h2 as h2_support
from support.silicon import basis_for, converged_test_size, run, si
from support.tolerances import SCF_DENSITY_ATOL, SCF_ENERGY_ATOL, SCF_TOL, SCF_TOL_TIGHT

DEFAULT_RATIO = SCFOptions().eigensolver_tol_ratio
NEW_HISTORY = ("eig_tol", "eig_iters_mean", "refined", "eig_drops", "eig_stagnated")

VARIANTS = {
    "no-buffer": {"buffer_bands": 0},
    "one-buffer": {"buffer_bands": 1},  # the default is 3
    "fixed-tolerance": {"eigensolver_tol_ratio": 0.0},
    "tenth-ratio": {"eigensolver_tol_ratio": DEFAULT_RATIO / 10},
    "k-chunks-of-3": {"k_batch_size": 3},
}


@pytest.mark.slow
@pytest.mark.parametrize("options", VARIANTS.values(), ids=VARIANTS.keys())
def test_converged_state_is_independent_of_eigensolver_settings(options):
    s, b, ref = converged_test_size()
    other = run(s, b, **options)
    np.testing.assert_allclose(float(other.total_energy), float(ref.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, ref.density, atol=SCF_DENSITY_ATOL)
    np.testing.assert_allclose(other.eigenvalues, ref.eigenvalues, atol=1e-9)


@functools.cache
def tight_run(ratio=None):
    s = si()
    b = basis_for(s)
    options = {} if ratio is None else {"eigensolver_tol_ratio": ratio}
    return run(s, b, tol=SCF_TOL_TIGHT, **options)


def test_history_has_the_eigensolver_fields():
    res = tight_run()
    n = int(res.n_iter) + 1
    for name in NEW_HISTORY:
        assert len(res.history[name]) == n, name
    assert np.all(res.history["eig_iters_mean"] <= res.history["eig_iters"])
    assert np.all(res.history["eig_drops"] >= 0)


def test_adaptive_tolerance_is_non_increasing_and_ends_tight():
    res = tight_run()
    tau = np.asarray(res.history["eig_tol"])
    refined = np.asarray(res.history["refined"])
    tight = EIGEN_TOL_FRACTION * SCF_TOL_TIGHT
    assert tau[0] == SCFOptions().eigensolver_tol_max
    assert np.all(np.diff(tau) <= 0)
    assert refined.sum() <= 1
    if refined.sum() == 1:  # the latch: tight from the refined step on
        first = int(np.argmax(refined))
        assert np.all(tau[first + 1 :] == tight)
    assert refined[-1] == 1 or tau[-1] == tight


def test_fixed_tolerance_never_refines():
    res = tight_run(ratio=0.0)
    np.testing.assert_array_equal(res.history["eig_tol"], EIGEN_TOL_FRACTION * SCF_TOL_TIGHT)
    assert np.asarray(res.history["refined"]).sum() == 0


SEEN = []


def spy_eigensolver(apply, dim, n_bands, **kwargs):
    """Records what the SCF asks for, at trace time, and solves with lobpcg."""
    SEEN.append((n_bands, kwargs.get("n_converge")))
    return lobpcg(apply, dim, n_bands, **kwargs)


def test_scf_converges_only_the_needed_bands():
    SEEN.clear()
    s = h2_support.h2(h2_support.L_SMALL)
    b = h2_support.basis_for(s, h2_support.ECUT_SMALL)
    scf(s, b, h2_support.h2_terms(), SCFOptions(tol=SCF_TOL, eigensolver=spy_eigensolver, buffer_bands=2))
    assert SEEN, "the eigensolver was not called"
    for n_solve, n_converge in SEEN:
        assert (n_solve, n_converge) == (4, 2)
