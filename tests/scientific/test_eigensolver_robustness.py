"""LOBPCG at the limit of its accuracy (theory post 02c, 'Robust stopping', acceptance
criterion 5; engineering log, '2026-11-12: Milestone 2c public API').

The problems of the per-iteration diagnosis (reference/gpu_lobpcg_floor/): H2 at
Gamma at its test size, and silicon at k-point 1 of the 2^3 grid at 12 Ha, each with
the Hamiltonian at its converged density and the SCF's preconditioner.

- The floor tau_floor = EIGEN_FLOOR_FACTOR eps_mach max(diag H) over the physical
  plane waves is reached, within the factor the SCF's eig_ok check allows, from
  every start.
- Past its attainable residual, silicon on the CPU climbs by orders of magnitude
  (11 of 12 forced runs in the diagnosis). The returned block must be the best seen,
  so the residual of what is returned never grows with max_iter, and an unreachable
  tolerance ends by the stagnation stop with a block at the floor.
"""

import functools
import itertools

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import apply_H, hamiltonian, lobpcg
from jaxdft.hamiltonian.preconditioners import diagonal_preconditioner
from jaxdft.scf.scf import EIGEN_FLOOR_FACTOR
from support import h2 as h2_support
from support import silicon as si_support
from support.tolerances import EIGEN_CHECK_FLOOR_MULTIPLE, STAGNATION_MAX_ITERS

SEEDS = (0, 1, 2)
MAX_ITERS = (40, 50, 60, 80, 100, 150)


@functools.cache
def problem(name):
    """(ham, n_pad, n_bands needed, floor) at the converged density."""
    if name == "H2":
        s, b, res = h2_support.converged_test_size()
        terms, ik = h2_support.h2_terms(), 0
    else:
        s, b, res = si_support.converged_test_size()
        terms, ik = si_support.si_terms(), 1
    ham = hamiltonian(s, b[ik], terms, density=res.density)
    n_bands = res.eigenvalues.shape[-1]
    dmax = float(jnp.max(jnp.where(ham.mask, ham.diagonal(), -jnp.inf)))
    floor = EIGEN_FLOOR_FACTOR * np.finfo(np.float64).eps * dmax
    return ham, b[ik].n_pad, n_bands, floor


def solve(name, seed, tol, max_iter=300):
    ham, n_pad, n_bands, _ = problem(name)

    def apply(p):
        return apply_H(ham, p)

    eps, X, info = lobpcg(
        apply,
        n_pad,
        n_bands + 1,
        preconditioner=diagonal_preconditioner(ham),
        tol=tol,
        n_converge=n_bands,
        max_iter=max_iter,
        seed=seed,
        return_info=True,
    )
    X = jnp.where(ham.mask[:, None], X, 0)
    r = jnp.linalg.norm(apply(X) - X * eps[None, :], axis=0)
    return float(jnp.max(r[:n_bands])), info


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", ["H2", "Si"])
def test_the_floor_is_reached(name, seed):
    *_, floor = problem(name)
    residual, _ = solve(name, seed, tol=floor)
    assert residual < EIGEN_CHECK_FLOOR_MULTIPLE * floor, (residual, floor)


@pytest.mark.slow
@pytest.mark.parametrize("seed", SEEDS)
def test_the_best_block_is_returned_past_the_attainable_residual(seed):
    *_, floor = problem("Si")
    residuals = [solve("Si", seed, tol=0.0, max_iter=m)[0] for m in MAX_ITERS]
    # Iterates are ranked by LOBPCG's in-loop residual, which drifts from the
    # true one by ~1.5e-14 on a V100 (measured, aicloud 2c test run; 1e-14 was
    # too tight). The climbs this test exists for reach ~1e-8.
    for earlier, later in itertools.pairwise(residuals):
        assert later <= earlier + floor, residuals


@pytest.mark.parametrize("seed", SEEDS)
def test_unreachable_tolerance_ends_by_stagnation_at_the_floor(seed):
    *_, floor = problem("Si")
    residual, info = solve("Si", seed, tol=0.0)
    assert bool(info.stagnated)
    assert int(info.iterations) < STAGNATION_MAX_ITERS
    assert residual < EIGEN_CHECK_FLOOR_MULTIPLE * floor, (residual, floor)
