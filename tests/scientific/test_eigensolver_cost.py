"""Eigensolver work in the SCF (theory post 02c, acceptance criteria 6 and 7;
engineering log, '2026-11-12: Milestone 2c public API').

Silicon on the 3^3 grid at 12 Ha: the 2^3 test grid cannot see the stopping fix
(193 -> 175), because with 8 k-points the maximum over k is close to the mean. The
counts are the SCF history's eig_iters (max over k per step) and eig_iters_mean,
summed over the SCF; with refinement a step counts both diagonalisations.
"""

import functools

import numpy as np
import pytest

from jaxdft import SCFOptions
from support.silicon import basis_for, run, si
from support.tolerances import (
    COST_3X3X3_SUM_MAX_ITERS_TODAY,
    COST_3X3X3_SUM_MEAN_ITERS_TODAY,
    COST_MAX_ITERS_FRACTION,
    COST_MEAN_ITERS_FRACTION,
    SCF_TOL,
    SCF_TOL_TIGHT,
)

GRID = (3, 3, 3)
DEFAULT_RATIO = SCFOptions().eigensolver_tol_ratio


@functools.cache
def history(tol, ratio=None):
    s = si()
    b = basis_for(s, kshape=GRID)
    options = {} if ratio is None else {"eigensolver_tol_ratio": ratio}
    return run(s, b, tol=tol, **options).history


@pytest.mark.slow
def test_eigensolver_work_on_the_3x3x3_grid_is_within_budget():
    h = history(SCF_TOL)
    sum_max = float(np.sum(h["eig_iters"]))
    sum_mean = float(np.sum(h["eig_iters_mean"]))
    assert sum_max <= COST_MAX_ITERS_FRACTION * COST_3X3X3_SUM_MAX_ITERS_TODAY, sum_max
    assert sum_mean <= COST_MEAN_ITERS_FRACTION * COST_3X3X3_SUM_MEAN_ITERS_TODAY, sum_mean


@pytest.mark.slow
def test_a_tight_ratio_at_a_tight_tolerance_costs_no_more_than_a_fixed_tolerance():
    """The scratch-check row the theory post could not explain: tol 1e-12 with c a
    tenth of the default needed 229 iterations against 89, and 1040 (three steps at
    the cap) before the penalty change. With the stagnation stop, the best iterate
    and an attainable floor, it must cost no more than the fixed tolerance and never
    reach the iteration cap."""
    tight = history(SCF_TOL_TIGHT, ratio=DEFAULT_RATIO / 10)
    fixed = history(SCF_TOL_TIGHT, ratio=0.0)
    assert np.max(tight["eig_iters"]) < 300
    assert np.sum(tight["eig_iters"]) <= np.sum(fixed["eig_iters"])
