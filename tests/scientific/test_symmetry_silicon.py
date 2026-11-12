"""Symmetry of the discrete silicon problem (theory post 02b, 'Grid symmetry', 'What a
shifted grid breaks'; review findings 1 and 2; decisions 3, 6 and 8).

The FFT grid carries the diamond space group if and only if N is even. N = 26 is the
discriminating case (even, not a multiple of 4); N = 25 is the negative control.
"""

import numpy as np
import pytest

from support.silicon import (
    ECUT_TEST,
    N_TEST,
    basis_for,
    forces,
    maps_grid,
    run,
    si,
    space_group,
    transform_grid,
    with_positions,
)
from support.tolerances import (
    EGGBOX_HALF_SPACING_MIN,
    ODD_GRID_FORCE_WINDOW,
    SCF_DENSITY_ATOL,
    SCF_ENERGY_ATOL,
    SHIFTED_GRID_FORCE_MIN,
    SYMMETRY_FORCE_ATOL,
)

pytestmark = pytest.mark.slow


def _grid(n):
    return (n, n, n)


@pytest.fixture(scope="module", params=[24, 26], ids=["N24", "N26"])
def even_grid_run(request):
    s = si()
    b = basis_for(s, fft_shape=_grid(request.param))
    return s, b, run(s, b)


@pytest.fixture(scope="module")
def odd_grid_run():
    s = si()
    b = basis_for(s, fft_shape=_grid(25))
    return s, b, run(s, b)


def test_even_grid_density_has_the_full_space_group(even_grid_run):
    s, _b, res = even_grid_run
    rho = np.asarray(res.density)
    ops = space_group(s)
    assert len(ops) == 48 and all(maps_grid(op, rho.shape) for op in ops)
    for op in ops:
        np.testing.assert_allclose(transform_grid(rho, op), rho, atol=SCF_DENSITY_ATOL)


def test_even_grid_forces_vanish_at_the_ideal_geometry(even_grid_run):
    s, b, _ = even_grid_run
    F = forces(s, b)
    assert np.abs(F).max() < SYMMETRY_FORCE_ATOL


def test_odd_grid_breaks_the_tetrahedral_operations_at_the_egg_box_level(odd_grid_run):
    """Inversion maps every grid onto itself, so the net force still vanishes; the
    largest force is at the egg-box level, not zero (measured 6e-7 Ha/Bohr)."""
    s, b, res = odd_grid_run
    rho = np.asarray(res.density)
    inversion = [op for op in space_group(s) if np.array_equal(op[0], -np.eye(3, dtype=int)) and np.allclose(op[1], 0)]
    assert len(inversion) == 1
    np.testing.assert_allclose(transform_grid(rho, inversion[0]), rho, atol=SCF_DENSITY_ATOL)
    F = forces(s, b)
    assert np.abs(F.sum(axis=0)).max() < SYMMETRY_FORCE_ATOL
    lo, hi = ODD_GRID_FORCE_WINDOW
    assert lo < np.abs(F).max() < hi


def test_half_shifted_grid_breaks_the_symmetry_of_the_density():
    """Positive control of the Gamma-centred choice (theory post; measured 1.0e-3
    Ha/Bohr): without density symmetrisation the shifted grid gives forces at the
    ideal geometry."""
    s = si()
    b = basis_for(s, shift=(0.5, 0.5, 0.5))
    assert np.abs(forces(s, b)).max() > SHIFTED_GRID_FORCE_MIN


def test_translation_by_a_whole_grid_spacing_is_exact_and_by_half_is_not():
    """Egg-box at 12 Ha (theory post: 2.0e-7 Ha for half a spacing, 2e-15 for a whole
    one). XC quadrature is the only term that is not translation-covariant."""
    s = si()
    b = basis_for(s)
    assert b.fft_shape == _grid(N_TEST)
    e0 = float(run(s, b, tol=1e-10).total_energy)
    f = np.asarray(s.positions)
    whole = float(run(with_positions(s, f + np.array([1.0, 0, 0]) / N_TEST), b, tol=1e-10).total_energy)
    half = float(run(with_positions(s, f + np.array([0.5, 0, 0]) / N_TEST), b, tol=1e-10).total_energy)
    assert abs(whole - e0) < SCF_ENERGY_ATOL
    assert abs(half - e0) > EGGBOX_HALF_SPACING_MIN


def test_compatible_grid_option_makes_the_symmetry_exact_at_15_ha():
    """Decision 3: the default rule gives N = 25 at 15 Ha; fft_multiple=2 gives 30,
    on which the forces at the ideal geometry vanish to the exact tolerance."""
    s = si()
    b = basis_for(s, ecut=15.0, fft_multiple=2)
    assert b.fft_shape == _grid(30)
    assert np.abs(forces(s, b)).max() < SYMMETRY_FORCE_ATOL


def test_default_grid_at_the_test_cutoff_is_even():
    assert basis_for(si(), ECUT_TEST).fft_shape == _grid(24)
