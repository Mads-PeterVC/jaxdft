"""Kohn-Sham bands of silicon from the converged density (theory post 02b, 'Bands from
the converged density', 'What the results should look like', 'Ghost states').

At the test size (2^3 density, 12 Ha) only structural checks are made here; the
comparison with Cohen-Bergstresser and the published LDA values belongs to the
validation document, on converged grids.
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import PlaneWaveBasis, band_structure, special_points_fcc
from jaxdft.analysis.bands import band_minimum_on_segment
from jaxdft.analysis.symmetry import group_levels
from support.silicon import ECUT_TEST, HARTREE_IN_EV, basis_for, converged_test_size, run, si, si_terms
from support.tolerances import BANDS_VS_SCF_ATOL, EXACT_ATOL, INTERIOR_MARGIN, SCF_TOL_TIGHT

POINTS = special_points_fcc()


@functools.cache
def tight():
    s = si()
    b = basis_for(s)
    return s, b, run(s, b, tol=SCF_TOL_TIGHT)


@functools.cache
def levels_at_special_points():
    s, _b, res = converged_test_size()
    k = jnp.stack([POINTS["G"], POINTS["X"], POINTS["L"]])
    basis = PlaneWaveBasis.build(s.lattice, k, ECUT_TEST)
    bands = band_structure(s, basis, n_bands=8, terms=si_terms(), density=res.density)
    return np.asarray(bands.eigenvalues)


def test_band_structure_at_grid_k_points_equals_the_scf_eigenvalues():
    """H[rho*](k) at a k-point of the SCF grid is the Hamiltonian the SCF diagonalised
    (an exact identity; at SCF_TOL_TIGHT the last input and the returned density
    coincide to far below the tolerance)."""
    s, b, res = tight()
    bands = band_structure(s, b, n_bands=5, terms=si_terms(), density=res.density)
    np.testing.assert_allclose(bands.eigenvalues, res.eigenvalues, atol=BANDS_VS_SCF_ATOL)


def test_degeneracies_at_gamma_x_and_l_are_exact():
    """1, 3, 3, 1 at Gamma; pairs at X; 1, 1, 2, 1, 2 at L (the first seven bands),
    each level degenerate to the exact tolerance on the even grid."""
    e = levels_at_special_points()
    for row, n, expected in ((0, 8, [1, 3, 3, 1]), (1, 8, [2, 2, 2, 2]), (2, 7, [1, 1, 2, 1, 2])):
        got = [deg for _e, deg in group_levels(e[row, :n], atol=EXACT_ATOL)]
        assert got == expected, (row, got)
        assert [deg for _e, deg in group_levels(e[row, :n], atol=1e-4)] == expected


def test_no_ghost_states():
    """Nothing below the 3s-derived band (theory post, 'Ghost states'): the lowest
    eigenvalue over the grid is Gamma_1, a single level, followed by the Gamma_25'
    triplet that is the valence-band maximum; the valence width is that of silicon
    (review: 11.98 eV on the 4^3 grid)."""
    _s, _b, res = converged_test_size()
    e = np.asarray(res.eigenvalues)
    assert int(np.argmin(e[:, 0])) == 0  # Gamma is k-point 0
    gamma = levels_at_special_points()[0]
    np.testing.assert_allclose(gamma[0], e[:, 0].min(), atol=1e-8)
    width = (gamma[3] - gamma[0]) * HARTREE_IN_EV
    assert 11.0 < width < 13.0, width
    np.testing.assert_allclose(gamma[3], e[:, 3].max(), atol=1e-8)  # VBM at Gamma


@pytest.mark.slow
def test_conduction_minimum_is_interior_to_gamma_x():
    """The Delta minimum lies inside Gamma-X (about 0.85 of the way) and below the
    conduction level at X (theory post; 1b)."""
    s, _b, res = converged_test_size()
    t, e_min = band_minimum_on_segment(
        s, POINTS["G"], POINTS["X"], band=4, ecut=ECUT_TEST, terms=si_terms(), density=res.density
    )
    assert 0.7 < t < 0.95, t
    x_level = levels_at_special_points()[1, 4]
    assert e_min < x_level - INTERIOR_MARGIN
