"""Brillouin-zone sampling (theory post 02b, 'Brillouin-zone sampling', 'Equivalence to
a supercell', 'Invariants and checks: k-point sampling'; decisions 4 and 5).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import (
    Kinetic,
    PlaneWaveBasis,
    band_structure,
    empty,
    fcc_lattice,
    monkhorst_pack,
    supercell,
    total_energy,
)
from support.silicon import A_SI, ECUT_TEST, basis_for, converged_displaced, run, si, si_terms
from support.tolerances import EXACT_ATOL, SCF_DENSITY_ATOL, SCF_ENERGY_ATOL

# --- the empty-lattice rung ----------------------------------------------------------


def test_empty_lattice_energy_is_the_weighted_sum_of_free_electron_bands():
    """With no potential the grid energy is sum_k w_k sum_n f_nk |k+G|^2 / 2, here with
    unequal weights so that the weights demonstrably enter."""
    lat = fcc_lattice(A_SI, unit="Bohr")
    k, _ = monkhorst_pack((2, 2, 1))
    w = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    b = PlaneWaveBasis.build(lat, k, 3.0, kweights=w)
    s = empty(lat)
    bands = band_structure(s, b, n_bands=4)
    f = jnp.asarray([[2.0, 2.0, 1.0, 0.0]] * 4)
    e, parts = total_energy(s, b, (Kinetic(),), bands.eigenvectors, f)
    B = np.asarray(lat.reciprocal)
    analytic = 0.0
    for i in range(4):
        m = np.asarray(b.miller[i])[np.asarray(b.mask[i])]
        q = (np.asarray(k[i]) + m) @ B
        levels = np.sort(0.5 * np.sum(q**2, axis=1))[:4]
        analytic += float(w[i]) * float(np.dot(np.asarray(f[i]), levels))
    np.testing.assert_allclose(float(e), analytic, atol=EXACT_ATOL)
    assert set(parts) == {"Kinetic"}


# --- time reversal (decision 5: a test, not a reduction) --------------------------------


@pytest.mark.slow
def test_eigenvalues_at_k_and_minus_k_agree():
    """On the 3^3 grid most points are not time-reversal invariant; the displaced
    geometry leaves no spatial symmetry, so only time reversal pairs them."""
    _s, b, res = converged_displaced()
    k = np.asarray(b.kpoints)
    keys = {tuple(np.round(np.mod(x, 1.0), 9) % 1.0): i for i, x in enumerate(k)}
    pairs = 0
    for i, x in enumerate(k):
        j = keys[tuple(np.round(np.mod(-x, 1.0), 9) % 1.0)]
        if j != i:
            pairs += 1
        np.testing.assert_allclose(res.eigenvalues[i], res.eigenvalues[j], atol=1e-9)
    assert pairs == 26


# --- supercell equivalence (decision 4) -----------------------------------------------------


def _supercell_pair(xc):
    """Gamma-only 2x1x1 supercell against the 2x1x1 grid in the primitive cell, with the
    FFT grids passed explicitly: (2 N1, N2, N3) against (N1, N2, N3)."""
    prim = si()
    b_prim = basis_for(prim, kshape=(2, 1, 1))
    n1, n2, n3 = b_prim.fft_shape
    sc = supercell(prim, (2, 1, 1))
    b_sc = PlaneWaveBasis.build(sc.lattice, jnp.zeros((1, 3)), ECUT_TEST, fft_shape=(2 * n1, n2, n3))
    terms = si_terms(xc=xc)
    # Without XC the gap over the 2x1x1 grid is 0.33 mHa: positive, so the band filling
    # is the ground state, but below the default min_gap (decided with the project lead).
    options = {} if xc else {"min_gap": 1e-4}
    return (prim, b_prim, run(prim, b_prim, terms=terms, **options)), (sc, b_sc, run(sc, b_sc, terms=terms, **options))


def test_supercell_basis_is_the_union_of_the_primitive_bases():
    prim = si()
    b_prim = basis_for(prim, kshape=(2, 1, 1))
    sc = supercell(prim, (2, 1, 1))
    b_sc = PlaneWaveBasis.build(sc.lattice, jnp.zeros((1, 3)), ECUT_TEST, fft_shape=(2 * b_prim.fft_shape[0], *b_prim.fft_shape[1:]))
    assert int(b_sc.n_pw[0]) == int(b_prim.n_pw.sum())


@pytest.mark.slow
@pytest.mark.parametrize("xc", [False, True], ids=["without-xc", "with-xc"])
def test_supercell_at_gamma_equals_the_primitive_k_grid(xc):
    """Same discrete problem: energy per primitive cell equal at SCF accuracy. The
    supercell SCF could find a state with only the supercell period; it starts from
    the symmetric atomic density, and the converged density must be invariant under
    the primitive translation a_1 (review finding 12)."""
    (_p, b_prim, r_prim), (_s, _b, r_sc) = _supercell_pair(xc)
    np.testing.assert_allclose(float(r_sc.total_energy) / 2, float(r_prim.total_energy), atol=SCF_ENERGY_ATOL)
    rho = np.asarray(r_sc.density)
    n1 = b_prim.fft_shape[0]
    np.testing.assert_allclose(np.roll(rho, n1, axis=0), rho, atol=SCF_DENSITY_ATOL)
    np.testing.assert_allclose(rho[:n1], np.asarray(r_prim.density), atol=SCF_DENSITY_ATOL)
    for name in r_prim.energies:
        np.testing.assert_allclose(float(r_sc.energies[name]) / 2, float(r_prim.energies[name]), atol=1e-9)
