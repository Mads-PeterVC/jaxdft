"""Self-consistent LDA for H2 in a box (theory post 02a, 'Invariants and checks',
'SCF and energy'; decisions 7 and 11).

Before implementation the acceptance values are analytic checks and invariants. Two
cross-check numbers come from two independent NumPy implementations written from the
post (the throwaway check while preparing it and the review); they are labelled as
such and are not reference-code values.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import (
    AtomicLocal,
    Anderson,
    Hartree,
    Kinetic,
    SCFError,
    SCFOptions,
    SCFResult,
    Xc,
    assert_double_precision,
    dense_eigensolver,
    energy,
    lobpcg,
    scf,
    total_energy,
)
from jaxdft.hamiltonian.energy_terms import term_potential
from jaxdft.hamiltonian.ewald import ewald_energy
from jaxdft.xc.lda import lda_pw92
from support.h2 import (
    ECUT_SMALL,
    L_SMALL,
    basis_for,
    converged_small,
    converged_test_size,
    h2,
    h2_terms,
    hydrogen,
    run,
    translated,
)
from support.mixers import AdaptiveLinear
from support.tolerances import (
    DOUBLE_COUNTING_ATOL,
    EIGEN_RESIDUAL_FRACTION,
    EXACT_ATOL,
    SCF_DENSITY_ATOL,
    SCF_ENERGY_ATOL,
    SCF_TOL,
)

# Cross-checks (not references): E and gap at L = 10, Ecut = 10, d = 1.45 from the
# post's NumPy check (N = 29: -1.104880727437) and the review (N = 30: -1.104880728,
# gap 0.347192). They agree to ~1e-9; the margin covers grid placement and SCF noise.
CROSS_CHECK_E = -1.1048807274
CROSS_CHECK_GAP = 0.347192
CROSS_CHECK_E_ATOL = 1e-7
CROSS_CHECK_GAP_ATOL = 1e-6


# --- the converged result at the test size ------------------------------------------


def test_result_structure_and_fixed_point():
    _s, b, res = converged_test_size()
    assert isinstance(res, SCFResult)
    assert bool(res.converged)
    assert float(res.residual) < SCF_TOL
    assert res.density.shape == b.fft_shape
    assert res.eigenvalues.shape == (1, 2)  # occupied + 1 by default
    assert res.orbitals.shape == (1, b.n_pad, 2)
    np.testing.assert_allclose(res.occupations, [[2.0, 0.0]], atol=1e-15)
    assert_double_precision(res)


def test_density_integrates_to_n_electrons():
    s, _b, res = converged_test_size()
    omega = float(s.lattice.volume)
    np.testing.assert_allclose(omega / res.density.size * float(res.density.sum()), 2.0, rtol=1e-12)
    assert float(res.density.min()) >= 0.0


def test_energy_breakdown():
    s, _b, res = converged_test_size()
    assert set(res.energies) == {"Kinetic", "AtomicLocal", "Hartree", "Xc", "Ewald"}
    np.testing.assert_allclose(sum(float(v) for v in res.energies.values()), float(res.total_energy), atol=1e-12)
    q = s.weights * hydrogen().ionic_charge
    np.testing.assert_allclose(float(res.energies["Ewald"]), float(ewald_energy(s, q)), atol=EXACT_ATOL)
    np.testing.assert_allclose(float(res.energies["Ewald"]), 0.126737138563, atol=EXACT_ATOL)
    assert float(res.energies["Kinetic"]) > 0 and float(res.energies["Hartree"]) > 0
    assert float(res.energies["Xc"]) < 0


def test_eigenpair_residuals_tied_to_scf_tolerance():
    _, _, res = converged_test_size()
    assert np.all(np.asarray(res.eigen_residuals) < EIGEN_RESIDUAL_FRACTION * SCF_TOL)


def test_cross_check_with_independent_numpy_implementations():
    _, _, res = converged_test_size()
    np.testing.assert_allclose(float(res.total_energy), CROSS_CHECK_E, atol=CROSS_CHECK_E_ATOL)
    eps = np.asarray(res.eigenvalues[0])
    np.testing.assert_allclose(eps[1] - eps[0], CROSS_CHECK_GAP, atol=CROSS_CHECK_GAP_ATOL)


def test_energy_function_equals_scf_total():
    s, b, res = converged_test_size()
    e = energy(s, b, h2_terms(), SCFOptions(tol=SCF_TOL))
    np.testing.assert_allclose(float(e), float(res.total_energy), atol=SCF_ENERGY_ATOL)


# --- self-consistency and variationality (small size) ----------------------------------


def test_direct_and_double_counting_energies_agree():
    s = h2(L_SMALL)
    b = basis_for(s, ECUT_SMALL)
    res = run(s, b, tol=1e-10)
    omega = float(s.lattice.volume)
    rho = res.density
    vxc = term_potential(Xc(lda_pw92), s, b, rho)
    int_vxc_rho = omega / rho.size * float(jnp.sum(vxc * rho))
    band = float(jnp.sum(res.occupations * res.eigenvalues))
    e_dc = band - float(res.energies["Hartree"]) + float(res.energies["Xc"]) - int_vxc_rho + float(res.energies["Ewald"])
    np.testing.assert_allclose(e_dc, float(res.total_energy), atol=DOUBLE_COUNTING_ATOL)


def test_total_energy_of_the_converged_orbitals_is_the_scf_energy():
    s, b, res = converged_small()
    e, parts = total_energy(s, b, h2_terms(), res.orbitals, res.occupations)
    np.testing.assert_allclose(float(e), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    assert set(parts) == set(res.energies)


def test_variational_in_the_orbitals():
    """Perturbing the occupied orbital raises E quadratically: the gapped aufbau solution
    is a minimum on the orthonormality manifold, not merely a stationary point."""
    s, b, res = converged_small()
    mask = np.asarray(b.mask[0])
    rng = np.random.default_rng(0)
    d = rng.normal(size=b.n_pad) + 1j * rng.normal(size=b.n_pad)
    d = np.where(mask, d, 0)
    c0 = np.asarray(res.orbitals[0, :, 0])
    d = d - c0 * np.vdot(c0, d)
    d /= np.linalg.norm(d)
    e0 = float(res.total_energy)

    def e_at(eps):
        c = res.orbitals.at[0, :, 0].set(jnp.asarray((c0 + eps * d) / np.sqrt(1 + eps**2)))
        return float(total_energy(s, b, h2_terms(), c, res.occupations)[0])

    d1, d2 = e_at(1e-3) - e0, e_at(2e-3) - e0
    assert d1 > 0 and d2 > 0
    np.testing.assert_allclose(d2 / d1, 4.0, rtol=0.05)


def test_monotone_decrease_with_cutoff_at_fixed_grid():
    s = h2(L_SMALL)
    shape = basis_for(s, 7.0).fft_shape
    energies = [float(run(s, basis_for(s, ec, fft_shape=shape)).total_energy) for ec in (3.0, 4.0, 5.0, 6.0, 7.0)]
    assert np.all(np.diff(energies) < 1e-10), energies


# --- independence of numerical choices ------------------------------------------------


def test_independent_of_initial_density():
    s, b, res = converged_small()
    uniform = jnp.full(b.fft_shape, 2.0 / float(s.lattice.volume))
    other = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL), density0=uniform)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


def test_independent_of_mixing_parameters():
    s, b, res = converged_small()
    for mixer in (Anderson(depth=3, beta=0.4), Anderson(depth=0, beta=0.5)):
        other = run(s, b, mixer=mixer, max_iter=300)
        np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)


def test_independent_of_the_mixing_scheme():
    """The SCF accepts any Mixer, not only Anderson, and the ground state does not
    depend on which one (engineering log, 'Mixing behind a protocol')."""
    s, b, res = converged_small()
    other = run(s, b, mixer=AdaptiveLinear(), max_iter=300)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


def test_dense_and_lobpcg_give_the_same_scf():
    s, b, res = converged_small()
    dense = run(s, b, eigensolver=dense_eigensolver)
    np.testing.assert_allclose(float(dense.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(dense.eigenvalues, res.eigenvalues, atol=1e-9)


# --- exact symmetries of the discrete problem ---------------------------------------------


def test_grid_vector_translation_is_exact():
    s, b, res = converged_small()
    N = b.fft_shape
    for steps in ((1, 0, 0), (0, 3, -2)):
        shift = np.asarray(steps) * L_SMALL / np.asarray(N)
        other = run(translated(s, shift), b)
        np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
        np.testing.assert_allclose(other.density, np.roll(np.asarray(res.density), steps, axis=(0, 1, 2)), atol=SCF_DENSITY_ATOL)


def test_axis_permutation_is_exact():
    _s, b, res = converged_small()
    for axis, perm in ((0, (2, 1, 0)), (1, (0, 2, 1))):
        other = run(h2(L_SMALL, axis=axis), b)
        np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
        np.testing.assert_allclose(other.density, np.transpose(np.asarray(res.density), perm), atol=SCF_DENSITY_ATOL)


def test_density_has_d4h_symmetry():
    """H2 along z, centre on the grid point N/2: the operations of the molecule that map
    the cubic grid to itself (D4h) are exact symmetries of the solution."""
    _, b, res = converged_small()
    rho = np.asarray(res.density)
    assert all(n % 2 == 0 for n in b.fft_shape)

    def reflect(x, axis):
        return np.roll(np.flip(x, axis), 1, axis)

    for op in (lambda x: np.swapaxes(x, 0, 1), lambda x: reflect(x, 0), lambda x: reflect(x, 2)):
        np.testing.assert_allclose(op(rho), rho, atol=SCF_DENSITY_ATOL)


def test_exchanging_the_atoms_changes_nothing():
    s, b, res = converged_small()
    from jaxdft import Structure

    swapped = Structure(s.lattice, s.positions[::-1], s.species, s.weights)
    other = run(swapped, b)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


# --- per-atom weights ------------------------------------------------------------------


def test_explicit_unit_weights_equal_default():
    _s, b, res = converged_small()
    other = run(h2(L_SMALL, weights=[1.0, 1.0]), b)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)


def test_fractional_weight_keeps_the_cell_neutral():
    """lambda = (0.7, 1): N_el = 1.7 electrons, ionic charges 0.7 and 1 in Ewald."""
    s = h2(L_SMALL, weights=[0.7, 1.0])
    b = basis_for(s, ECUT_SMALL)
    res = run(s, b)
    assert bool(res.converged)
    omega = float(s.lattice.volume)
    np.testing.assert_allclose(omega / res.density.size * float(res.density.sum()), 1.7, rtol=1e-12)
    np.testing.assert_allclose(res.occupations[0], [1.7, 0.0], atol=1e-15)
    np.testing.assert_allclose(float(res.energies["Ewald"]), float(ewald_energy(s, jnp.asarray([0.7, 1.0]))), atol=EXACT_ATOL)
    full = converged_small()[2]
    assert float(res.total_energy) > float(full.total_energy)


# --- configuration errors and failure reporting ---------------------------------------


def test_ionic_terms_without_ewald_raise():
    s, b, _ = converged_small()
    terms = (Kinetic(), AtomicLocal({"H": hydrogen()}), Hartree(), Xc(lda_pw92))
    with pytest.raises(ValueError, match="Ewald"):
        scf(s, b, terms, SCFOptions())


@pytest.mark.parametrize(
    "options",
    [
        SCFOptions(tol=SCF_TOL, max_iter=2),
        SCFOptions(tol=SCF_TOL, min_gap=1.0),
        SCFOptions(tol=SCF_TOL, eigensolver=partial(lobpcg, max_iter=1)),
    ],
    ids=["scf-unconverged", "gap-guard", "eigensolver-unconverged"],
)
def test_failures_raise_in_scf_and_give_nan_in_energy(options):
    """scf() raises a descriptive SCFError; energy() and its gradient are NaN
    (engineering log, 2a API: failure under tracing is NaN)."""
    s, b, _ = converged_small()
    with pytest.raises(SCFError):
        scf(s, b, h2_terms(), options)
    assert np.isnan(float(energy(s, b, h2_terms(), options)))
    g = jax.grad(lambda st: energy(st, b, h2_terms(), options))(s)
    assert np.all(np.isnan(np.asarray(g.positions)))
