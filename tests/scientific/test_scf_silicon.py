"""Self-consistent LDA silicon (theory post 02b, 'The SCF for a bulk semiconductor',
'Invariants and checks: SCF silicon', 'Test size'; decisions 2 and 9).

Before implementation the acceptance values are invariants and analytic identities.
Cross-check numbers come from independent NumPy implementations written from the
post (the throwaway check and the review), labelled in ``support/tolerances.py``.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import (
    Anderson,
    AtomicLocal,
    AtomicNonlocal,
    Ewald,
    Hartree,
    Kinetic,
    SCFError,
    SCFOptions,
    SCFResult,
    Xc,
    assert_double_precision,
    dense_eigensolver,
    dense_matrix,
    energy,
    hamiltonian,
    lda_terms,
    lobpcg,
    scf,
    total_energy,
)
from jaxdft.hamiltonian.energy_terms import term_potential
from jaxdft.hamiltonian.ewald import ewald_energy
from jaxdft.xc.lda import lda_pw92
from support.h2 import converged_small as converged_h2_small
from support.h2 import h2_terms
from support.silicon import (
    A_SI,
    ECUT_TEST,
    HARTREE_IN_EV,
    N_TEST,
    basis_for,
    converged_test_size,
    grid_gap,
    local_only_silicon,
    run,
    si,
    si_terms,
    silicon,
    with_positions,
)
from support.tolerances import (
    CROSS_CHECK_SI_ENERGY,
    CROSS_CHECK_SI_ENERGY_ATOL,
    CROSS_CHECK_SI_EWALD,
    CROSS_CHECK_SI_EWALD_ATOL,
    CROSS_CHECK_SI_GRID_GAP_ATOL_EV,
    CROSS_CHECK_SI_GRID_GAP_EV,
    DOUBLE_COUNTING_ATOL,
    EIGEN_RESIDUAL_FRACTION,
    EXACT_ATOL,
    MONOTONE_SLACK,
    PROJECTION_AT_CONVERGENCE_MAX,
    SCF_DENSITY_ATOL,
    SCF_ENERGY_ATOL,
    SCF_TOL,
)

TERM_NAMES = {"Kinetic", "AtomicNonlocal", "AtomicLocal", "Hartree", "Xc", "Ewald"}


# --- the model ----------------------------------------------------------------------


def test_lda_terms_include_the_nonlocal_term_only_when_needed():
    names = [type(t).__name__ for t in lda_terms({"Si": silicon()})]
    assert set(names) == TERM_NAMES and len(names) == 6
    assert "AtomicNonlocal" not in [type(t).__name__ for t in h2_terms()]


def test_test_size():
    """544 plane waves at the largest k, the rule grid 24^3 (N even), 8 k-points."""
    s = si()
    b = basis_for(s)
    assert int(b.n_pw.max()) == 544
    assert b.fft_shape == (N_TEST,) * 3
    assert b.kpoints.shape == (8, 3)
    np.testing.assert_allclose(float(jnp.sum(b.kweights)), 1.0, atol=1e-15)


def test_miller_indices_stay_below_half_the_grid():
    """The k-shifted sphere reaches |G| <= Gmax + |k|: max |m_i| < N_i / 2 (theory
    post, 'The FFT grid at k != 0'; review: 6 against 12 at 12 Ha)."""
    s = si()
    b = basis_for(s)
    m = np.abs(np.asarray(b.miller)[np.asarray(b.mask)])
    assert np.all(m.max(axis=0) < np.asarray(b.fft_shape) / 2)


# --- the converged result at the test size ---------------------------------------------


def test_result_structure_and_fixed_point():
    _s, b, res = converged_test_size()
    assert isinstance(res, SCFResult)
    assert bool(res.converged) and float(res.residual) < SCF_TOL
    assert res.density.shape == b.fft_shape
    assert res.eigenvalues.shape == (8, 5)  # N_el / 2 + 1 bands; buffer bands not returned
    assert res.orbitals.shape == (8, b.n_pad, 5)
    expected = np.zeros((8, 5))
    expected[:, :4] = 2.0
    np.testing.assert_allclose(res.occupations, expected, atol=1e-15)
    assert_double_precision(res)


def test_density_integrates_to_eight_electrons():
    s, _b, res = converged_test_size()
    omega = float(s.lattice.volume)
    np.testing.assert_allclose(omega / res.density.size * float(res.density.sum()), 8.0, rtol=1e-12)
    assert float(res.density.min()) > 0.0  # no vacuum


def test_clip_and_renormalise_does_nothing_at_convergence():
    _s, _b, res = converged_test_size()
    assert float(res.history["projection"][-1]) < PROJECTION_AT_CONVERGENCE_MAX


def test_energy_breakdown():
    s, _b, res = converged_test_size()
    assert set(res.energies) == TERM_NAMES
    np.testing.assert_allclose(sum(float(v) for v in res.energies.values()), float(res.total_energy), atol=1e-12)
    q = s.weights * silicon().ionic_charge
    np.testing.assert_allclose(float(res.energies["Ewald"]), float(ewald_energy(s, q)), atol=EXACT_ATOL)
    e = {k: float(v) for k, v in res.energies.items()}
    assert e["Kinetic"] > 0 and e["Hartree"] > 0 and e["Xc"] < 0
    assert e["AtomicNonlocal"] > 0  # repulsive projectors (theory post)


def test_eigenpair_residuals_tied_to_scf_tolerance():
    _, _, res = converged_test_size()
    assert np.all(np.asarray(res.eigen_residuals) < EIGEN_RESIDUAL_FRACTION * SCF_TOL)


def test_cross_checks_with_independent_numpy_implementations():
    """Ewald (post and review), total energy (review), gap over the grid (both)."""
    _s, _b, res = converged_test_size()
    np.testing.assert_allclose(float(res.energies["Ewald"]), CROSS_CHECK_SI_EWALD, atol=CROSS_CHECK_SI_EWALD_ATOL)
    np.testing.assert_allclose(float(res.total_energy), CROSS_CHECK_SI_ENERGY, atol=CROSS_CHECK_SI_ENERGY_ATOL)
    gap_ev = grid_gap(res.eigenvalues) * HARTREE_IN_EV
    np.testing.assert_allclose(gap_ev, CROSS_CHECK_SI_GRID_GAP_EV, atol=CROSS_CHECK_SI_GRID_GAP_ATOL_EV)


def test_history_records_the_gap_over_the_grid():
    """Decision 9: the gap over all k-points is logged at every iteration; at
    convergence it is the gap the guard tests."""
    _s, _b, res = converged_test_size()
    gap = np.asarray(res.history["gap"])
    assert gap.shape == (int(res.n_iter) + 1,)
    np.testing.assert_allclose(gap[-1], grid_gap(res.eigenvalues), atol=1e-8)


def test_energy_function_equals_scf_total():
    s, b, res = converged_test_size()
    e = energy(s, b, si_terms(), SCFOptions(tol=SCF_TOL))
    np.testing.assert_allclose(float(e), float(res.total_energy), atol=SCF_ENERGY_ATOL)


# --- self-consistency and variationality --------------------------------------------------


def test_direct_and_double_counting_energies_agree():
    """E = sum_k w_k sum_n f eps - E_H + E_xc - int v_xc rho + E_Ewald; the nonlocal
    energy is inside the band sum (theory post, 'Energy and derivatives')."""
    s = si()
    b = basis_for(s)
    res = run(s, b, tol=1e-10)
    rho = res.density
    omega = float(s.lattice.volume)
    vxc = term_potential(Xc(lda_pw92), s, b, rho)
    int_vxc_rho = omega / rho.size * float(jnp.sum(vxc * rho))
    band = float(jnp.sum(b.kweights[:, None] * res.occupations * res.eigenvalues))
    e = {k: float(v) for k, v in res.energies.items()}
    e_dc = band - e["Hartree"] + e["Xc"] - int_vxc_rho + e["Ewald"]
    np.testing.assert_allclose(e_dc, float(res.total_energy), atol=DOUBLE_COUNTING_ATOL)


def test_total_energy_of_the_converged_orbitals_is_the_scf_energy():
    s, b, res = converged_test_size()
    e, parts = total_energy(s, b, si_terms(), res.orbitals, res.occupations)
    np.testing.assert_allclose(float(e), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    assert set(parts) == TERM_NAMES


def test_nonlocal_energy_is_the_expectation_value_of_the_dense_operator():
    """E_nl = sum_k w_k sum_n f_nk <psi|V_nl|psi> with V_nl from the dense matrix."""
    s, b, res = converged_test_size()
    term = AtomicNonlocal({"Si": silicon()})
    e = 0.0
    for k in range(b.kpoints.shape[0]):
        V = np.asarray(dense_matrix(hamiltonian(s, b[k], (term,))))
        n = int(b[k].mask.sum())
        C = np.asarray(res.orbitals[k])[:n]
        expect = np.real(np.einsum("in,ij,jn->n", C.conj(), V[:n, :n], C))
        e += float(b.kweights[k]) * float(np.dot(np.asarray(res.occupations[k]), expect))
    np.testing.assert_allclose(float(res.energies["AtomicNonlocal"]), e, atol=1e-10)


def test_variational_in_the_orbitals():
    """Perturbing an occupied orbital (band 0 at k = 0, orthogonalised to the occupied
    bands at that k) raises E quadratically."""
    s, b, res = converged_test_size()
    mask = np.asarray(b.mask[0])
    rng = np.random.default_rng(0)
    d = np.where(mask, rng.normal(size=b.n_pad) + 1j * rng.normal(size=b.n_pad), 0)
    C = np.asarray(res.orbitals[0, :, :4])
    d = d - C @ (C.conj().T @ d)
    d /= np.linalg.norm(d)
    c0 = C[:, 0]
    e0 = float(res.total_energy)

    def e_at(eps):
        c = res.orbitals.at[0, :, 0].set(jnp.asarray((c0 + eps * d) / np.sqrt(1 + eps**2)))
        return float(total_energy(s, b, si_terms(), c, res.occupations)[0])

    d1, d2 = e_at(1e-3) - e0, e_at(2e-3) - e0
    assert d1 > 0 and d2 > 0
    np.testing.assert_allclose(d2 / d1, 4.0, rtol=0.05)


@pytest.mark.slow
def test_monotone_decrease_with_cutoff_at_fixed_grid():
    """Variational at a fixed FFT grid and k-grid. The default grid changes with the
    cutoff, and that part is not variational (review finding 9), so it is held fixed."""
    s = si()
    shape = basis_for(s, ECUT_TEST).fft_shape
    energies = [float(run(s, basis_for(s, ec, fft_shape=shape)).total_energy) for ec in (6.0, 8.0, 10.0, 12.0)]
    assert np.all(np.diff(energies) < MONOTONE_SLACK), energies


# --- independence of numerical choices ------------------------------------------------


def test_independent_of_initial_density():
    s, b, res = converged_test_size()
    uniform = jnp.full(b.fft_shape, 8.0 / float(s.lattice.volume))
    other = scf(s, b, si_terms(), SCFOptions(tol=SCF_TOL), density0=uniform)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


@pytest.mark.slow
def test_independent_of_mixing_parameters():
    s, b, res = converged_test_size()
    other = run(s, b, mixer=Anderson(depth=3, beta=0.4), max_iter=300)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)


@pytest.mark.slow
@pytest.mark.parametrize("buffer", [0, 3])
def test_independent_of_buffer_bands(buffer):
    """Buffer bands change the cost, never the result (decision 2)."""
    s, b, res = converged_test_size()
    other = run(s, b, buffer_bands=buffer)
    assert other.eigenvalues.shape == res.eigenvalues.shape
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.eigenvalues, res.eigenvalues, atol=1e-9)


@pytest.mark.slow
def test_dense_and_lobpcg_give_the_same_scf():
    s, b, res = converged_test_size()
    dense = run(s, b, eigensolver=dense_eigensolver)
    np.testing.assert_allclose(float(dense.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(dense.eigenvalues, res.eigenvalues, atol=1e-9)


def test_scf_options_defaults():
    o = SCFOptions()
    assert o.buffer_bands == 3  # set in Milestone 2c from the V100 measurement
    assert o.degeneracy_tol == 1e-6


def test_buffer_bands_are_not_returned_for_h2():
    s, b, res = converged_h2_small()
    other = scf(s, b, h2_terms(), SCFOptions(tol=SCF_TOL, buffer_bands=3))
    assert other.eigenvalues.shape == res.eigenvalues.shape == (1, 2)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)


# --- exact symmetries of the discrete problem ---------------------------------------------


@pytest.mark.slow
def test_k_grid_in_minus_half_to_half_is_the_same_problem():
    """DFTK lists the grid in [-1/2, 1/2): the basis at k + G0 is the basis at k
    relabelled (theory post, 'Fractional coordinates in [0, 1)')."""
    from jaxdft import PlaneWaveBasis

    s, b, res = converged_test_size()
    k = np.asarray(b.kpoints)
    shifted = np.where(k >= 0.5, k - 1.0, k)
    other_basis = PlaneWaveBasis.build(s.lattice, jnp.asarray(shifted), ECUT_TEST, kweights=b.kweights)
    other = run(s, other_basis)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.eigenvalues, res.eigenvalues, atol=1e-9)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


def test_exchanging_the_atoms_changes_nothing():
    s, b, res = converged_test_size()
    other = run(with_positions(s, np.asarray(s.positions)[::-1]), b)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, res.density, atol=SCF_DENSITY_ATOL)


@pytest.mark.slow
def test_translation_by_a_grid_vector_is_exact():
    s, b, res = converged_test_size()
    steps = np.array([1, -2, 3])
    other = run(with_positions(s, np.asarray(s.positions) + steps / N_TEST), b)
    np.testing.assert_allclose(float(other.total_energy), float(res.total_energy), atol=SCF_ENERGY_ATOL)
    np.testing.assert_allclose(other.density, np.roll(np.asarray(res.density), steps, axis=(0, 1, 2)), atol=SCF_DENSITY_ATOL)


# --- the gap guard over all k-points --------------------------------------------------------


def test_gap_guard_rejects_a_negative_gap_over_the_grid():
    """Positive control (theory post; review: gap over the grid -0.063 Ha while every
    per-k gap is positive): local-only silicon on the 2^3 grid at 12 Ha. Its band
    filling is not the aufbau filling of the crystal. With fixed occupations this
    semimetal does not converge either (a band crossing at the Fermi level), so the
    test asks only that the guard rejects it for the gap, not that the SCF converged
    first (decided 2026-11-11 with the project lead; the post said 'converged')."""
    s = si()
    b = basis_for(s)
    terms = (Kinetic(), AtomicLocal({"Si": local_only_silicon()}), Hartree(), Xc(lda_pw92), Ewald({"Si": local_only_silicon()}))
    with pytest.raises(SCFError) as err:
        scf(s, b, terms, SCFOptions(tol=SCF_TOL))
    assert "gap over all k-points" in str(err.value)
    assert float(err.value.history["gap"][-1]) < -0.05
    assert np.isnan(float(energy(s, b, terms, SCFOptions(tol=SCF_TOL))))


@pytest.mark.parametrize(
    "options",
    [SCFOptions(tol=SCF_TOL, max_iter=2), SCFOptions(tol=SCF_TOL, min_gap=1.0), SCFOptions(tol=SCF_TOL, eigensolver=partial(lobpcg, max_iter=1))],
    ids=["scf-unconverged", "gap-guard", "eigensolver-unconverged"],
)
def test_failures_raise_in_scf_and_give_nan_in_energy(options):
    s, b, _ = converged_test_size()
    with pytest.raises(SCFError):
        scf(s, b, si_terms(), options)
    assert np.isnan(float(energy(s, b, si_terms(), options)))
    g = jax.grad(lambda st: energy(st, b, si_terms(), options))(s)
    assert np.all(np.isnan(np.asarray(g.positions)))


def test_lattice_constant_helper_is_the_experimental_value():
    from jaxdft import units

    np.testing.assert_allclose(units.convert(A_SI, "Bohr", "Angstrom"), 5.431, atol=1e-3)
