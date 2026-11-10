"""The FFT local pseudopotential term (theory post 02a, 'The local term moves to real
space', 'Nyquist planes', 'Electrostatics and the G = 0 terms')."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import (
    AtomicLocal,
    Kinetic,
    LocalPotential,
    PlaneWaveBasis,
    band_structure,
    dense_matrix,
    diamond,
    hamiltonian,
)
from jaxdft.hamiltonian.energy_terms import term_potential
from jaxdft.pseudo.cohen_bergstresser import silicon
from support.h2 import ECUT_TEST, L_TEST, basis_for, displaced, h2, hydrogen
from support.tolerances import EXACT_ATOL



@pytest.fixture(scope="module")
def displaced_h2():
    """Not inversion-symmetric about a grid point: exercises the Nyquist planes
    (review 2.7 measured |Im V| = 3.9e-4 Ha here if they are kept)."""
    return displaced(h2(), 0, [1e-3, 0.0, 0.0])


def test_dense_fft_operator_equals_v_of_g_minus_g_prime(displaced_h2):
    """Exactness claim 2: the dense matrix of the FFT operator is V~(G - G')."""
    s = displaced_h2
    b = basis_for(s)[0]
    term = AtomicLocal({"H": hydrogen()})
    H = np.asarray(dense_matrix(hamiltonian(s, b, (term,), density=None)))
    mask = np.asarray(b.mask)
    m = np.asarray(b.miller)[mask]
    sub = H[np.ix_(mask, mask)]
    # check a random sample of rows against the analytic coefficients
    rows = np.random.default_rng(0).choice(len(m), 40, replace=False)
    d = m[rows][:, None, :] - m[None, :, :]
    ref = np.asarray(term.coefficients(s, jnp.asarray(d.reshape(-1, 3)))).reshape(d.shape[:2])
    np.testing.assert_allclose(sub[rows], ref, atol=1e-13)


def test_hermitian_on_a_non_symmetric_geometry(displaced_h2):
    s = displaced_h2
    b = basis_for(s)[0]
    H = np.asarray(dense_matrix(hamiltonian(s, b, (AtomicLocal({"H": hydrogen()}),), density=None)))
    assert np.max(np.abs(H - H.conj().T)) < 1e-14 * np.max(np.abs(H))


def test_nyquist_planes_are_zeroed(displaced_h2):
    s = displaced_h2
    b = basis_for(s)
    v = np.asarray(AtomicLocal({"H": hydrogen()}).grid_coefficients(s, b))
    N = b.fft_shape
    assert all(n % 2 == 0 for n in N)
    assert np.all(v[N[0] // 2, :, :] == 0) and np.all(v[:, N[1] // 2, :] == 0) and np.all(v[:, :, N[2] // 2] == 0)
    pot = term_potential(AtomicLocal({"H": hydrogen()}), s, b, jnp.zeros(N))
    assert pot.dtype == jnp.float64


def test_alpha_convention_values_at_the_test_size():
    """G = 0: the eigenvalue shift sum_j alpha_j / Omega = -2.60e-6 Ha is the average of
    V_loc, and a uniform density gives E_loc = N sum_j alpha_j / Omega = -5.19e-6 Ha
    (decision 3; DFTK's PspCorrection must later equal the second)."""
    s = h2()
    b = basis_for(s)
    term = AtomicLocal({"H": hydrogen()})
    omega = L_TEST**3
    alpha = hydrogen().alpha
    rho = jnp.full(b.fft_shape, 2.0 / omega)
    np.testing.assert_allclose(float(term.energy(s, b, rho)), 2 * 2 * alpha / omega, atol=1e-15)
    np.testing.assert_allclose(float(term.energy(s, b, rho)), -5.19e-6, atol=0.005e-6)
    v = np.asarray(term_potential(term, s, b, rho))
    np.testing.assert_allclose(v.mean(), 2 * alpha / omega, atol=1e-15)
    np.testing.assert_allclose(v.mean(), -2.60e-6, atol=0.005e-6)


def test_energy_is_linear_and_its_potential_density_independent():
    s = h2()
    b = basis_for(s)
    term = AtomicLocal({"H": hydrogen()})
    rng = np.random.default_rng(1)
    r1, r2 = (jnp.asarray(rng.uniform(0, 0.1, b.fft_shape)) for _ in range(2))
    np.testing.assert_allclose(
        float(term.energy(s, b, r1 + 2 * r2)), float(term.energy(s, b, r1)) + 2 * float(term.energy(s, b, r2)), rtol=1e-12
    )
    np.testing.assert_allclose(term_potential(term, s, b, r1), term_potential(term, s, b, r2), atol=1e-12)


# --- Cohen-Bergstresser: gather (1b) vs FFT (2a) ---------------------------------------


@pytest.fixture(scope="module")
def si():
    return diamond(5.43, "Si", unit="Angstrom")


def test_coefficients_agree_with_gather_term(si):
    m = jnp.asarray(np.array(list(np.ndindex(7, 7, 7))) - 3)
    np.testing.assert_allclose(
        AtomicLocal({"Si": silicon()}).coefficients(si, m), LocalPotential({"Si": silicon()}).coefficients(si, m), atol=1e-15
    )


def test_fft_term_reproduces_cohen_bergstresser_bands(si):
    """Promised in the 1b post: gather and FFT local terms agree to 1e-10 Ha."""
    k = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5], [0.1, 0.2, 0.33]])
    b = PlaneWaveBasis.build(si.lattice, k, 6.0)
    gather = band_structure(si, b, 8, terms=(Kinetic(), LocalPotential({"Si": silicon()}))).eigenvalues
    fft = band_structure(si, b, 8, terms=(Kinetic(), AtomicLocal({"Si": silicon()}))).eigenvalues
    np.testing.assert_allclose(fft, gather, atol=EXACT_ATOL)


def test_unknown_species_raises():
    with pytest.raises(ValueError):
        AtomicLocal({"He": hydrogen()}).energy(h2(), basis_for(h2()), jnp.zeros((30, 30, 30)))


def test_test_size_basis():
    """1503 plane waves on a 30^3 grid (review)."""
    b = basis_for(h2(), ECUT_TEST)
    assert int(b.n_pw[0]) == 1503 and b.fft_shape == (30, 30, 30)
