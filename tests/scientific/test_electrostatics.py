"""Analytic electrostatics (theory post 02a, 'Analytic electrostatic checks', checked
while preparing the post and by the review): Madelung constants, Ewald
eta-independence, the Gaussian Hartree identity, Hartree of a cosine density."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Hartree, Lattice, PlaneWaveBasis, Structure
from jaxdft.hamiltonian.energy_terms import term_potential
from jaxdft.hamiltonian.ewald import ewald_energy
from support.tolerances import EXACT_ATOL

GAMMA = np.zeros((1, 3))
SC = np.eye(3)
BCC = 0.5 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=float)
FCC = 0.5 * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


def point_charges(A, frac, charges):
    return Structure(Lattice(A), np.asarray(frac, dtype=float), ["X"] * len(charges)), jnp.asarray(charges, float)


@pytest.mark.parametrize(
    ("A", "madelung"),
    [(SC, -0.8800594421), (BCC, -0.8959292557), (FCC, -0.8958736152)],
    ids=["sc", "bcc", "fcc"],
)
@pytest.mark.parametrize("eta_scale", [3.0, 5.0])
def test_madelung_one_charge_in_background(A, madelung, eta_scale):
    """Energy per ion in units of 1/r_ws, r_ws = (3 Omega / 4 pi)^(1/3). Tests the
    background term, the self term and the signs."""
    a = 7.3
    s, q = point_charges(a * A, [[0, 0, 0]], [1.0])
    r_ws = (3 * abs(np.linalg.det(a * A)) / (4 * np.pi)) ** (1 / 3)
    e = float(ewald_energy(s, q, eta=eta_scale / a))
    np.testing.assert_allclose(e * r_ws, madelung, atol=1e-9)


@pytest.mark.parametrize("eta_scale", [3.0, 5.0])
def test_madelung_nacl(eta_scale):
    """Rocksalt, charges +-1: -1.7475645946 per ion pair in units of 1/r0, r0 = a/2."""
    a = 10.6
    s, q = point_charges(a * FCC, [[0, 0, 0], [0.5, 0.5, 0.5]], [1.0, -1.0])
    e = float(ewald_energy(s, q, eta=eta_scale / a))
    np.testing.assert_allclose(e * a / 2, -1.7475645946, atol=1e-9)


def test_default_eta_and_h2_cell_value():
    """H2 at L = 10, d = 1.45: 0.126737138563 Ha, from two independent NumPy
    implementations (the post's check and the review's), eta-independent to 1e-14."""
    s, q = point_charges(10.0 * np.eye(3), [[0.5, 0.5, 0.5 - 0.0725], [0.5, 0.5, 0.5 + 0.0725]], [1.0, 1.0])
    for eta in (None, 0.3, 0.6, 1.2):
        np.testing.assert_allclose(float(ewald_energy(s, q, eta=eta)), 0.126737138563, atol=EXACT_ATOL)


def test_gaussian_hartree_identity():
    """E_H[rho_sigma] = E_Ewald(Z) + Z^2/(2 sqrt(pi) sigma) + 2 pi Z^2 sigma^2 / Omega for
    rho~(G) = (Z/Omega) exp(-G^2 sigma^2 / 2), up to erfc(L / 2 sigma) ~ 1e-21."""
    L, sigma, Z = 8.0, 0.6, 1.0
    lat = Lattice(L * np.eye(3))
    basis = PlaneWaveBasis.build(lat, GAMMA, 2.0, fft_shape=(64, 64, 64))
    N = 64
    g1 = np.fft.fftfreq(N, 1 / N) * 2 * np.pi / L
    G2 = g1[:, None, None] ** 2 + g1[None, :, None] ** 2 + g1[None, None, :] ** 2
    rho_g = Z / L**3 * np.exp(-G2 * sigma**2 / 2)  # exp(-G^2 sigma^2 / 2) < 1e-60 at the grid edge
    rho = np.real(np.fft.ifftn(rho_g)) * N**3
    empty = Structure(lat, np.zeros((0, 3)), [])
    e_h = float(Hartree().energy(empty, basis, jnp.asarray(rho)))
    one, q = point_charges(L * np.eye(3), [[0.2, 0.3, 0.4]], [Z])
    expected = float(ewald_energy(one, q)) + Z**2 / (2 * np.sqrt(np.pi) * sigma) + 2 * np.pi * Z**2 * sigma**2 / L**3
    np.testing.assert_allclose(e_h, expected, atol=EXACT_ATOL)


def test_hartree_of_a_cosine_density():
    """rho = rho0 + a cos(G.r): V_H = (4 pi a / G^2) cos(G.r), E_H = pi Omega a^2 / G^2;
    the uniform part contributes nothing (G = 0 dropped)."""
    A = np.diag([6.0, 7.0, 8.0])
    lat = Lattice(A)
    basis = PlaneWaveBasis.build(lat, GAMMA, 3.0)
    N = basis.fft_shape
    j = np.stack(np.meshgrid(*[np.arange(n) / n for n in N], indexing="ij"), -1)
    m = np.array([1, 0, 2])
    G = m @ np.asarray(lat.reciprocal)
    cos = np.cos(2 * np.pi * j @ m)
    a, rho0 = 0.03, 0.05
    rho = jnp.asarray(rho0 + a * cos)
    empty = Structure(lat, np.zeros((0, 3)), [])
    omega = float(np.linalg.det(A))
    np.testing.assert_allclose(float(Hartree().energy(empty, basis, rho)), np.pi * omega * a**2 / (G @ G), rtol=1e-13)
    v = np.asarray(term_potential(Hartree(), empty, basis, rho))
    assert v.dtype == np.float64 and v.shape == tuple(N)
    np.testing.assert_allclose(v, 4 * np.pi * a / (G @ G) * cos, atol=1e-13)
