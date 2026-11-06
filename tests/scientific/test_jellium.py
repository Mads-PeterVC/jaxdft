"""Uniform density through the full density -> energy path (theory post 02a, 'Uniform
density (the jellium rung)'): E_H = 0 and E_xc = N eps_xc(r_s), exactly on any grid."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Hartree, Lattice, PlaneWaveBasis, Structure, Xc
from jaxdft.hamiltonian.energy_terms import term_potential
from jaxdft.xc.lda import lda_pw92, lda_x

A = np.diag([6.0, 7.0, 8.0])
OMEGA = 336.0


@pytest.fixture(scope="module")
def setup():
    lat = Lattice(A)
    return Structure(lat, np.zeros((0, 3)), []), PlaneWaveBasis.build(lat, np.zeros((1, 3)), 3.0)


@pytest.mark.parametrize("rs", [1.0, 2.0, 5.0])
def test_uniform_density(setup, rs):
    s, b = setup
    rho0 = 3 / (4 * np.pi * rs**3)
    n_el = rho0 * OMEGA
    rho = jnp.full(b.fft_shape, rho0)
    assert abs(float(Hartree().energy(s, b, rho))) < 1e-14
    np.testing.assert_allclose(float(Xc(lda_x).energy(s, b, rho)), -n_el * 0.4581652932831429 / rs, rtol=1e-13)
    np.testing.assert_allclose(float(Xc(lda_pw92).energy(s, b, rho)), n_el * float(lda_pw92.eps(rho0)), rtol=1e-13)
    # the potential of a uniform density is uniform: d(rho eps)/d rho
    v = np.asarray(term_potential(Xc(lda_x), s, b, rho))
    np.testing.assert_allclose(v, 4 / 3 * -0.4581652932831429 / rs, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(term_potential(Hartree(), s, b, rho)), 0.0, atol=1e-14)
