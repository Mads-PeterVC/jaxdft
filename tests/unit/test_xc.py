"""LDA exchange-correlation: Slater exchange + PW92 correlation, unpolarised
(theory post 02a, 'Exchange and correlation'; decision 11 defers libxc values)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft.xc.lda import LDA, PW92_UNPOLARISED, lda_pw92, lda_x

RS = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0])


def rho_of_rs(rs):
    return 3.0 / (4.0 * np.pi * np.asarray(rs) ** 3)


def pw92_eps_c_reference(rs):
    """PW92 eq. 10 with the unpolarised Table I parameters, written independently."""
    A, a1, b1, b2, b3, b4 = 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294
    rs = np.asarray(rs, dtype=float)
    den = 2 * A * (b1 * rs**0.5 + b2 * rs + b3 * rs**1.5 + b4 * rs**2)
    return -2 * A * (1 + a1 * rs) * np.log1p(1 / den)


def test_pw92_parameters_are_the_published_ones():
    """lda_c_pw, not lda_c_pw_mod (theory post: the two differ by 2e-7 Ha on H2)."""
    assert PW92_UNPOLARISED == {
        "A": 0.031091,
        "alpha1": 0.21370,
        "beta1": 7.5957,
        "beta2": 3.5876,
        "beta3": 1.6382,
        "beta4": 0.49294,
    }


def test_slater_exchange_per_electron():
    """eps_x = -(3/4)(3/2pi)^(2/3) / r_s = -0.4581652932831429 / r_s."""
    np.testing.assert_allclose(lda_x.eps(rho_of_rs(RS)), -0.4581652932831429 / RS, rtol=1e-14)


def test_exchange_potential_is_four_thirds_eps():
    rho = rho_of_rs(RS)
    e, v = lda_x(jnp.asarray(rho))
    np.testing.assert_allclose(e, rho * lda_x.eps(rho), rtol=1e-14)
    np.testing.assert_allclose(v, 4 / 3 * lda_x.eps(rho), rtol=1e-14)


def test_pw92_correlation_matches_independent_formula():
    rho = rho_of_rs(RS)
    eps_c = np.asarray(lda_pw92.eps(rho)) - np.asarray(lda_x.eps(rho))
    np.testing.assert_allclose(eps_c, pw92_eps_c_reference(RS), rtol=1e-13)


def test_pw92_correlation_is_negative_and_increasing_in_rs():
    rs = np.linspace(0.05, 100, 2000)
    rho = rho_of_rs(rs)
    eps_c = np.asarray(lda_pw92.eps(rho)) - np.asarray(lda_x.eps(rho))
    assert np.all(eps_c < 0)
    assert np.all(np.diff(eps_c) > 0)


@pytest.mark.parametrize("functional", [lda_x, lda_pw92], ids=["x", "pw92"])
def test_potential_is_derivative_of_energy_density(functional):
    rho = rho_of_rs(RS)
    _, v = functional(jnp.asarray(rho))
    h = 1e-6 * rho
    ep, _ = functional(jnp.asarray(rho + h))
    em, _ = functional(jnp.asarray(rho - h))
    np.testing.assert_allclose(v, (np.asarray(ep) - np.asarray(em)) / (2 * h), rtol=1e-7)


@pytest.mark.parametrize("functional", [lda_x, lda_pw92], ids=["x", "pw92"])
def test_finite_at_zero_density_including_gradients(functional):
    """Vacuum: value and potential finite at rho = 0 and at tiny rho, and autodiff of the
    total energy through rho = 0 gives no NaN (double-where guard)."""
    rho = jnp.asarray([0.0, 1e-300, 1e-30, 1e-14, 1e-8])
    e, v = functional(rho)
    assert np.all(np.isfinite(np.asarray(e))) and np.all(np.isfinite(np.asarray(v)))
    assert float(e[0]) == 0.0
    g = jax.grad(lambda r: jnp.sum(functional(r)[0]))(rho)
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(g, v, rtol=1e-12, atol=1e-300)


def test_lda_is_configurable_and_lda_pw92_is_exchange_plus_correlation():
    assert isinstance(lda_pw92, LDA) and isinstance(lda_x, LDA)
    rho = rho_of_rs(RS)
    assert np.all(np.asarray(lda_pw92.eps(rho)) < np.asarray(lda_x.eps(rho)))


def test_double_precision():
    e, v = lda_pw92(jnp.asarray(rho_of_rs(RS)))
    assert e.dtype == jnp.float64 and v.dtype == jnp.float64
