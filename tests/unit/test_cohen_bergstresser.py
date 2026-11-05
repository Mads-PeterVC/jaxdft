"""The Cohen-Bergstresser Si pseudopotential (theory post 1b, 'The Cohen-Bergstresser
parameterisation', 'Consequence: the form factors are tied to one cell')."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft.pseudo.cohen_bergstresser import CohenBergstresser, silicon

A0 = 0.529177210544
A_CB = 5.43 / A0
T = 2 * np.pi / A_CB
OMEGA_AT = A_CB**3 / 8  # volume per atom in diamond: Omega / 2 = (a^3 / 4) / 2


def test_publication_data_are_stored_as_published():
    cb = silicon()
    assert isinstance(cb, CohenBergstresser)
    assert cb.lattice_constant_angstrom == 5.43
    assert cb.form_factors_ry == {3: -0.21, 8: 0.04, 11: 0.08}
    np.testing.assert_allclose(cb.lattice_constant, A_CB, rtol=1e-15)


def test_physical_shell_radii():
    np.testing.assert_allclose(silicon().shell_q, np.sqrt([3.0, 8.0, 11.0]) * T, rtol=1e-14)


@pytest.mark.parametrize("shell", [3, 8, 11])
def test_local_form_factor_on_tabulated_shells(shell):
    """v_hat = V_S * Omega_at, converted Ry -> Ha (theory post normalisation)."""
    v_s_ha = 0.5 * {3: -0.21, 8: 0.04, 11: 0.08}[shell]
    np.testing.assert_allclose(silicon().local(np.sqrt(shell) * T), v_s_ha * OMEGA_AT, rtol=1e-13)


def test_local_form_factor_elsewhere_is_zero():
    """G = 0 (V_0 = 0 by convention), beyond the support, and off-table values are 0;
    the off-table case is refused by validate, not by local (it must stay traceable)."""
    cb = silicon()
    q = jnp.array([0.0, np.sqrt(4.0) * T, np.sqrt(5.0) * T, np.sqrt(12.0) * T, 10.0])
    np.testing.assert_array_equal(cb.local(q), 0.0)


def test_local_is_vectorised_jittable_and_double():
    cb = silicon()
    q = jnp.sqrt(jnp.array([3.0, 8.0, 11.0])) * T
    out = jax.jit(cb.local)(q)
    assert out.shape == (3,) and out.dtype == jnp.float64
    np.testing.assert_allclose(out, cb.local(q))


def test_support_radius_brackets_the_last_shell():
    s = silicon().local_support
    assert np.sqrt(11.0) * T < s < np.sqrt(12.0) * T


def test_validate_accepts_tabulated_and_trivial_q():
    silicon().validate(jnp.array([0.0, *np.sqrt([3.0, 8.0, 11.0]) * T, 1e3]))


@pytest.mark.parametrize("q_over_t", [np.sqrt(4.0), np.sqrt(5.0), np.sqrt(3.0) * (1 + 1e-6), np.sqrt(3.0) * 5.43 / 5.5])
def test_validate_refuses_untabulated_q_within_support(q_over_t):
    """Shell 4 is unknown (CB never fitted it; it vanishes in diamond only because S = 0),
    and an isotropically scaled cell moves q off the physical shells."""
    with pytest.raises(ValueError):
        silicon().validate(jnp.array([q_over_t * T]))
