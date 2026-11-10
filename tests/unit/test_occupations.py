"""Aufbau occupations with a one-sided slope at integer electron counts
(theory post 02a, 'Occupations', 'Per-atom weights'; engineering log: side=)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft.scf.occupations import aufbau_occupations


@pytest.mark.parametrize(
    ("n_el", "n_bands", "expected"),
    [
        (2.0, 3, [2, 0, 0]),
        (2.5, 3, [2, 0.5, 0]),
        (1.3, 2, [1.3, 0]),
        (4.0, 3, [2, 2, 0]),
        (5.75, 4, [2, 2, 1.75, 0]),
        (0.0, 2, [0, 0]),
    ],
)
def test_values(n_el, n_bands, expected):
    for side in (-1, +1):
        f = aufbau_occupations(jnp.asarray(n_el), n_bands, side=side)
        np.testing.assert_allclose(f, expected, atol=1e-15)
        assert f.dtype == jnp.float64


def test_too_many_electrons_raises():
    with pytest.raises(ValueError):
        aufbau_occupations(jnp.asarray(6.5), 3)


@pytest.mark.parametrize(("side", "band"), [(-1, 0), (+1, 1)])
def test_one_sided_slope_at_integer_count(side, band):
    """d/dN sum_n f_n eps_n is eps_HOMO from the left and eps_LUMO from the right
    (Janak); the side picks which, at exactly N = 2."""
    eps = jnp.asarray([-0.4, -0.01, 0.06])
    g = jax.grad(lambda n: jnp.sum(aufbau_occupations(n, 3, side=side) * eps))(jnp.asarray(2.0))
    np.testing.assert_allclose(g, eps[band], atol=1e-15)


@pytest.mark.parametrize("side", [-1, +1])
def test_slope_away_from_integers_is_side_independent(side):
    eps = jnp.asarray([-0.4, -0.01, 0.06])
    g = jax.grad(lambda n: jnp.sum(aufbau_occupations(n, 3, side=side) * eps))(jnp.asarray(2.4))
    np.testing.assert_allclose(g, eps[1], atol=1e-15)


def test_invalid_side_raises():
    with pytest.raises(ValueError):
        aufbau_occupations(jnp.asarray(2.0), 3, side=0)
