"""Lattice invariants (theory post: 'Invariants', Lattice)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings

from jaxdft import Lattice, assert_double_precision
from support.free_electron import fcc_vectors, reciprocal
from support.strategies import cells


@given(cells())
def test_reciprocal_duality(A):
    lat = Lattice(A)
    np.testing.assert_allclose(lat.vectors @ lat.reciprocal.T, 2 * np.pi * np.eye(3), atol=1e-12)
    np.testing.assert_allclose(lat.reciprocal, reciprocal(A), rtol=1e-12, atol=1e-14)


@given(cells())
def test_volume_and_brillouin_zone_volume(A):
    lat = Lattice(A)
    np.testing.assert_allclose(lat.volume, abs(np.linalg.det(A)), rtol=1e-13)
    np.testing.assert_allclose(abs(np.linalg.det(lat.reciprocal)), (2 * np.pi) ** 3 / lat.volume, rtol=1e-12)


@settings(max_examples=20)
@given(cells())
def test_plane_waves_are_lattice_periodic(A):
    lat = Lattice(A)
    m = np.array(list(itertools.product(range(-2, 3), repeat=3)))
    G = m @ np.asarray(lat.reciprocal)
    R = m @ np.asarray(lat.vectors)
    phase = np.exp(1j * G @ R.T)
    np.testing.assert_allclose(phase, 1.0, atol=1e-11)


def test_fcc_constructor():
    a = 10.26
    lat = Lattice.fcc(a)
    np.testing.assert_allclose(lat.vectors, fcc_vectors(a), rtol=1e-15)
    np.testing.assert_allclose(lat.volume, a**3 / 4, rtol=1e-14)
    expected_B = 2 * np.pi / a * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])
    np.testing.assert_allclose(lat.reciprocal, expected_B, rtol=1e-14)


def test_lattice_is_double_precision_even_from_float32_input():
    lat = Lattice(np.asarray(fcc_vectors(10.0), dtype=np.float32))
    assert lat.vectors.dtype == jnp.float64
    assert_double_precision(lat)
    assert_double_precision((lat.reciprocal, lat.volume))


def test_lattice_is_a_pytree_through_jit_and_grad():
    """Cell derivatives must be possible later: grad of a function of a Lattice
    returns a Lattice, and d|det A|/dA = |det A| A^{-T}."""
    lat = Lattice.fcc(10.26)
    assert jax.jit(lambda L: L.volume)(lat) == lat.volume
    g = jax.grad(lambda L: L.volume)(lat)
    assert isinstance(g, Lattice)
    A = np.asarray(lat.vectors)
    np.testing.assert_allclose(g.vectors, abs(np.linalg.det(A)) * np.linalg.inv(A).T, rtol=1e-12)
