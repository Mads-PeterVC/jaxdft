"""Structure and builders (theory post 1b, 'The diamond structure'; 1b API decisions)."""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft import Lattice, Structure, assert_double_precision, diamond, empty, fcc_lattice
from support.free_electron import fcc_vectors
from support.strategies import cells

A0 = 0.529177210544
positions_strategy = hnp.arrays(float, st.tuples(st.integers(1, 5), st.just(3)), elements=st.floats(-1.0, 1.0))


@given(cells(), positions_strategy)
def test_from_cartesian_stores_fractional_and_round_trips(A, frac):
    cart = frac @ A
    s = Structure.from_cartesian(A, cart, ["X"] * len(frac), unit="Bohr")
    np.testing.assert_allclose(s.positions, frac, atol=1e-12)
    np.testing.assert_allclose(s.cartesian_positions, cart, atol=1e-11)
    np.testing.assert_allclose(s.lattice.vectors, A, rtol=1e-15)


@settings(max_examples=20)
@given(cells(), positions_strategy)
# Regression: subnormal cell entries are flushed to zero by the conversion arithmetic.
@example(A=np.array([[6.0, 1.33504432e-310, 1.33504432e-310], [1.33504432e-310, 6.0, 1.33504432e-310], [1.33504432e-310, 1.33504432e-310, 6.0]]), frac=np.zeros((1, 3)))
def test_from_cartesian_units_are_consistent(A, frac):
    cart = frac @ A
    s_bohr = Structure.from_cartesian(A, cart, ["X"] * len(frac), unit="Bohr")
    s_ang = Structure.from_cartesian(A * A0, cart * A0, ["X"] * len(frac), unit="Angstrom")
    # Conversion rounding scales with the size of the cell vectors, not of each entry,
    # so near-zero entries need an absolute floor; relative-only cannot accept 0 vs 1e-310.
    np.testing.assert_allclose(s_ang.lattice.vectors, s_bohr.lattice.vectors, rtol=1e-14, atol=1e-14 * np.abs(A).max())
    np.testing.assert_allclose(s_ang.positions, s_bohr.positions, atol=1e-12)


def test_from_cartesian_requires_a_unit():
    assert inspect.signature(Structure.from_cartesian).parameters["unit"].default is inspect.Parameter.empty
    with pytest.raises(ValueError):
        Structure.from_cartesian(np.eye(3), np.zeros((1, 3)), ["X"], unit="nm")


def test_constructor_defaults_and_validation():
    lat = Lattice(5.0 * np.eye(3))
    s = Structure(lat, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], ["A", "B"])
    assert s.species == ("A", "B") and s.n_atoms == 2
    np.testing.assert_array_equal(s.weights, [1.0, 1.0])
    assert_double_precision(s)
    with pytest.raises(ValueError):
        Structure(lat, [[0.0, 0.0]], ["A"])  # positions must be (n, 3)
    with pytest.raises(ValueError):
        Structure(lat, [[0.0, 0.0, 0.0]], ["A", "B"])  # species count
    with pytest.raises(ValueError):
        Structure(lat, [[0.0, 0.0, 0.0]], ["A"], weights=[1.0, 1.0])  # weight count


def test_weights_are_not_range_checked():
    """Physical range is [0, 1], but finite differences at 1 step past it (API decision)."""
    s = Structure(Lattice(np.eye(3)), [[0.0, 0.0, 0.0]], ["A"], weights=[1.001])
    assert float(s.weights[0]) == 1.001


def test_positions_are_fractional_under_cell_change():
    """Positions are stored fractional: changing the cell moves Cartesian positions."""
    s = Structure(Lattice(5.0 * np.eye(3)), [[0.25, 0.0, 0.0]], ["A"])
    scaled = Structure(Lattice(2 * s.lattice.vectors), s.positions, s.species, s.weights)
    np.testing.assert_allclose(scaled.positions, s.positions)
    np.testing.assert_allclose(scaled.cartesian_positions, 2 * s.cartesian_positions)


def test_structure_is_a_pytree_with_static_species():
    s = diamond(5.43, "Si", unit="Angstrom")
    leaves = jax.tree_util.tree_leaves(s)
    assert all(isinstance(x, jax.Array) for x in leaves)
    assert jax.jit(lambda t: t.cartesian_positions.sum())(s) == pytest.approx(float(s.cartesian_positions.sum()), rel=1e-14)
    g = jax.grad(lambda t: jnp.sum(t.cartesian_positions) + jnp.sum(t.weights))(s)
    assert isinstance(g, Structure) and g.species == s.species
    A = np.asarray(s.lattice.vectors)
    np.testing.assert_allclose(g.positions, np.tile(A.sum(axis=1), (2, 1)), rtol=1e-13)
    np.testing.assert_allclose(g.weights, [1.0, 1.0])


def test_fcc_lattice_builder():
    a = 5.43
    np.testing.assert_allclose(fcc_lattice(a, unit="Angstrom").vectors, fcc_vectors(a / A0), rtol=1e-15)
    np.testing.assert_allclose(fcc_lattice(a / A0, unit="Bohr").vectors, fcc_vectors(a / A0), rtol=1e-15)
    with pytest.raises(TypeError):
        fcc_lattice(a)  # unit is required


def test_diamond_builder():
    s = diamond(5.43, "Si", unit="Angstrom")
    assert s.species == ("Si", "Si")
    np.testing.assert_allclose(s.lattice.vectors, fcc_vectors(5.43 / A0), rtol=1e-15)
    np.testing.assert_allclose(s.positions, [[0.125] * 3, [-0.125] * 3], atol=0)
    np.testing.assert_allclose(s.cartesian_positions[0], 5.43 / A0 / 8 * np.ones(3), rtol=1e-14)
    np.testing.assert_array_equal(s.weights, [1.0, 1.0])


def test_empty_builder():
    lat = fcc_lattice(10.0, unit="Bohr")
    s = empty(lat)
    assert s.n_atoms == 0 and s.positions.shape == (0, 3) and s.weights.shape == (0,)
    assert s.lattice is lat
