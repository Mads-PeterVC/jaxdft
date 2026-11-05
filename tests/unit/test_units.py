"""Unit conversions at the boundary (theory post 1b, 'Units at the boundary').
CLAUDE.md: every unit conversion needs a test."""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from jaxdft import units


def test_codata_2022_constants():
    """Values from NIST, CODATA 2022 (cited in the theory post)."""
    assert units.BOHR_IN_ANGSTROM == 0.529177210544
    assert units.HARTREE_IN_EV == 27.211386245981
    assert units.RYDBERG_IN_HARTREE == 0.5


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1.0, "Bohr", "Angstrom", 0.529177210544),
        (0.529177210544, "Angstrom", "Bohr", 1.0),
        (1.0, "Rydberg", "Hartree", 0.5),
        (1.0, "Hartree", "Rydberg", 2.0),
        (1.0, "Hartree", "eV", 27.211386245981),
        (27.211386245981, "eV", "Hartree", 1.0),
        (1.0, "Rydberg", "eV", 0.5 * 27.211386245981),
        (3.0, "Bohr", "Bohr", 3.0),
    ],
)
def test_convert_direction_and_value(value, src, dst, expected):
    np.testing.assert_allclose(units.convert(value, src, dst), expected, rtol=1e-15)


@given(st.floats(-1e6, 1e6, allow_nan=False), st.sampled_from([("Bohr", "Angstrom"), ("Hartree", "eV"), ("Rydberg", "Hartree"), ("Rydberg", "eV")]))
def test_round_trip(x, pair):
    a, b = pair
    np.testing.assert_allclose(units.convert(units.convert(x, a, b), b, a), x, rtol=1e-14, atol=1e-300)


def test_arrays_stay_double_precision():
    y = units.convert(np.ones(3, dtype=np.float32), "Angstrom", "Bohr")
    assert jnp.asarray(y).dtype == jnp.float64


def test_mismatched_dimensions_and_unknown_units_raise():
    with pytest.raises(ValueError):
        units.convert(1.0, "Bohr", "eV")
    with pytest.raises(ValueError):
        units.convert(1.0, "angstrom", "Bohr")  # case-sensitive, no guessing
    with pytest.raises(ValueError):
        units.convert(1.0, "nm", "Bohr")
