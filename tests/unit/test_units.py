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


# --- Milestone 2a: wavenumbers and the proton mass (theory post 02a, 'Scope') --------


def test_codata_2022_constants_2a():
    """CODATA 2022: E_h / hc = 2.194 746 313 6314e5 cm^-1; m_p / m_e = 1836.152 673 426."""
    assert units.HARTREE_IN_INVERSE_CM == 219474.63136314
    assert units.PROTON_MASS_IN_ELECTRON_MASS == 1836.152673426


@pytest.mark.parametrize(
    ("value", "src", "dst", "expected"),
    [
        (1.0, "Hartree", "cm-1", 219474.63136314),
        (219474.63136314, "cm-1", "Hartree", 1.0),
        (1.0, "eV", "cm-1", 219474.63136314 / 27.211386245981),
    ],
)
def test_convert_wavenumbers(value, src, dst, expected):
    np.testing.assert_allclose(units.convert(value, src, dst), expected, rtol=1e-15)


@given(st.floats(-1e6, 1e6, allow_nan=False), st.sampled_from([("Hartree", "cm-1"), ("eV", "cm-1")]))
def test_wavenumber_round_trip(x, pair):
    a, b = pair
    np.testing.assert_allclose(units.convert(units.convert(x, a, b), b, a), x, rtol=1e-14, atol=1e-300)


def test_harmonic_frequency_of_known_oscillator():
    """omega = sqrt(k / mu) in Ha for k in Ha/Bohr^2 and mu in electron masses. For H2,
    mu = m_p / 2; k = 0.37 Ha/Bohr^2 gives about 4400 cm^-1 (the experimental
    omega_e is 4401 cm^-1)."""
    mu = units.PROTON_MASS_IN_ELECTRON_MASS / 2
    omega = units.convert(np.sqrt(0.37 / mu), "Hartree", "cm-1")
    assert 4300 < float(omega) < 4500


# --- pressure (Milestone 2b validation: bulk modulus) -------------------------------


def test_atomic_unit_of_pressure_in_gpa():
    """E_h / a_0^3 from the CODATA 2022 E_h and a_0. The CODATA 2018 value of the
    atomic unit of pressure, 2.9421015697e13 Pa, differs by 2e-9 relative because
    a_0 changed by 6.8e-10 between the two adjustments."""
    e_h, a0 = 4.3597447222060e-18, 0.529177210544e-10  # J, m
    np.testing.assert_allclose(units.HARTREE_PER_BOHR3_IN_GPA, e_h / a0**3 * 1e-9, rtol=1e-14)
    np.testing.assert_allclose(units.HARTREE_PER_BOHR3_IN_GPA, 29421.015697, rtol=5e-9)
    np.testing.assert_allclose(float(units.convert(1.0, "Hartree/Bohr^3", "GPa")), units.HARTREE_PER_BOHR3_IN_GPA, rtol=1e-15)


@given(st.floats(-1e3, 1e3, allow_nan=False))
def test_pressure_round_trip(x):
    back = units.convert(units.convert(x, "GPa", "Hartree/Bohr^3"), "Hartree/Bohr^3", "GPa")
    np.testing.assert_allclose(float(back), x, rtol=1e-14, atol=1e-12)


def test_pressure_does_not_convert_to_energy():
    with pytest.raises(ValueError):
        units.convert(1.0, "GPa", "Hartree")
