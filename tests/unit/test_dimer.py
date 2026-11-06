"""The dimer builder (engineering log, 2a API: 'dimer_in_box')."""

import numpy as np
import pytest

from jaxdft import Structure, dimer_in_box


def test_dimer_geometry_bohr():
    s = dimer_in_box("H", 1.45, 10.0, axis=2)
    assert isinstance(s, Structure)
    assert s.species == ("H", "H")
    np.testing.assert_allclose(s.lattice.vectors, 10.0 * np.eye(3), atol=0)
    r = np.asarray(s.cartesian_positions)
    np.testing.assert_allclose(r.mean(axis=0), [5.0, 5.0, 5.0], atol=1e-14)
    np.testing.assert_allclose(r[1] - r[0], [0.0, 0.0, 1.45], atol=1e-14)
    np.testing.assert_allclose(s.weights, [1.0, 1.0])


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_dimer_axis(axis):
    r = np.asarray(dimer_in_box("H", 1.4, 8.0, axis=axis).cartesian_positions)
    expected = np.zeros(3)
    expected[axis] = 1.4
    np.testing.assert_allclose(r[1] - r[0], expected, atol=1e-14)


def test_dimer_centre_is_a_grid_point_of_any_even_grid():
    """Fractional centre (1/2, 1/2, 1/2): index N/2 on an even grid (D4h tests)."""
    s = dimer_in_box("H", 1.45, 10.0)
    np.testing.assert_allclose(np.asarray(s.positions).mean(axis=0), [0.5, 0.5, 0.5], atol=1e-15)


def test_dimer_angstrom_input_converts_once():
    a0 = 0.529177210544
    s = dimer_in_box("H", 0.74, 6.0, unit="Angstrom")
    np.testing.assert_allclose(s.lattice.vectors, 6.0 / a0 * np.eye(3), rtol=1e-15)
    r = np.asarray(s.cartesian_positions)
    np.testing.assert_allclose(np.linalg.norm(r[1] - r[0]), 0.74 / a0, rtol=1e-14)


def test_heteronuclear_species_pair():
    assert dimer_in_box(("H", "Li"), 3.0, 12.0).species == ("H", "Li")


def test_bond_longer_than_box_raises():
    with pytest.raises(ValueError):
        dimer_in_box("H", 11.0, 10.0)
