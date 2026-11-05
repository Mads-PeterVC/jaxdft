"""Checks of the test-local reference itself, independent of the package.

If these fail, the reference (or the hand-derived table in the theory post) is
wrong, not the package.
"""

import numpy as np
import pytest

from support.free_electron import (
    A_SI,
    DEGENERACY_TABLE,
    SPECIAL_POINTS_FCC,
    SPECIAL_POINTS_FCC_CART,
    e0,
    fcc_vectors,
    free_electron_levels,
    group_levels,
    reciprocal,
)
from support.tolerances import DEGENERACY_ATOL


def test_fractional_and_cartesian_special_points_agree():
    A = fcc_vectors(A_SI)
    B = reciprocal(A)
    for label, f in SPECIAL_POINTS_FCC.items():
        cart = np.asarray(f) @ B / (2 * np.pi / A_SI)
        np.testing.assert_allclose(cart, SPECIAL_POINTS_FCC_CART[label], atol=1e-14)


@pytest.mark.parametrize("label", sorted(DEGENERACY_TABLE))
def test_hand_derived_degeneracy_table(label):
    A = fcc_vectors(A_SI)
    table = DEGENERACY_TABLE[label]
    emax = (float(table[-1][0]) + 0.5) * e0(A_SI)
    levels = group_levels(free_electron_levels(A, SPECIAL_POINTS_FCC[label], emax), DEGENERACY_ATOL)
    assert [n for _, n in levels] == [n for _, n in table]
    np.testing.assert_allclose([x / e0(A_SI) for x, _ in levels], [float(x) for x, _ in table], atol=1e-12)
