"""Analysis helpers (1b API decisions), validated on cases with closed-form answers."""

import numpy as np
import pytest

from jaxdft import Kinetic, PlaneWaveBasis, band_structure, empty, fcc_lattice
from jaxdft.analysis.bands import band_minimum_on_segment, valence_band_maximum
from jaxdft.analysis.symmetry import group_levels, inversion_parities
from support.free_electron import A_SI, e0

LAT = fcc_lattice(A_SI, unit="Bohr")
EMPTY = empty(LAT)
T = 2 * np.pi / A_SI


def to_frac(cart_in_2pi_over_a):
    return np.linalg.solve(np.asarray(LAT.reciprocal).T, T * np.asarray(cart_in_2pi_over_a, dtype=float))


def test_group_levels():
    e = np.array([0.0, 1.0, 1.0 + 1e-10, 2.0, 2.0, 2.0 + 5e-9, 3.0])
    levels = group_levels(e, atol=1e-8)
    assert [d for _, d in levels] == [1, 2, 3, 1]
    np.testing.assert_allclose([x for x, _ in levels], [0.0, 1.0, 2.0, 3.0], atol=1e-8)


def test_valence_band_maximum():
    ks = np.array([to_frac([0, 0, 0]), to_frac([0, 1, 0]), to_frac([0.5, 0.5, 0.5])])
    basis = PlaneWaveBasis.build(LAT, ks, 2.0)
    bands = band_structure(EMPTY, basis, n_bands=4)
    assert valence_band_maximum(bands, n_valence=2) == pytest.approx(float(np.max(bands.eigenvalues[:, 1])), abs=0)


def test_band_minimum_on_segment_free_electron():
    """Band 0 of the empty lattice near Gamma is 1/2 |k|^2: along a straight segment its
    minimum is the point closest to Gamma, here at t = 0.4."""
    ka, kb = to_frac([-0.2, 0.1, 0.05]), to_frac([0.3, 0.1, 0.05])
    t, energy = band_minimum_on_segment(EMPTY, ka, kb, band=0, ecut=1.0, terms=(Kinetic(),))
    assert abs(t - 0.4) < 1e-5
    np.testing.assert_allclose(energy, 0.5 * T**2 * (0.1**2 + 0.05**2), atol=1e-10)


def test_band_minimum_at_an_endpoint():
    ka, kb = to_frac([0.1, 0.0, 0.0]), to_frac([0.4, 0.0, 0.0])
    t, energy = band_minimum_on_segment(EMPTY, ka, kb, band=0, ecut=1.0, terms=(Kinetic(),))
    assert t < 1e-5
    np.testing.assert_allclose(energy, 0.5 * (0.1 * T) ** 2, atol=1e-10)


def test_inversion_parities_free_electron_gamma():
    """At Gamma, inversion maps G -> -G. The single G = 0 level is even; the 8-fold and
    6-fold shells are permuted with no fixed vector, so their mean parity is 0."""
    basis = PlaneWaveBasis.build(LAT, np.zeros((1, 3)), 4.5 * e0(A_SI))
    bands = band_structure(EMPTY, basis, n_bands=int(basis.n_pw[0]))
    out = inversion_parities(bands, k_index=0, atol=1e-8)
    assert [(d, p) for _, d, p in out[:3]] == [(1, 1.0), (8, 0.0), (6, 0.0)]
    np.testing.assert_allclose([x for x, _, _ in out[:3]], [0.0, 3 * e0(A_SI), 4 * e0(A_SI)], atol=1e-10)


def test_inversion_parities_require_a_time_reversal_invariant_k():
    basis = PlaneWaveBasis.build(LAT, to_frac([0.1, 0.2, 0.3])[None], 1.0)
    bands = band_structure(EMPTY, basis, n_bands=2)
    with pytest.raises(ValueError):
        inversion_parities(bands, k_index=0)
