"""Symmetry, periodicity, scaling and variational invariants of the empty-lattice spectrum.

Only eigenvalues below ecut are compared: those are exact and complete (theory post,
'Kinetic-energy cutoff', property 2).
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft import Lattice, PlaneWaveBasis, band_structure, empty, fcc_lattice
from support.budget import examples
from support.free_electron import A_SI, ecut_between_shells, fcc_vectors, oh_operations, reciprocal
from support.strategies import cells, kfracs
from support.tolerances import EIGENVALUE_ATOL

TARGET_ECUTS = st.floats(1.0, 2.5)


def spectra_below_cutoff(lat, ks, ecut):
    basis = PlaneWaveBasis.build(lat, np.asarray(ks), ecut)
    e = np.asarray(band_structure(empty(lat), basis, n_bands=int(np.min(basis.n_pw))).eigenvalues)
    return [row[row <= ecut] for row in e]


@settings(max_examples=examples(10))
@given(cells(), kfracs(), st.tuples(*[st.integers(-2, 2)] * 3), TARGET_ECUTS)
def test_spectrum_periodic_in_reciprocal_lattice(A, k, m0, target):
    ks = np.array([k, k + np.array(m0)])
    ecut = ecut_between_shells(A, ks, target)
    e, e_shift = spectra_below_cutoff(Lattice(A), ks, ecut)
    assert len(e) == len(e_shift)
    np.testing.assert_allclose(e, e_shift, atol=EIGENVALUE_ATOL)


@settings(max_examples=examples(5))
@given(kfracs(-0.5, 0.5), TARGET_ECUTS)
def test_spectrum_invariant_under_oh_in_fcc(k, target):
    A = fcc_vectors(A_SI)
    B = reciprocal(A)
    images = np.array([np.linalg.solve(B.T, S @ (k @ B)) for S in oh_operations()])
    ecut = ecut_between_shells(A, images, target)
    spectra = spectra_below_cutoff(Lattice(A), images, ecut)
    for e in spectra[1:]:
        assert len(e) == len(spectra[0])
        np.testing.assert_allclose(e, spectra[0], atol=EIGENVALUE_ATOL)


@settings(max_examples=examples(10))
@given(kfracs(-0.5, 0.5), TARGET_ECUTS, st.floats(6.0, 14.0))
def test_scaling_with_lattice_constant(k, target, a2):
    """a^2 eps is independent of a at fixed fractional k, with ecut scaled as a^-2."""
    a1 = A_SI
    ecut1 = ecut_between_shells(fcc_vectors(a1), [k], target)
    ecut2 = ecut1 * (a1 / a2) ** 2
    (e1,) = spectra_below_cutoff(fcc_lattice(a1, unit="Bohr"), [k], ecut1)
    (e2,) = spectra_below_cutoff(fcc_lattice(a2, unit="Bohr"), [k], ecut2)
    assert len(e1) == len(e2)
    np.testing.assert_allclose(a1**2 * e1, a2**2 * e2, atol=EIGENVALUE_ATOL * a1**2)


@settings(max_examples=examples(10))
@given(cells(), kfracs(), TARGET_ECUTS, st.floats(1.2, 2.0))
def test_eigenvalues_non_increasing_with_cutoff(A, k, target, factor):
    """Courant-Fischer on nested bases: the n-th eigenvalue never increases with ecut."""
    lat = Lattice(A)

    def all_bands(ecut):
        basis = PlaneWaveBasis.build(lat, k[None], ecut)
        return np.asarray(band_structure(empty(lat), basis, n_bands=int(basis.n_pw[0])).eigenvalues[0])

    e_lo = all_bands(ecut_between_shells(A, [k], target))
    e_hi = all_bands(ecut_between_shells(A, [k], target * factor))
    n = len(e_lo)
    assert np.all(e_hi[:n] <= e_lo + EIGENVALUE_ATOL)
