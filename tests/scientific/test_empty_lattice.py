"""Milestone 1a acceptance tests: fcc empty-lattice bands against the analytic folded
free-electron bands (validation ladder rung 1). See docs/theory/01a_empty_lattice.qmd.

Reference values come from tests/support/free_electron.py, an independent NumPy
enumeration, and the hand-derived degeneracy table in the theory post.
Tolerances and their justification: tests/support/tolerances.py.
"""

import numpy as np
import pytest

from jaxdft import (
    Bands,
    Lattice,
    PlaneWaveBasis,
    assert_double_precision,
    band_structure,
    free_electron_bands,
    kpath,
    special_points_fcc,
)
from support.free_electron import (
    A_SI,
    DEGENERACY_TABLE,
    SPECIAL_POINTS_FCC,
    e0,
    ecut_between_shells,
    fcc_vectors,
    free_electron_levels,
    group_levels,
    lowest_free_electron_levels,
)
from support.tolerances import DEGENERACY_ATOL, EIGENVALUE_ATOL, RESIDUAL_RTOL

LABELS = ["G", "X", "L", "W", "K", "U"]
SI_PATH = [["L", "G", "X", "U"], ["K", "G"]]
# Above the sixth tabulated level at every high-symmetry point (max 12 E0 at Gamma).
TARGET_ECUT = 12.5 * e0(A_SI)


@pytest.fixture(scope="module")
def lattice():
    return Lattice.fcc(A_SI)


@pytest.fixture(scope="module")
def special_bands(lattice):
    """One calculation per high-symmetry point, each returning all N_PW(k) bands,
    so every eigenvalue below ecut is available (a shared batch would be limited to
    min_k N_PW bands)."""
    ks = np.array([SPECIAL_POINTS_FCC[label] for label in LABELS])
    ecut = ecut_between_shells(fcc_vectors(A_SI), ks, TARGET_ECUT)
    out = {}
    for label, k in zip(LABELS, ks):
        basis = PlaneWaveBasis.build(lattice, k[None], ecut)
        out[label] = band_structure(lattice, basis, n_bands=int(basis.n_pw[0]))
    return out


@pytest.fixture(scope="module")
def path_bands(lattice):
    path = kpath(lattice, special_points_fcc(), SI_PATH, n_points=60)
    ks = np.asarray(path.kpoints)
    ecut = ecut_between_shells(fcc_vectors(A_SI), ks, TARGET_ECUT)
    basis = PlaneWaveBasis.build(lattice, ks, ecut)
    return band_structure(lattice, basis, n_bands=int(np.min(basis.n_pw)))


def assert_matches_folded_bands(bands: Bands):
    """Every returned eigenvalue <= ecut equals the analytic folded band, and none is
    missing (up to the n_bands requested)."""
    A = fcc_vectors(A_SI)
    ecut = bands.basis.ecut
    for k, e in zip(np.asarray(bands.kpoints), np.asarray(bands.eigenvalues)):
        ref = free_electron_levels(A, k, ecut)[: len(e)]
        got = e[e <= ecut]
        assert len(got) == len(ref), f"k={k}: {len(got)} eigenvalues below ecut, expected {len(ref)}"
        np.testing.assert_allclose(got, ref, atol=EIGENVALUE_ATOL)


@pytest.mark.parametrize("label", LABELS)
def test_high_symmetry_points_match_folded_bands(special_bands, label):
    assert_matches_folded_bands(special_bands[label])


@pytest.mark.parametrize("label", LABELS)
def test_degeneracy_table(special_bands, label):
    bands = special_bands[label]
    e = np.asarray(bands.eigenvalues[0])
    e = e[e <= bands.basis.ecut]
    levels = group_levels(e, DEGENERACY_ATOL)[: len(DEGENERACY_TABLE[label])]
    assert [n for _, n in levels] == [n for _, n in DEGENERACY_TABLE[label]]
    np.testing.assert_allclose(
        [x for x, _ in levels], [float(x) * e0(A_SI) for x, _ in DEGENERACY_TABLE[label]], atol=EIGENVALUE_ATOL
    )


def test_K_and_U_spectra_identical(special_bands):
    ecut = special_bands["K"].basis.ecut
    eK, eU = (np.asarray(special_bands[p].eigenvalues[0]) for p in ("K", "U"))
    eK, eU = eK[eK <= ecut], eU[eU <= ecut]
    assert len(eK) == len(eU)
    np.testing.assert_allclose(eK, eU, atol=EIGENVALUE_ATOL)


def test_band_path_matches_folded_bands(path_bands):
    assert_matches_folded_bands(path_bands)


def test_vmapped_path_matches_per_k(lattice, path_bands):
    """The batched computation equals independent single-k computations
    (each with its own, smaller, padding length)."""
    ecut = path_bands.basis.ecut
    for i in [0, 7, 23, 41, 59]:
        k = np.asarray(path_bands.kpoints[i])[None]
        single = band_structure(lattice, PlaneWaveBasis.build(lattice, k, ecut), n_bands=path_bands.eigenvalues.shape[1])
        np.testing.assert_allclose(single.eigenvalues[0], path_bands.eigenvalues[i], atol=EIGENVALUE_ATOL)


def test_bands_result_structure(path_bands):
    n_k, n_bands = path_bands.eigenvalues.shape
    basis = path_bands.basis
    assert isinstance(basis, PlaneWaveBasis)
    assert path_bands.eigenvectors.shape == (n_k, basis.n_pad, n_bands)
    np.testing.assert_array_equal(path_bands.kpoints, basis.kpoints)
    assert_double_precision(path_bands)
    assert np.all(np.diff(np.asarray(path_bands.eigenvalues), axis=1) >= 0)


def test_eigenvectors_orthonormal_and_zero_on_padding(path_bands):
    V = np.asarray(path_bands.eigenvectors)
    mask = np.asarray(path_bands.basis.mask)
    n_bands = V.shape[2]
    for i in range(V.shape[0]):
        np.testing.assert_allclose(V[i].conj().T @ V[i], np.eye(n_bands), atol=1e-12)
        np.testing.assert_array_equal(V[i][~mask[i]], 0)


def test_residuals(lattice, path_bands):
    """||H c - eps c|| <= tau ||H||, with H the kinetic diagonal ||H_phys|| <= ecut."""
    from support.free_electron import reciprocal

    B = reciprocal(fcc_vectors(A_SI))
    basis = path_bands.basis
    for i in range(0, path_bands.eigenvalues.shape[0], 10):
        mask = np.asarray(basis.mask[i])
        m = np.asarray(basis.miller[i])[mask]
        kin = 0.5 * np.sum((np.asarray(basis.kpoints[i]) @ B + m @ B) ** 2, axis=1)
        c = np.asarray(path_bands.eigenvectors[i])[mask]
        eps = np.asarray(path_bands.eigenvalues[i])
        resid = np.linalg.norm(kin[:, None] * c - c * eps[None, :], axis=0)
        assert np.all(resid <= RESIDUAL_RTOL * basis.ecut)


def test_library_analytic_reference_matches_independent_enumeration(lattice):
    """The package's analytic reference (used by the validation docs) agrees with
    the test-local enumeration, at the special points and along the path."""
    path = kpath(lattice, special_points_fcc(), SI_PATH, n_points=60)
    ks = np.concatenate([np.array([SPECIAL_POINTS_FCC[label] for label in LABELS]), np.asarray(path.kpoints)])
    n_bands = 40
    got = np.asarray(free_electron_bands(lattice, ks, n_bands))
    assert got.shape == (len(ks), n_bands)
    assert_double_precision(got)
    for k, e in zip(ks, got):
        np.testing.assert_allclose(e, lowest_free_electron_levels(fcc_vectors(A_SI), k, n_bands), atol=EIGENVALUE_ATOL)
