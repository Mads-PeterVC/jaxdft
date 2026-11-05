"""Hamiltonian operator, padding and eigensolver invariants
(theory post: 'The kinetic operator', 'Batching over k-points', 'Invariants')."""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft import (
    Hamiltonian,
    Kinetic,
    Lattice,
    PlaneWaveBasis,
    apply_H,
    assert_double_precision,
    band_structure,
    dense_eigensolver,
    dense_matrix,
    hamiltonian,
)
from support.free_electron import A_SI, ecut_between_shells, fcc_vectors, reciprocal
from support.strategies import cells, kfracs
from support.terms import ConstantShift, MillerCoupling
from support.tolerances import EIGENVALUE_ATOL, RESIDUAL_RTOL

TARGET_ECUTS = st.floats(1.0, 2.5)
PAD_EXTRA = 6  # force genuine padding entries


def padded_basis(A, k, target, extra=PAD_EXTRA):
    ecut = ecut_between_shells(A, [k], target)
    lat = Lattice(A)
    b = PlaneWaveBasis.build(lat, np.asarray(k)[None], ecut)
    return lat, PlaneWaveBasis.build(lat, np.asarray(k)[None], ecut, n_pad=b.n_pad + extra)


def random_block(rng, n, m):
    return rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))


TERMS = {
    "kinetic": (Kinetic(),),
    "kinetic+coupling": (Kinetic(), MillerCoupling()),
}


@pytest.mark.parametrize("terms", TERMS.values(), ids=TERMS.keys())
@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS, seed=st.integers(0, 2**32 - 1))
def test_apply_H_agrees_with_dense_matrix(terms, A, k, target, seed):
    lat, basis = padded_basis(A, k, target)
    ham = hamiltonian(lat, basis[0], terms)
    assert isinstance(ham, Hamiltonian)
    H = dense_matrix(ham)
    assert H.shape == (basis.n_pad, basis.n_pad)
    assert_double_precision(H)
    psi = jnp.asarray(random_block(np.random.default_rng(seed), basis.n_pad, 4))
    Hpsi = apply_H(ham, psi)
    assert Hpsi.shape == psi.shape
    np.testing.assert_allclose(Hpsi, H @ psi, atol=1e-12 * float(jnp.abs(H).max()) * basis.n_pad)


@pytest.mark.parametrize("terms", TERMS.values(), ids=TERMS.keys())
@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS)
def test_dense_matrix_is_hermitian(terms, A, k, target):
    lat, basis = padded_basis(A, k, target)
    H = np.asarray(dense_matrix(hamiltonian(lat, basis[0], terms)))
    np.testing.assert_allclose(H, H.conj().T, atol=1e-14 * np.abs(H).max())


@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS)
def test_empty_lattice_physical_block_is_kinetic_diagonal(A, k, target):
    lat, basis = padded_basis(A, k, target)
    b = basis[0]
    H = np.asarray(dense_matrix(hamiltonian(lat, b, (Kinetic(),))))
    mask = np.asarray(b.mask)
    m = np.asarray(b.miller)[mask]
    expected = 0.5 * np.sum((k @ reciprocal(A) + m @ reciprocal(A)) ** 2, axis=1)
    phys = H[np.ix_(mask, mask)]
    np.testing.assert_allclose(np.diag(phys), expected, rtol=1e-13)
    np.testing.assert_array_equal(phys - np.diag(np.diag(phys)), 0)


@pytest.mark.parametrize("terms", TERMS.values(), ids=TERMS.keys())
@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS)
def test_padding_is_exactly_decoupled_with_bounded_sentinel(terms, A, k, target):
    """Zero rows/cols between padding and physical entries, even for a term that
    couples every pair of Miller indices; padded diagonal is one sentinel s with
    ecut < s <= 10 ecut."""
    lat, basis = padded_basis(A, k, target)
    b = basis[0]
    H = np.asarray(dense_matrix(hamiltonian(lat, b, terms)))
    mask = np.asarray(b.mask)
    pad = ~mask
    assert pad.sum() >= PAD_EXTRA
    np.testing.assert_array_equal(H[np.ix_(pad, mask)], 0)
    np.testing.assert_array_equal(H[np.ix_(mask, pad)], 0)
    block = H[np.ix_(pad, pad)]
    s = block[0, 0]
    np.testing.assert_array_equal(block, s * np.eye(pad.sum()))
    assert basis.ecut < s.real <= 10 * basis.ecut and s.imag == 0


@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS)
def test_non_diagonal_hamiltonian_eigenvalues(A, k, target):
    """Kinetic + a coupling defined outside the package: the bands equal the
    eigenvalues of the physical matrix built independently in NumPy."""
    lat, basis = padded_basis(A, k, target)
    b = basis[0]
    coupling = MillerCoupling()
    mask = np.asarray(b.mask)
    m = np.asarray(b.miller)[mask]
    kin = 0.5 * np.sum((k @ reciprocal(A) + m @ reciprocal(A)) ** 2, axis=1)
    H_ref = np.diag(kin) + coupling.reference_matrix(m)
    ref = np.linalg.eigvalsh(H_ref)
    # Only eigenvalues below ecut are guaranteed to lie below the padding sentinel.
    n = int(np.sum(ref < basis.ecut))
    assert n >= 1
    bands = band_structure(lat, basis, n_bands=int(b.n_pw), terms=(Kinetic(), coupling))
    np.testing.assert_allclose(bands.eigenvalues[0][:n], ref[:n], atol=EIGENVALUE_ATOL)


@settings(max_examples=10)
@given(A=cells(), ks=st.lists(kfracs(), min_size=1, max_size=3), target=TARGET_ECUTS, extra=st.integers(1, 20))
def test_bands_independent_of_n_pad(A, ks, target, extra):
    """Solver-independent leak test: different padding lengths give the same bands."""
    ks = np.array(ks)
    lat = Lattice(A)
    ecut = ecut_between_shells(A, ks, target)
    b1 = PlaneWaveBasis.build(lat, ks, ecut)
    b2 = PlaneWaveBasis.build(lat, ks, ecut, n_pad=b1.n_pad + extra)
    n_bands = int(np.min(b1.n_pw))
    terms = (Kinetic(), MillerCoupling())
    e1 = np.asarray(band_structure(lat, b1, n_bands=n_bands, terms=terms).eigenvalues)
    e2 = np.asarray(band_structure(lat, b2, n_bands=n_bands, terms=terms).eigenvalues)
    below = e1 < ecut  # bands guaranteed below the sentinel
    np.testing.assert_allclose(e1[below], e2[below], atol=EIGENVALUE_ATOL)
    assert np.all((e2 < ecut) == below)


@settings(max_examples=25)
@given(
    n=st.integers(2, 40),
    data=st.data(),
)
def test_dense_eigensolver_on_arbitrary_hermitian_operator(n, data):
    """The non-trivial path that the diagonal empty lattice does not exercise."""
    seed = data.draw(st.integers(0, 2**32 - 1))
    n_bands = data.draw(st.integers(1, n))
    rng = np.random.default_rng(seed)
    X = random_block(rng, n, n)
    M = (X + X.conj().T) / 2
    Mj = jnp.asarray(M)
    eps, vecs = dense_eigensolver(lambda Y: Mj @ Y, dim=n, n_bands=n_bands)
    assert eps.shape == (n_bands,) and vecs.shape == (n, n_bands)
    assert_double_precision((eps, vecs))
    np.testing.assert_allclose(eps, np.linalg.eigvalsh(M)[:n_bands], atol=1e-12 * np.linalg.norm(M, 2))
    assert np.all(np.diff(np.asarray(eps)) >= 0)
    V = np.asarray(vecs)
    np.testing.assert_allclose(V.conj().T @ V, np.eye(n_bands), atol=1e-12)
    resid = np.linalg.norm(M @ V - V * np.asarray(eps)[None, :], axis=0)
    assert np.all(resid <= RESIDUAL_RTOL * np.linalg.norm(M, 2) * max(1, np.sqrt(n)))


@settings(max_examples=10)
@given(A=cells(), k=kfracs(), target=TARGET_ECUTS, v0=st.floats(-1.0, 1.0))
def test_constant_potential_shifts_every_eigenvalue(A, k, target, v0):
    """A term defined outside the package (the 1b extension point) shifts by V0."""
    lat, basis = padded_basis(A, k, target)
    n_bands = int(basis.n_pw[0])
    e = np.asarray(band_structure(lat, basis, n_bands=n_bands).eigenvalues)
    e_shift = np.asarray(
        band_structure(lat, basis, n_bands=n_bands, terms=(Kinetic(), ConstantShift(v0))).eigenvalues
    )
    # Shifted bands above ecut may interleave with the padding sentinel; compare below it.
    below = e + v0 < basis.ecut
    np.testing.assert_allclose(e_shift[below], e[below] + v0, atol=EIGENVALUE_ATOL)


def test_too_many_bands_raises():
    lat = Lattice.fcc(A_SI)
    ks = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    basis = PlaneWaveBasis.build(lat, ks, ecut_between_shells(fcc_vectors(A_SI), ks, 1.5))
    band_structure(lat, basis, n_bands=int(np.min(basis.n_pw)))
    with pytest.raises(ValueError):
        band_structure(lat, basis, n_bands=int(np.min(basis.n_pw)) + 1)
