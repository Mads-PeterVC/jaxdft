"""LocalPotential invariants (theory post 1b, 'Local pseudopotential and structure factor',
'The Hamiltonian matrix', 'Invariants and checks').

CB-specific checks use the package's CB silicon; checks of the general formula (any
positions, any weights, the phase sign) use a smooth test-only Gaussian pseudopotential,
because CB is only defined for perfect diamond at a_CB.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft import (
    Kinetic,
    Lattice,
    LocalPotential,
    PlaneWaveBasis,
    Structure,
    apply_H,
    assert_double_precision,
    band_structure,
    dense_matrix,
    diamond,
    hamiltonian,
)
from jaxdft.pseudo.cohen_bergstresser import silicon
from support import cohen_bergstresser as cbref
from support.free_electron import oh_operations, reciprocal
from support.pseudos import GaussianPseudo, UnboundedPseudo
from support.strategies import cells, kfracs
from support.tolerances import EIGENVALUE_ATOL

A0 = 0.529177210544
SMALL_ECUT = 2.0  # Ha: ~40 plane waves, enough for invariants


@pytest.fixture(scope="module")
def si():
    return diamond(5.43, "Si", unit="Angstrom")


@pytest.fixture(scope="module")
def vloc():
    return LocalPotential({"Si": silicon()})


def miller_from_cubic(hkl, structure):
    """Miller indices m with m B = (2 pi / a)(h, k, l), for the fcc cell of ``structure``."""
    a = float(np.linalg.norm(structure.lattice.vectors[0])) * np.sqrt(2)
    G = 2 * np.pi / a * np.asarray(hkl, dtype=float)
    return np.rint(G @ np.linalg.inv(np.asarray(structure.lattice.reciprocal))).astype(int)


def spectrum(structure, k, ecut, terms, n_pad=None):
    basis = PlaneWaveBasis.build(structure.lattice, np.atleast_2d(k), ecut, n_pad=n_pad)
    return np.asarray(band_structure(structure, basis, n_bands=int(np.min(basis.n_pw)), terms=terms).eigenvalues)


# --- the form factor and structure factor -------------------------------------------


def test_coefficients_match_independent_numpy_model(si, vloc):
    m = np.array(list(np.ndindex(9, 9, 9))) - 4
    got = np.asarray(vloc.coefficients(si, jnp.asarray(m)))
    assert got.dtype == np.complex128
    np.testing.assert_allclose(got, cbref.v_tilde(m), atol=1e-14)


def test_structure_factor_differs_within_a_shell(si, vloc):
    """S = cos(pi(h+k+l)/4): -sqrt2/2 for (1,1,1), +sqrt2/2 for (-1,1,1) on |G|^2 = 3."""
    v_s = 0.5 * -0.21
    m = np.array([miller_from_cubic(h, si) for h in [(1, 1, 1), (-1, 1, 1)]])
    got = np.asarray(vloc.coefficients(si, jnp.asarray(m)))
    np.testing.assert_allclose(got, [-np.sqrt(2) / 2 * v_s, np.sqrt(2) / 2 * v_s], atol=1e-15)


def test_mean_potential_is_zero_for_cb(si, vloc):
    assert vloc.coefficients(si, jnp.zeros((1, 3), dtype=int))[0] == 0


@settings(max_examples=20)
@given(A=cells(), f=hnp.arrays(float, (2, 3), elements=st.floats(-1, 1)), w=hnp.arrays(float, (2,), elements=st.floats(0, 1)))
def test_general_coefficients_hermitian_and_linear_in_weights(A, f, w):
    """V(-G) = conj V(G) for any positions and weights; V is linear in the weights."""
    vg = LocalPotential({"X": GaussianPseudo()})
    s = Structure(Lattice(A), f, ["X", "X"], weights=w)
    m = jnp.asarray(np.array(list(np.ndindex(5, 5, 5))) - 2)
    v = np.asarray(vg.coefficients(s, m))
    np.testing.assert_allclose(np.asarray(vg.coefficients(s, -m)), np.conj(v), atol=1e-14)
    s3 = Structure(Lattice(A), f, ["X", "X"], weights=0.3 * w)
    np.testing.assert_allclose(np.asarray(vg.coefficients(s3, m)), 0.3 * v, atol=1e-14)


@settings(max_examples=20)
@given(A=cells())
def test_single_atom_at_origin_gives_v_hat_over_omega(A):
    """The 1/Omega normalisation, independently of CB's tabulated per-atom volume."""
    p = GaussianPseudo()
    s = Structure(Lattice(A), [[0.0, 0.0, 0.0]], ["X"])
    m = np.array(list(np.ndindex(5, 5, 5))) - 2
    q = np.linalg.norm(m @ reciprocal(A), axis=1)
    expected = np.asarray(p.local(q)) / abs(np.linalg.det(A))
    np.testing.assert_allclose(np.asarray(LocalPotential({"X": p}).coefficients(s, jnp.asarray(m))), expected, atol=1e-14)


def test_phase_sign_places_the_atom_at_its_position():
    """Real-space check (eigenvalues cannot see the sign): V(r) synthesised by direct
    summation over the G within support is inversion-symmetric about the atom at tau,
    not about -tau."""
    A = 7.0 * (np.eye(3) + 0.1 * np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
    f = np.array([0.1, 0.2, 0.3])
    p = GaussianPseudo(support=4.0)
    s = Structure(Lattice(A), [f], ["X"])
    n = 8
    m = np.array(list(np.ndindex(2 * n + 1, 2 * n + 1, 2 * n + 1))) - n
    G = m @ reciprocal(A)
    keep = np.linalg.norm(G, axis=1) <= p.local_support
    v = np.asarray(LocalPotential({"X": p}).coefficients(s, jnp.asarray(m[keep])))
    G = G[keep]

    def V(r):
        return np.sum(v * np.exp(1j * G @ r))

    tau = f @ A
    rng = np.random.default_rng(0)
    asym = []
    for _ in range(5):
        d = rng.normal(size=3)
        assert abs(V(tau + d).imag) < 1e-12
        np.testing.assert_allclose(V(tau + d), V(tau - d), atol=1e-12)
        asym.append(abs(V(-tau + d) - V(-tau - d)))
    assert max(asym) > 1e-3 * abs(V(tau))


# --- the operator --------------------------------------------------------------------


@settings(max_examples=10)
@given(k=kfracs())
def test_dense_physical_block_matches_numpy_hamiltonian(si, vloc, k):
    basis = PlaneWaveBasis.build(si.lattice, k[None], SMALL_ECUT)
    b = basis[0]
    H = np.asarray(dense_matrix(hamiltonian(si, b, (Kinetic(), vloc))))
    mask = np.asarray(b.mask)
    m_pkg = np.asarray(b.miller)[mask]
    m_ref, H_ref = cbref.hamiltonian(k, SMALL_ECUT)
    order = {tuple(x): i for i, x in enumerate(m_ref)}
    perm = np.array([order[tuple(x)] for x in m_pkg])
    np.testing.assert_allclose(H[np.ix_(mask, mask)], H_ref[np.ix_(perm, perm)], atol=1e-13)


@settings(max_examples=10)
@given(k=kfracs(), seed=st.integers(0, 2**32 - 1), extra=st.integers(1, 12))
def test_operator_dense_hermitian_and_padding_decoupled(si, vloc, k, seed, extra):
    b0 = PlaneWaveBasis.build(si.lattice, k[None], SMALL_ECUT)
    b = PlaneWaveBasis.build(si.lattice, k[None], SMALL_ECUT, n_pad=b0.n_pad + extra)[0]
    ham = hamiltonian(si, b, (Kinetic(), vloc))
    H = np.asarray(dense_matrix(ham))
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(b.n_pad, 3)) + 1j * rng.normal(size=(b.n_pad, 3))
    np.testing.assert_allclose(np.asarray(apply_H(ham, jnp.asarray(psi))), H @ psi, atol=1e-12)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-14)
    mask = np.asarray(b.mask)
    np.testing.assert_array_equal(H[np.ix_(~mask, mask)], 0)
    np.testing.assert_array_equal(H[np.ix_(mask, ~mask)], 0)


@settings(max_examples=10)
@given(k=kfracs())
def test_trace_of_H_equals_trace_of_T(si, vloc, k):
    b = PlaneWaveBasis.build(si.lattice, k[None], SMALL_ECUT)[0]
    mask = np.asarray(b.mask)
    H = np.asarray(dense_matrix(hamiltonian(si, b, (Kinetic(), vloc))))[np.ix_(mask, mask)]
    T = np.asarray(dense_matrix(hamiltonian(si, b, (Kinetic(),))))[np.ix_(mask, mask)]
    np.testing.assert_allclose(np.trace(H), np.trace(T), rtol=1e-13)


def test_prepared_term_is_matrix_free(si, vloc):
    """The prepared potential stores O(K N_pad) data (K coupling vectors), not O(N_pad^2)."""
    b = PlaneWaveBasis.build(si.lattice, np.zeros((1, 3)), 10.0)[0]
    assert b.n_pad > 400
    prepared = vloc.prepare(si, b)
    size = sum(np.size(x) for x in jax.tree_util.tree_leaves(prepared))
    assert size <= 256 * b.n_pad < b.n_pad**2


# --- spectrum invariants ---------------------------------------------------------------


@settings(max_examples=10)
@given(k=kfracs(-0.5, 0.5), t=hnp.arrays(float, (3,), elements=st.floats(-1, 1)))
def test_spectrum_invariant_under_translation_and_exchange(si, vloc, k, t):
    """A rigid shift rephases V(G - G') (a unitary change of basis); the complex path
    must work for this. Exchanging the two atoms is inversion about the bond centre."""
    terms = (Kinetic(), vloc)
    e = spectrum(si, k, SMALL_ECUT, terms)
    shifted = Structure(si.lattice, si.positions + t, si.species)
    swapped = Structure(si.lattice, si.positions[::-1], si.species)
    np.testing.assert_allclose(spectrum(shifted, k, SMALL_ECUT, terms), e, atol=EIGENVALUE_ATOL)
    np.testing.assert_allclose(spectrum(swapped, k, SMALL_ECUT, terms), e, atol=EIGENVALUE_ATOL)


@settings(max_examples=10)
@given(k=kfracs())
def test_zero_weights_recover_the_empty_lattice(si, vloc, k):
    off = Structure(si.lattice, si.positions, si.species, weights=[0.0, 0.0])
    np.testing.assert_allclose(
        spectrum(off, k, SMALL_ECUT, (Kinetic(), vloc)), spectrum(off, k, SMALL_ECUT, (Kinetic(),)), atol=EIGENVALUE_ATOL
    )


@settings(max_examples=5)
@given(k=kfracs(-0.5, 0.5))
def test_spectrum_invariant_under_oh(si, vloc, k):
    B = np.asarray(si.lattice.reciprocal)
    images = np.array([np.linalg.solve(B.T, S @ (k @ B)) for S in oh_operations()])
    basis = PlaneWaveBasis.build(si.lattice, images, SMALL_ECUT)
    n = int(np.min(basis.n_pw))
    e = np.asarray(band_structure(si, basis, n_bands=n, terms=(Kinetic(), vloc)).eigenvalues)
    below = e[0] < SMALL_ECUT
    for row in e[1:]:
        np.testing.assert_allclose(row[below], e[0][below], atol=EIGENVALUE_ATOL)


@settings(max_examples=10)
@given(k=kfracs(), factor=st.floats(1.2, 2.0))
def test_eigenvalues_non_increasing_with_cutoff(si, vloc, k, factor):
    """Courant-Fischer on nested bases, now with a non-diagonal H."""
    terms = (Kinetic(), vloc)
    lo = spectrum(si, k, SMALL_ECUT, terms)[0]
    hi = spectrum(si, k, SMALL_ECUT * factor, terms)[0]
    assert np.all(hi[: len(lo)] <= lo + EIGENVALUE_ATOL)


@settings(max_examples=10)
@given(ks=st.lists(kfracs(), min_size=1, max_size=3), extra=st.integers(1, 20))
def test_bands_independent_of_n_pad(si, vloc, ks, extra):
    ks = np.array(ks)
    terms = (Kinetic(), vloc)
    b1 = PlaneWaveBasis.build(si.lattice, ks, SMALL_ECUT)
    b2 = PlaneWaveBasis.build(si.lattice, ks, SMALL_ECUT, n_pad=b1.n_pad + extra)
    n = int(np.min(b1.n_pw))
    e1 = np.asarray(band_structure(si, b1, n_bands=n, terms=terms).eigenvalues)
    e2 = np.asarray(band_structure(si, b2, n_bands=n, terms=terms).eigenvalues)
    below = e1 < SMALL_ECUT
    np.testing.assert_allclose(e1[below], e2[below], atol=EIGENVALUE_ATOL)


def test_results_are_double_precision(si, vloc):
    basis = PlaneWaveBasis.build(si.lattice, np.zeros((1, 3)), SMALL_ECUT)
    assert_double_precision(band_structure(si, basis, n_bands=4, terms=(Kinetic(), vloc)))


# --- refusals ----------------------------------------------------------------------------


def _gamma(structure, ecut=1.5):
    return PlaneWaveBasis.build(structure.lattice, np.zeros((1, 3)), ecut)


def test_missing_species_raises(si):
    with pytest.raises(ValueError, match="Si"):
        band_structure(si, _gamma(si), n_bands=2, terms=(Kinetic(), LocalPotential({"Ge": silicon()})))


def test_unbounded_pseudopotential_needs_fft(si):
    with pytest.raises(NotImplementedError):
        band_structure(si, _gamma(si), n_bands=2, terms=(Kinetic(), LocalPotential({"Si": UnboundedPseudo()})))


def conventional_si(weights=None):
    a = 5.43
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    pos = np.concatenate([fcc + 0.125, fcc - 0.125]) * a
    return Structure.from_cartesian(a * np.eye(3), pos, ["Si"] * 8, unit="Angstrom", weights=weights)


def test_perfect_supercell_is_accepted(vloc):
    """In the 8-atom conventional cell the extra reciprocal vectors have S = 0, so CB is
    never evaluated off its table."""
    s = conventional_si()
    band_structure(s, _gamma(s), n_bands=2, terms=(Kinetic(), vloc))


@pytest.mark.parametrize(
    "structure",
    [
        pytest.param(lambda: diamond(5.5, "Si", unit="Angstrom"), id="isotropic-scaling"),
        pytest.param(
            lambda: Structure(
                Lattice(np.asarray(diamond(5.43, "Si", unit="Angstrom").lattice.vectors) @ np.diag([1.0, 1.0, 1.01])),
                [[0.125] * 3, [-0.125] * 3],
                ["Si", "Si"],
            ),
            id="anisotropic-strain",
        ),
        pytest.param(lambda: conventional_si(weights=[0.5] + [1.0] * 7), id="supercell-unequal-weights"),
        pytest.param(
            lambda: Structure(diamond(5.43, "Si", unit="Angstrom").lattice, [[0.125] * 3, [-0.125] * 3], ["Si", "Si"], weights=[1.0, 0.5]),
            id="primitive-unequal-weights-uses-shell-4",
        ),
    ],
)
def test_cb_refuses_structures_it_does_not_define(vloc, structure):
    s = structure()
    with pytest.raises(ValueError):
        band_structure(s, _gamma(s), n_bands=2, terms=(Kinetic(), vloc))
