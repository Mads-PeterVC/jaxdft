"""The separable nonlocal operator (theory post 02b, 'Matrix elements and the
operator', 'The angular factor at k + G = 0', 'Invariants and checks: Nonlocal term').

The Legendre-formula reference is an independent NumPy transcription
(``support/hgh.py``). A synthetic pseudopotential "X" carries every tabulated
(l, i), so l = 2 and three-projector channels are covered although silicon has
neither.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft import (
    AtomicNonlocal,
    Kinetic,
    Lattice,
    PlaneWaveBasis,
    Structure,
    apply_H,
    dense_matrix,
    fcc_lattice,
    hamiltonian,
)
from jaxdft.pseudo.gth import GTH, HGHChannel, hgh_lda
from support.budget import examples
from support.hgh import legendre_nonlocal_matrix
from support.silicon import A_SI
from support.strategies import cells, kfracs

SI = hgh_lda("Si")
H = hgh_lda("H")
X = GTH(
    3,
    0.4,
    (-3.0,),
    channels=(
        HGHChannel(0, 0.35, ((1.2, -0.3, 0.1), (-0.3, 0.8, 0.05), (0.1, 0.05, 0.4))),
        HGHChannel(1, 0.45, ((0.9, 0.2, -0.1), (0.2, -0.6, 0.15), (-0.1, 0.15, 0.3))),
        HGHChannel(2, 0.55, ((0.7, -0.25), (-0.25, 0.5))),
    ),
)
PSEUDOS = {"Si": {"Si": SI}, "X": {"X": X}}
N_PROJ = {"Si": 2 + 3 * 1, "X": 3 + 3 * 3 + 5 * 2}

# A general geometry with no symmetry, unequal weights (theory post: 'checked ... on a
# geometry with no symmetry, at a general k-point and with lambda = (1, 0.7)').
GENERAL_POSITIONS = [[0.13, 0.11, 0.12], [-0.125, -0.14, -0.1]]
GENERAL_K = [[0.1, 0.2, -0.3]]


def structure(species, positions=GENERAL_POSITIONS, weights=(1.0, 0.7), lattice=None):
    lattice = fcc_lattice(A_SI, unit="Bohr") if lattice is None else lattice
    return Structure(lattice, positions, [species] * len(positions), list(weights))


def operator(s, basis_k, pseudos):
    """Dense V_nl on the physical block, from the operator applied to the identity."""
    term = AtomicNonlocal(pseudos)
    if hasattr(term, "bind"):
        term = term.bind(s)
    H = np.asarray(dense_matrix(hamiltonian(s, basis_k, (term,))))
    n = int(basis_k.mask.sum())
    return H[:n, :n]


def physical_millers(basis_k):
    n = int(basis_k.mask.sum())
    return np.asarray(basis_k.miller)[:n]


@pytest.mark.parametrize("species", ["Si", "X"])
def test_operator_equals_the_legendre_formula(species):
    s = structure(species)
    b = PlaneWaveBasis.build(s.lattice, jnp.asarray(GENERAL_K), 4.0)[0]
    V = operator(s, b, PSEUDOS[species])
    ref = legendre_nonlocal_matrix(s, PSEUDOS[species], np.asarray(GENERAL_K[0]), physical_millers(b))
    scale = np.abs(ref).max()
    np.testing.assert_allclose(V, ref, atol=1e-12 * scale)


@pytest.mark.parametrize("species", ["Si", "X"])
def test_operator_at_gamma_equals_the_legendre_formula(species):
    """The basis contains k + G = 0: the solid-harmonic form needs no special case."""
    s = structure(species)
    b = PlaneWaveBasis.build(s.lattice, jnp.zeros((1, 3)), 4.0)[0]
    V = operator(s, b, PSEUDOS[species])
    assert np.all(np.isfinite(V))
    ref = legendre_nonlocal_matrix(s, PSEUDOS[species], np.zeros(3), physical_millers(b))
    np.testing.assert_allclose(V, ref, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize("species", ["Si", "X"])
def test_rank_is_at_most_the_number_of_projectors(species):
    s = structure(species)
    b = PlaneWaveBasis.build(s.lattice, jnp.asarray(GENERAL_K), 6.0)[0]
    ev = np.linalg.eigvalsh(operator(s, b, PSEUDOS[species]))
    assert len(ev) > 2 * N_PROJ[species]
    assert np.sum(np.abs(ev) > 1e-10 * np.abs(ev).max()) == 2 * N_PROJ[species]


def test_linear_in_the_weights_and_zero_for_a_removed_atom():
    b = PlaneWaveBasis.build(fcc_lattice(A_SI, unit="Bohr"), jnp.asarray(GENERAL_K), 4.0)[0]
    V = {w: operator(structure("Si", weights=w), b, PSEUDOS["Si"]) for w in ((1.0, 0.7), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0))}
    np.testing.assert_allclose(V[(1.0, 0.7)], V[(1.0, 0.0)] + 0.7 * V[(0.0, 1.0)], atol=1e-13)
    np.testing.assert_allclose(V[(0.0, 0.0)], 0.0, atol=0)


def test_zero_for_a_pseudopotential_without_channels():
    s = structure("H")
    b = PlaneWaveBasis.build(s.lattice, jnp.asarray(GENERAL_K), 4.0)[0]
    np.testing.assert_allclose(operator(s, b, {"H": H}), 0.0, atol=0)


@settings(max_examples=examples(5))
@given(t=hnp.arrays(float, (3,), elements=st.floats(-1.0, 1.0)))
def test_translation_covariance(t):
    """Translating every atom by t multiplies V(G, G') by e^{-i(G - G').t}: the
    spectrum is unchanged for any t (theory post: no egg-box contribution)."""
    s = structure("X")
    b = PlaneWaveBasis.build(s.lattice, jnp.asarray(GENERAL_K), 4.0)[0]
    st_ = structure("X", positions=np.asarray(GENERAL_POSITIONS) + t)
    V0, V1 = operator(s, b, PSEUDOS["X"]), operator(st_, b, PSEUDOS["X"])
    m = physical_millers(b)
    D = np.exp(-2j * np.pi * (m @ t))
    np.testing.assert_allclose(V1, D[:, None] * V0 * D.conj()[None, :], atol=1e-13 * np.abs(V0).max())
    np.testing.assert_allclose(np.linalg.eigvalsh(V1), np.linalg.eigvalsh(V0), atol=1e-12 * np.abs(V0).max())


def test_shifting_k_by_a_reciprocal_lattice_vector_leaves_the_spectrum():
    s = structure("X")
    lat = s.lattice
    k = np.asarray(GENERAL_K)
    e = []
    for shift in ((0, 0, 0), (1, 0, -1), (-2, 1, 1)):
        b = PlaneWaveBasis.build(lat, jnp.asarray(k + np.asarray(shift)), 4.0)[0]
        H = np.asarray(dense_matrix(hamiltonian(s, b, (Kinetic(), AtomicNonlocal(PSEUDOS["X"])))))
        n = int(b.mask.sum())
        e.append(np.linalg.eigvalsh(H[:n, :n]))
    for other in e[1:]:
        np.testing.assert_allclose(other, e[0], atol=1e-10)


# --- the operator against its own dense matrix, on random cells ------------------------


@settings(max_examples=examples(10))
@given(
    A=cells(),
    k=kfracs(),
    f=hnp.arrays(float, (2, 3), elements=st.floats(-0.5, 0.5)),
    w=hnp.arrays(float, (2,), elements=st.floats(0.0, 1.0)),
    seed=st.integers(0, 2**32 - 1),
)
def test_apply_agrees_with_dense_and_is_hermitian(A, k, f, w, seed):
    lat = Lattice(A)
    s = Structure(lat, f, ["X", "X"], w)
    b0 = PlaneWaveBasis.build(lat, np.asarray(k)[None], 3.0)
    b = PlaneWaveBasis.build(lat, np.asarray(k)[None], 3.0, n_pad=b0.n_pad + 5)[0]
    ham = hamiltonian(s, b, (Kinetic(), AtomicNonlocal(PSEUDOS["X"])))
    Hd = np.asarray(dense_matrix(ham))
    scale = max(np.abs(Hd).max(), 1.0)
    np.testing.assert_allclose(Hd, Hd.conj().T, atol=1e-13 * scale)
    rng = np.random.default_rng(seed)
    psi = jnp.asarray(rng.normal(size=(b.n_pad, 3)) + 1j * rng.normal(size=(b.n_pad, 3)))
    np.testing.assert_allclose(apply_H(ham, psi), Hd @ np.asarray(psi), atol=1e-12 * scale * b.n_pad)
    # padding stays decoupled
    n = int(b.mask.sum())
    np.testing.assert_allclose(Hd[:n, n:], 0.0, atol=0)


# --- derivatives through k + G = 0 ------------------------------------------------------


def _quadratic_form(s, basis_k, c):
    ham = hamiltonian(s, basis_k, (AtomicNonlocal(PSEUDOS["X"]),))
    return jnp.real(jnp.vdot(c, apply_H(ham, c[:, None])[:, 0]))


def _gamma_setup():
    s = structure("X")
    b = PlaneWaveBasis.build(s.lattice, jnp.zeros((1, 3)), 4.0)[0]
    assert np.any(np.all(np.asarray(b.miller)[np.asarray(b.mask)] == 0, axis=1))
    rng = np.random.default_rng(3)
    c = jnp.where(b.mask, jnp.asarray(rng.normal(size=b.n_pad) + 1j * rng.normal(size=b.n_pad)), 0)
    return s, b, c


def test_derivatives_with_respect_to_positions_and_cell_are_finite_at_gamma():
    """First and second derivatives: no square root of q^2 anywhere (decision 7)."""
    s, b, c = _gamma_setup()

    def of_cell(A):
        return _quadratic_form(Structure(Lattice(A), s.positions, s.species, s.weights), b, c)

    def of_positions(f):
        return _quadratic_form(Structure(s.lattice, f, s.species, s.weights), b, c)

    A = s.lattice.vectors
    for fn, x in ((of_cell, A), (of_positions, s.positions)):
        assert np.all(np.isfinite(np.asarray(jax.grad(fn)(x))))
        assert np.all(np.isfinite(np.asarray(jax.hessian(fn)(x))))


def test_derivatives_with_respect_to_k_are_finite_at_gamma():
    s, b, c = _gamma_setup()
    leaves, treedef = jax.tree_util.tree_flatten(b)
    (index,) = [i for i, leaf in enumerate(leaves) if leaf is b.kpoints]

    def of_k(k):
        new = list(leaves)
        new[index] = k
        return _quadratic_form(s, jax.tree_util.tree_unflatten(treedef, new), c)

    k0 = jnp.zeros(3)
    assert np.all(np.isfinite(np.asarray(jax.grad(of_k)(k0))))
    assert np.all(np.isfinite(np.asarray(jax.hessian(of_k)(k0))))
    # and the derivative is the derivative: central difference of the gradient
    h = 1e-5
    g = np.asarray(jax.grad(of_k)(k0))
    fd = [(float(of_k(k0.at[i].set(h))) - float(of_k(k0.at[i].set(-h)))) / (2 * h) for i in range(3)]
    np.testing.assert_allclose(g, fd, atol=1e-6 * max(1.0, np.abs(g).max()))
