"""Plane-wave basis invariants (theory post: 'Kinetic-energy cutoff', 'Invariants', Basis).

Cutoffs are placed mid-gap between shells (ecut_between_shells), so the reference
enumeration stays independent of the package's arithmetic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft import Lattice, PlaneWaveBasis, fcc_lattice
from support.free_electron import (
    A_SI,
    basis_set,
    ecut_between_shells,
    fcc_vectors,
    oh_operations,
    reciprocal,
)
from support.strategies import cells, kfracs

TARGET_ECUTS = st.floats(0.8, 3.0)  # Ha; tens to a few hundred plane waves


def miller_sets(basis):
    """Per-k sets of real (unmasked) Miller indices."""
    m, mask = np.asarray(basis.miller), np.asarray(basis.mask)
    return [{tuple(int(x) for x in row) for row in m[i][mask[i]]} for i in range(m.shape[0])]


@settings(max_examples=30)
@given(cells(), st.lists(kfracs(), min_size=1, max_size=3), TARGET_ECUTS)
def test_basis_is_exactly_the_cutoff_sphere(A, ks, target):
    """Every G in the basis satisfies the cutoff, and every G that does is included,
    for skewed cells and k-points also outside the first zone."""
    ks = np.array(ks)
    ecut = ecut_between_shells(A, ks, target)
    basis = PlaneWaveBasis.build(Lattice(A), ks, ecut)
    for k, got in zip(ks, miller_sets(basis)):
        assert got == basis_set(A, k, ecut)


@settings(max_examples=20)
@given(cells(), st.lists(kfracs(), min_size=1, max_size=3), TARGET_ECUTS)
def test_shapes_dtypes_and_padding_length(A, ks, target):
    ks = np.array(ks)
    ecut = ecut_between_shells(A, ks, target)
    basis = PlaneWaveBasis.build(Lattice(A), ks, ecut)
    n_k = len(ks)
    n_pw = np.asarray(basis.n_pw)
    assert basis.miller.shape == (n_k, basis.n_pad, 3)
    assert basis.mask.shape == (n_k, basis.n_pad)
    assert jnp.issubdtype(basis.miller.dtype, jnp.integer)
    assert basis.mask.dtype == jnp.bool_
    assert basis.kpoints.dtype == jnp.float64
    np.testing.assert_array_equal(basis.kpoints, ks)
    assert basis.ecut == ecut
    np.testing.assert_array_equal(n_pw, np.asarray(basis.mask).sum(axis=1))
    assert basis.n_pad == n_pw.max()  # default: exactly max_k N_PW


@settings(max_examples=20)
@given(cells(), kfracs(), TARGET_ECUTS, st.floats(1.1, 2.0))
def test_nesting(A, k, target, factor):
    e1 = ecut_between_shells(A, [k], target)
    e2 = ecut_between_shells(A, [k], target * factor)
    lat = Lattice(A)
    (s1,) = miller_sets(PlaneWaveBasis.build(lat, k[None], e1))
    (s2,) = miller_sets(PlaneWaveBasis.build(lat, k[None], e2))
    assert s1 <= s2


@settings(max_examples=20)
@given(cells(), kfracs(), st.tuples(*[st.integers(-2, 2)] * 3), TARGET_ECUTS)
def test_shift_by_reciprocal_lattice_vector_relabels_basis(A, k, m0, target):
    """B_{k+G0} = B_k - G0, so in particular N_PW(k + G0) = N_PW(k)."""
    m0 = np.array(m0)
    ecut = ecut_between_shells(A, [k, k + m0], target)
    basis = PlaneWaveBasis.build(Lattice(A), np.array([k, k + m0]), ecut)
    s, s_shift = miller_sets(basis)
    assert s_shift == {tuple(int(x) for x in np.array(m) - m0) for m in s}


@settings(max_examples=15)
@given(kfracs(-0.5, 0.5), TARGET_ECUTS)
def test_basis_size_invariant_under_oh_in_fcc(k, target):
    A = fcc_vectors(A_SI)
    B = reciprocal(A)
    images = np.array([np.linalg.solve(B.T, S @ (k @ B)) for S in oh_operations()])
    ecut = ecut_between_shells(A, images, target)
    basis = PlaneWaveBasis.build(Lattice(A), images, ecut)
    n_pw = np.asarray(basis.n_pw)
    assert np.all(n_pw == n_pw[0])


@settings(max_examples=20)
@given(kfracs(-0.5, 0.5), st.floats(1.0, 40.0))
def test_basis_size_within_rigorous_bound(k, target):
    """(1 - r/Gmax)^3 <= N_PW/N_asym <= (1 + r/Gmax)^3, r = BZ circumradius = |W| (fcc)."""
    A = fcc_vectors(A_SI)
    ecut = ecut_between_shells(A, [k], target)
    lat = Lattice(A)
    n_pw = int(PlaneWaveBasis.build(lat, k[None], ecut).n_pw[0])
    gmax = np.sqrt(2 * ecut)
    r_bz = np.sqrt(5) / 2 * (2 * np.pi / A_SI)
    n_asym = float(lat.volume) * (2 * ecut) ** 1.5 / (6 * np.pi**2)
    assert (1 - r_bz / gmax) ** 3 <= n_pw / n_asym <= (1 + r_bz / gmax) ** 3


def test_n_pad_override_and_validation():
    lat = fcc_lattice(A_SI, unit="Bohr")
    ks = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])
    ecut = ecut_between_shells(np.asarray(lat.vectors), ks, 2.0)
    basis = PlaneWaveBasis.build(lat, ks, ecut)
    bigger = PlaneWaveBasis.build(lat, ks, ecut, n_pad=basis.n_pad + 7)
    assert bigger.n_pad == basis.n_pad + 7
    assert miller_sets(bigger) == miller_sets(basis)
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(lat, ks, ecut, n_pad=basis.n_pad - 1)


def test_single_kpoint_view_and_vmap():
    """basis[i] is a PlaneWaveBasis without the k axis; vmap hands the mapped
    function the same thing."""
    lat = fcc_lattice(A_SI, unit="Bohr")
    ks = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.5, 0.5, 0.5]])
    basis = PlaneWaveBasis.build(lat, ks, ecut_between_shells(np.asarray(lat.vectors), ks, 2.0))
    b1 = basis[1]
    assert isinstance(b1, PlaneWaveBasis)
    assert b1.miller.shape == (basis.n_pad, 3)
    assert b1.mask.shape == (basis.n_pad,)
    np.testing.assert_array_equal(b1.kpoints, ks[1])
    assert b1.n_pad == basis.n_pad and b1.ecut == basis.ecut
    assert int(b1.n_pw) == int(basis.n_pw[1])

    def per_k(b):
        assert isinstance(b, PlaneWaveBasis)
        assert b.miller.shape == (basis.n_pad, 3)
        return b.mask.sum()

    np.testing.assert_array_equal(jax.vmap(per_k)(basis), basis.n_pw)
