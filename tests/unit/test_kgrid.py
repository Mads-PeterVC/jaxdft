"""k-grids, k-point weights on the basis, symmetry-compatible FFT grids and supercells
(theory post 02b, 'Brillouin-zone sampling', 'Grid symmetry'; engineering log, 2b API).
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import PlaneWaveBasis, Structure, diamond, fcc_lattice, monkhorst_pack, supercell
from jaxdft.basis.planewave import min_fft_shape, rule_fft_shape
from jaxdft.scf.density import electron_density
from support.silicon import A_SI, cubic_operations, maps_grid, space_group

LATTICE = fcc_lattice(A_SI, unit="Bohr")


def _is_smooth(n):
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


# --- monkhorst_pack -------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 3, 4)])
def test_unshifted_grid(shape):
    k, w = monkhorst_pack(shape)
    n = int(np.prod(shape))
    assert k.shape == (n, 3) and w.shape == (n,)
    assert k.dtype == jnp.float64 and w.dtype == jnp.float64
    np.testing.assert_allclose(w, 1.0 / n, atol=1e-16)
    np.testing.assert_allclose(float(jnp.sum(w)), 1.0, atol=1e-15)
    k = np.asarray(k)
    assert np.all(k >= 0) and np.all(k < 1)
    assert len({tuple(np.round(x * np.asarray(shape)).astype(int)) for x in k}) == n
    np.testing.assert_allclose(k * np.asarray(shape), np.round(k * np.asarray(shape)), atol=1e-12)
    np.testing.assert_allclose(k[0], 0.0, atol=0)  # Gamma is on the grid


def test_shifted_grid_points():
    """k = (n + s) / N: the half-shifted 2^3 grid is {1/4, 3/4}^3."""
    k, _ = monkhorst_pack((2, 2, 2), shift=(0.5, 0.5, 0.5))
    expected = set(itertools.product((0.25, 0.75), repeat=3))
    assert {tuple(float(v) for v in x) for x in np.asarray(k)} == expected


def _closed(points, transform):
    pts = np.mod(np.asarray(points), 1.0)
    keys = {tuple(np.round(p, 9) % 1.0) for p in pts}
    return all(tuple(np.round(np.mod(transform(p), 1.0), 9) % 1.0) in keys for p in pts)


def _reciprocal_ops():
    """The cubic group acting on fractional k (rows): k -> k R_k."""
    A = np.asarray(LATTICE.vectors)
    B = 2 * np.pi * np.linalg.inv(A).T
    return [np.rint(B @ C.T @ np.linalg.inv(B)) for C in cubic_operations()]


@pytest.mark.parametrize("n", [2, 3, 4])
def test_unshifted_grid_is_closed_under_inversion_and_the_point_group(n):
    """The unshifted grid is the reciprocal lattice of the supercell lattice, so the
    cubic group maps it onto itself (theory post, 'The grid is unshifted')."""
    k, _ = monkhorst_pack((n, n, n))
    assert _closed(k, lambda p: -p)
    for R in _reciprocal_ops():
        assert _closed(k, lambda p, R=R: p @ R)


def test_half_shifted_fcc_grid_is_not_closed_under_the_cubic_group():
    """Positive control of the argument: (1/4,1/4,1/4)-type points rotate off the grid."""
    k, _ = monkhorst_pack((2, 2, 2), shift=(0.5, 0.5, 0.5))
    assert not all(_closed(k, lambda p, R=R: p @ R) for R in _reciprocal_ops())


@pytest.mark.parametrize("n", [2, 4])
def test_half_shifted_grid_holds_a_quarter_of_every_star(n):
    """The review's decomposition: the shifted grid contains the same fraction (1/4) of
    every cubic star it touches, which is why invariant scalars stay symmetric."""
    k, _ = monkhorst_pack((n, n, n), shift=(0.5, 0.5, 0.5))
    pts = {tuple(np.round(np.mod(p, 1.0), 9) % 1.0) for p in np.asarray(k)}
    ops = _reciprocal_ops()
    seen = set()
    for p in sorted(pts):
        if p in seen:
            continue
        star = {tuple(np.round(np.mod(np.asarray(p) @ R, 1.0), 9) % 1.0) for R in ops}
        inside = star & pts
        seen |= inside
        assert 4 * len(inside) == len(star), (p, len(inside), len(star))


@pytest.mark.parametrize("bad", [(0, 2, 2), (2, 2), (-1, 2, 2)])
def test_bad_grid_shape_raises(bad):
    with pytest.raises(ValueError):
        monkhorst_pack(bad)


# --- weights on the basis ------------------------------------------------------------


def test_default_weights_are_uniform():
    k, _ = monkhorst_pack((2, 2, 2))
    b = PlaneWaveBasis.build(LATTICE, k, 3.0)
    assert b.kweights.shape == (8,) and b.kweights.dtype == jnp.float64
    np.testing.assert_allclose(b.kweights, 1 / 8, atol=1e-16)


def test_weights_are_a_leaf_that_follows_the_k_axis():
    k, _ = monkhorst_pack((2, 2, 2))
    w = jnp.arange(1.0, 9.0) / 36.0
    b = PlaneWaveBasis.build(LATTICE, k, 3.0, kweights=w)
    np.testing.assert_allclose(b.kweights, w, atol=0)
    np.testing.assert_allclose(b[3].kweights, w[3], atol=0)
    assert any(leaf is b.kweights for leaf in jax.tree_util.tree_leaves(b))


@pytest.mark.parametrize(
    "w",
    [np.full(8, 0.1), np.full(7, 1 / 7), np.r_[np.full(7, 1 / 6), -1 / 6]],
    ids=["sum-not-one", "wrong-length", "negative"],
)
def test_invalid_weights_raise(w):
    k, _ = monkhorst_pack((2, 2, 2))
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(LATTICE, k, 3.0, kweights=w)


def test_density_integrates_to_n_for_any_weights():
    """int rho = sum_k w_k sum_n f_nk for orthonormal orbitals, whatever the weights
    (theory post: 'int rho = N_el for any weights summing to 1')."""
    k, _ = monkhorst_pack((2, 1, 1))
    w = jnp.asarray([0.3, 0.7])
    b = PlaneWaveBasis.build(LATTICE, k, 4.0, kweights=w)
    rng = np.random.default_rng(1)
    c = np.zeros((2, b.n_pad, 3), dtype=complex)
    for i in range(2):
        n = int(b.n_pw[i])
        m = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
        c[i, :n] = np.linalg.qr(m)[0]
    f = jnp.asarray([[2.0, 2.0, 0.5], [2.0, 1.0, 0.0]])
    rho = electron_density(LATTICE, b, jnp.asarray(c), f)
    n_el = float(LATTICE.volume) / rho.size * float(rho.sum())
    np.testing.assert_allclose(n_el, 0.3 * 4.5 + 0.7 * 3.0, rtol=1e-12)


# --- symmetry-compatible FFT grids ------------------------------------------------------


@pytest.mark.parametrize("ecut", [5.0, 8.0, 10.0, 12.0, 15.0, 20.0])
@pytest.mark.parametrize("multiple", [1, 2, 3])
def test_fft_multiple(ecut, multiple):
    """Each N_i is the smallest 2-3-5-smooth multiple of fft_multiple at or above the
    no-wraparound bound."""
    k, _ = monkhorst_pack((1, 1, 1))
    shape = PlaneWaveBasis.build(LATTICE, k, ecut, fft_multiple=multiple).fft_shape
    bound = min_fft_shape(LATTICE, ecut)
    for n, b in zip(shape, bound):
        assert n >= b and n % multiple == 0 and _is_smooth(n)
        assert not any(m % multiple == 0 and _is_smooth(m) for m in range(b, n))
    if multiple == 1:
        assert shape == rule_fft_shape(LATTICE, ecut)


def test_fft_multiple_for_diamond_at_15_ha():
    """The rule gives the odd N = 25 at 15 Ha; the diamond-compatible grid is 30
    (theory post, decision 3: 25 -> 30 with radices 2, 3, 5)."""
    k, _ = monkhorst_pack((1, 1, 1))
    assert PlaneWaveBasis.build(LATTICE, k, 15.0).fft_shape == (25, 25, 25)
    assert PlaneWaveBasis.build(LATTICE, k, 15.0, fft_multiple=2).fft_shape == (30, 30, 30)
    assert PlaneWaveBasis.build(LATTICE, k, 15.0, fft_multiple=(2, 1, 1)).fft_shape == (30, 25, 25)


@pytest.mark.parametrize("bad", [0, -2, (2, 2)])
def test_fft_multiple_invalid_raises(bad):
    k, _ = monkhorst_pack((1, 1, 1))
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(LATTICE, k, 12.0, fft_multiple=bad)


def test_fft_multiple_with_explicit_shape_raises():
    k, _ = monkhorst_pack((1, 1, 1))
    with pytest.raises(ValueError):
        PlaneWaveBasis.build(LATTICE, k, 12.0, fft_shape=(24, 24, 24), fft_multiple=2)


# --- the diamond space group on the FFT grid (the review's finding 1) ------------------


def test_diamond_space_group_translations_are_half_lattice_vectors():
    """48 operations; fractional translations (0,0,0), (1/2,0,0), (0,1/2,0), (0,0,1/2)
    in the primitive basis, 12 operations each."""
    ops = space_group(diamond(A_SI, "Si", unit="Bohr"))
    assert len(ops) == 48
    counts = {}
    for _R, t in ops:
        key = tuple(np.round(t, 9))
        counts[key] = counts.get(key, 0) + 1
    assert counts == {(0, 0, 0): 12, (0.5, 0, 0): 12, (0, 0.5, 0): 12, (0, 0, 0.5): 12}


@pytest.mark.parametrize(("n", "expected"), [(24, 48), (25, 12), (26, 48), (27, 12), (28, 48), (30, 48)])
def test_operations_mapping_the_grid_onto_itself(n, expected):
    """N even, not 4 | N: 26 and 30 keep all 48 operations."""
    ops = space_group(diamond(A_SI, "Si", unit="Bohr"))
    assert sum(maps_grid(op, (n, n, n)) for op in ops) == expected


# --- supercells ---------------------------------------------------------------------


def test_supercell_builder():
    s = Structure(LATTICE, [[0.1, 0.2, 0.3], [0.6, 0.5, 0.4]], ["Si", "C"], [1.0, 0.5])
    sc = supercell(s, (2, 1, 3))
    A = np.asarray(LATTICE.vectors)
    np.testing.assert_allclose(sc.lattice.vectors, A * np.array([[2], [1], [3]]), atol=1e-14)
    assert sc.n_atoms == 12
    assert sc.species == ("Si", "C") * 6
    np.testing.assert_allclose(sc.weights, np.tile([1.0, 0.5], 6), atol=0)
    # translations outer, atoms inner; Cartesian positions are images of the originals
    r = np.asarray(sc.cartesian_positions)
    r0 = np.asarray(s.cartesian_positions)
    for idx, t in enumerate(itertools.product(range(2), range(1), range(3))):
        np.testing.assert_allclose(r[2 * idx : 2 * idx + 2], r0 + np.asarray(t) @ A, atol=1e-12)
    np.testing.assert_allclose(float(sc.lattice.volume), 6 * float(LATTICE.volume), rtol=1e-14)


def test_supercell_rejects_bad_repeats():
    s = diamond(A_SI, "Si", unit="Bohr")
    for bad in ((0, 1, 1), (2, 1)):
        with pytest.raises(ValueError):
            supercell(s, bad)
