"""Occupations over all k-points (theory post 02b, 'Occupations, the gap and the
frontier over all k-points'; engineering log, 2b API: the definition of
``grid_occupations``).

Synthetic eigenvalues: four "valence" bands below a gap and two "conduction" bands,
on eight k-points of weight 1/8, with the frontier placed by hand.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft.scf.occupations import aufbau_occupations, grid_occupations
from support.budget import examples

N_K = 8
W = jnp.full(N_K, 1.0 / N_K)
TOL = 1e-6


def synthetic(vbm_k=3, cbm_k=5):
    """Valence bands -1.0 .. -0.4 with a unique VBM at k = vbm_k; conduction bands with
    a unique CBM at k = cbm_k. Ascending per k."""
    rng = np.random.default_rng(7)
    e = np.empty((N_K, 6))
    for k in range(N_K):
        e[k, :4] = np.sort(rng.uniform(-1.0, -0.45, 4))
        e[k, 4:] = np.sort(rng.uniform(0.2, 0.6, 2))
    e[vbm_k, 3] = -0.40
    e[cbm_k, 4] = 0.10
    return jnp.asarray(e)


def total(f, w=W):
    return float(jnp.sum(w[:, None] * f))


# --- integer electron count ------------------------------------------------------------


@pytest.mark.parametrize("side", [-1, +1])
def test_gapped_integer_count_fills_bands(side):
    f = grid_occupations(synthetic(), W, 8.0, side=side, tol=TOL)
    expected = np.zeros((N_K, 6))
    expected[:, :4] = 2.0
    np.testing.assert_allclose(f, expected, atol=1e-15)
    assert f.dtype == jnp.float64


def test_band_filling_even_when_bands_overlap_across_k():
    """At even N the reference is band filling, not crystal aufbau: a semimetal keeps
    f = 2 in band 3 at every k even where band 4 elsewhere is lower. The gap guard,
    not the occupations, rejects such a state."""
    e = np.asarray(synthetic()).copy()
    e[5, 4] = -0.5  # CBM below the VBM: negative gap over the grid
    e[5] = np.sort(e[5])
    f = grid_occupations(jnp.asarray(e), W, 8.0, tol=TOL)
    np.testing.assert_allclose(f[:, :4], 2.0, atol=1e-15)
    np.testing.assert_allclose(f[:, 4:], 0.0, atol=1e-15)


# --- fractional counts --------------------------------------------------------------


@pytest.mark.parametrize("delta", [0.01, 0.2])
def test_extra_charge_goes_to_the_crystal_cbm(delta):
    e = synthetic(cbm_k=5)
    f = np.array(grid_occupations(e, W, 8.0 + delta, tol=TOL))
    np.testing.assert_allclose(f[5, 4], delta * N_K, atol=1e-14)
    f[5, 4] = 0.0
    np.testing.assert_allclose(f[:, :4], 2.0, atol=1e-15)
    np.testing.assert_allclose(f[:, 4:], 0.0, atol=1e-15)


@pytest.mark.parametrize("delta", [0.01, 0.2])
def test_missing_charge_comes_from_the_crystal_vbm(delta):
    e = synthetic(vbm_k=3)
    f = np.asarray(grid_occupations(e, W, 8.0 - delta, tol=TOL))
    np.testing.assert_allclose(f[3, 3], 2.0 - delta * N_K, atol=1e-14)
    assert total(jnp.asarray(f)) == pytest.approx(8.0 - delta, abs=1e-13)


def test_degenerate_frontier_is_shared_equally_across_k_and_bands():
    """A frontier level spread over two k-points and, at one of them, two bands
    (within tol): each of its three states gets the same change of f."""
    e = np.asarray(synthetic()).copy()
    e[3, 3] = -0.40
    e[6, 3] = -0.40 + 0.3 * TOL
    e[6, 2] = -0.40 - 0.3 * TOL
    delta = 0.03
    f = np.asarray(grid_occupations(jnp.asarray(e), W, 8.0 - delta, tol=TOL))
    change = 2.0 - np.array([f[3, 3], f[6, 3], f[6, 2]])
    np.testing.assert_allclose(change, change[0], atol=1e-14)
    np.testing.assert_allclose(np.sum(change) / N_K, delta, atol=1e-14)
    assert np.sum(f[:, :4] < 2.0 - 1e-14) == 3


def test_states_just_outside_the_tolerance_are_not_shared():
    e = np.asarray(synthetic()).copy()
    e[6, 3] = -0.40 - 10 * TOL
    f = np.asarray(grid_occupations(jnp.asarray(e), W, 8.0 - 0.01, tol=TOL))
    assert f[6, 3] == 2.0 and f[3, 3] < 2.0


def test_charge_spills_into_the_next_level():
    """The CBM state holds 2 w = 0.25 electrons; the rest goes to the next level."""
    e = np.asarray(synthetic(cbm_k=5)).copy()
    e[2, 4] = 0.15  # second-lowest conduction state
    f = np.asarray(grid_occupations(jnp.asarray(e), W, 8.3, tol=TOL))
    np.testing.assert_allclose(f[5, 4], 2.0, atol=1e-14)
    np.testing.assert_allclose(f[2, 4], (0.3 - 0.25) * N_K, atol=1e-13)


def test_non_uniform_weights():
    e = synthetic(cbm_k=5)
    w = jnp.asarray([0.05, 0.1, 0.15, 0.1, 0.2, 0.25, 0.1, 0.05])
    f = grid_occupations(e, w, 8.0 + 0.02, tol=TOL)
    np.testing.assert_allclose(float(f[5, 4]), 0.02 / 0.25, atol=1e-14)
    assert total(f, w) == pytest.approx(8.02, abs=1e-13)


@settings(max_examples=examples(20))
@given(st.floats(0.0, 2.0 * 6 - 1e-6))
def test_conserves_the_electron_count(n_el):
    assert total(grid_occupations(synthetic(), W, n_el, tol=TOL)) == pytest.approx(n_el, abs=1e-12)


@settings(max_examples=examples(20))
@given(st.floats(0.0, 7.999), st.sampled_from([-1, 1]))
def test_single_k_point_equals_aufbau(n_el, side):
    """One k-point of weight 1: identical to the 2a occupations, so every 2a result is
    unchanged."""
    e = jnp.asarray([[-1.0, -0.5, 0.1, 0.7]])
    f = grid_occupations(e, jnp.ones(1), n_el, side=side, tol=TOL)
    np.testing.assert_allclose(f[0], aufbau_occupations(n_el, 4, side=side), atol=1e-13)


# --- the one-sided derivative ---------------------------------------------------------


def _dfdN(e, w, n_el, side):
    return np.asarray(jax.jacfwd(lambda n: grid_occupations(e, w, n, side=side, tol=TOL))(jnp.asarray(n_el)))


def test_derivative_at_integer_count_is_one_sided():
    e = synthetic(vbm_k=3, cbm_k=5)
    left = _dfdN(e, W, 8.0, -1)
    right = _dfdN(e, W, 8.0, +1)
    expected_left = np.zeros((N_K, 6))
    expected_left[3, 3] = N_K
    expected_right = np.zeros((N_K, 6))
    expected_right[5, 4] = N_K
    np.testing.assert_allclose(left, expected_left, atol=1e-12)
    np.testing.assert_allclose(right, expected_right, atol=1e-12)


def test_derivative_of_a_degenerate_frontier_is_shared():
    e = np.asarray(synthetic()).copy()
    e[3, 3] = -0.40
    e[6, 3] = -0.40
    d = _dfdN(jnp.asarray(e), W, 8.0, -1)
    np.testing.assert_allclose(d[3, 3], d[6, 3], atol=1e-12)
    np.testing.assert_allclose(np.sum(W[:, None] * d), 1.0, atol=1e-12)


@pytest.mark.parametrize("n_el", [8.0 - 0.07, 8.0 + 0.07])
def test_sides_agree_away_from_kinks(n_el):
    e = synthetic()
    np.testing.assert_allclose(_dfdN(e, W, n_el, -1), _dfdN(e, W, n_el, +1), atol=1e-12)


@pytest.mark.parametrize("side", [-1, +1])
def test_derivative_moves_exactly_one_electron(side):
    d = _dfdN(synthetic(), W, 8.0, side)
    np.testing.assert_allclose(np.sum(np.asarray(W)[:, None] * d), 1.0, atol=1e-12)


def test_janak_slope_is_the_frontier_eigenvalue():
    """d/dN sum_k w_k sum_n f_nk eps_nk = eps_VBM (left) or eps_CBM (right)."""
    e = synthetic(vbm_k=3, cbm_k=5)
    for side, level in ((-1, -0.40), (+1, 0.10)):
        g = jax.grad(lambda n: jnp.sum(W[:, None] * grid_occupations(e, W, n, side=side, tol=TOL) * e))(8.0)
        np.testing.assert_allclose(float(g), level, atol=1e-12)


# --- errors -------------------------------------------------------------------------


def test_too_many_electrons_raise():
    with pytest.raises(ValueError):
        grid_occupations(synthetic(), W, 12.5, tol=TOL)


def test_bad_side_raises():
    with pytest.raises(ValueError):
        grid_occupations(synthetic(), W, 8.0, side=0, tol=TOL)
