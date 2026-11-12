"""LOBPCG (theory post 02a, 'The eigensolver'; decision 1): lowest eigenpairs of a
Hermitian operator, warm starts, preconditioning, and agreement with dense eigh."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import dense_eigensolver, lobpcg


def kinetic_like(n=400, seed=0, coupling=0.05):
    """Hermitian matrix with a spread diagonal (like 1/2|k+G|^2) and dense weak coupling."""
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0, 20, n))
    X = coupling * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    H = np.diag(d) + (X + X.conj().T) / 2
    return jnp.asarray(H), np.linalg.eigvalsh(H)


@pytest.mark.parametrize("n_bands", [1, 2, 5])
def test_matches_eigh(n_bands):
    H, ref = kinetic_like()
    eps, vecs = lobpcg(lambda x: H @ x, 400, n_bands, tol=1e-10)
    np.testing.assert_allclose(eps, ref[:n_bands], atol=1e-9)
    np.testing.assert_allclose(np.asarray(vecs).conj().T @ np.asarray(vecs), np.eye(n_bands), atol=1e-12)
    res = np.linalg.norm(np.asarray(H @ vecs) - np.asarray(vecs) * np.asarray(eps), axis=0)
    assert np.all(res < 1e-8)


def test_preconditioner_and_warm_start_agree_with_cold_start():
    H, ref = kinetic_like(seed=1)
    diag = jnp.real(jnp.diag(H))
    precond = lambda r, eps: r / (diag[:, None] - eps[None, :] + 1.0)
    cold, _ = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10)
    pre, vecs = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10, preconditioner=precond)
    warm, _ = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10, x0=vecs + 1e-3)
    for e in (cold, pre, warm):
        np.testing.assert_allclose(e, ref[:3], atol=1e-9)


def test_degenerate_lowest_level():
    """A degenerate lowest level must be returned in full (the block spans it)."""
    rng = np.random.default_rng(3)
    Q, _ = np.linalg.qr(rng.normal(size=(200, 200)) + 1j * rng.normal(size=(200, 200)))
    ev = np.concatenate([[-1.0, -1.0, -1.0], np.linspace(0.5, 30, 197)])
    H = jnp.asarray(Q @ np.diag(ev) @ Q.conj().T)
    eps, _ = lobpcg(lambda x: H @ x, 200, 4, tol=1e-10)
    np.testing.assert_allclose(eps, [-1, -1, -1, 0.5], atol=1e-9)


def test_deterministic_default_start():
    H, _ = kinetic_like(seed=2)
    a, _ = lobpcg(lambda x: H @ x, 400, 2, tol=1e-10)
    b, _ = lobpcg(lambda x: H @ x, 400, 2, tol=1e-10)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_same_interface_as_dense_solver():
    H, _ = kinetic_like(n=120, seed=4)
    d_eps, _ = dense_eigensolver(lambda x: H @ x, 120, 3, x0=None, preconditioner=None)
    l_eps, _ = lobpcg(lambda x: H @ x, 120, 3)
    np.testing.assert_allclose(l_eps, d_eps, atol=1e-9)


def test_double_precision():
    H, _ = kinetic_like(n=100)
    eps, vecs = lobpcg(lambda x: H @ x, 100, 2)
    assert eps.dtype == jnp.float64 and vecs.dtype == jnp.complex128


# --- Milestone 2c (theory post 02c; engineering log, '2026-11-12: Milestone 2c public
# API') ----------------------------------------------------------------------------------


def edge_gap_matrix(n=200, m=3, edge_gap=1e-4, seed=5):
    """Hermitian matrix whose lowest m eigenvalues are well separated, followed by a
    nearly degenerate pair: lambda_{m+1}, lambda_{m+1} + edge_gap. A block of m + 1
    vectors must separate that pair to converge its last band, the situation of a
    buffer band next to a small gap (theory post 02c, 'Stopping on the needed bands')."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    low = np.array([-3.0, -2.0, -1.0])[:m]
    ev = np.concatenate([low, [0.5, 0.5 + edge_gap], np.linspace(2.0, 30.0, n - m - 2)])
    return jnp.asarray(Q @ np.diag(ev) @ Q.conj().T), np.sort(ev)


def residuals(H, eps, vecs):
    return np.linalg.norm(np.asarray(H @ vecs) - np.asarray(vecs) * np.asarray(eps), axis=0)


def test_n_converge_stops_on_the_leading_bands():
    """With n_converge = m only the first m residuals must meet tol; their eigenvalues
    agree with eigh to within what that residual implies; and the run is shorter than
    one that must also converge the band next to the small gap."""
    H, ref = edge_gap_matrix()
    tol = 1e-10
    eps, vecs, info = lobpcg(lambda x: H @ x, 200, 4, tol=tol, n_converge=3, return_info=True)
    assert np.all(residuals(H, eps, vecs)[:3] < tol)
    np.testing.assert_allclose(eps[:3], ref[:3], atol=1e-9)
    _, _, info_all = lobpcg(lambda x: H @ x, 200, 4, tol=tol, return_info=True)
    assert int(info.iterations) < int(info_all.iterations)


def test_n_converge_equal_to_the_block_is_the_default():
    H, _ = kinetic_like(seed=6)
    a, va = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10)
    b, vb = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10, n_converge=3)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(va), np.asarray(vb))


def test_info_of_an_ordinary_run():
    H, _ = kinetic_like(seed=7)
    eps, vecs, info = lobpcg(lambda x: H @ x, 400, 3, tol=1e-10, return_info=True)
    assert 0 < int(info.iterations) < 300
    assert not bool(info.stagnated)
    assert np.all(residuals(H, eps, vecs) < 1e-10)


def test_drops_at_the_first_iteration_are_not_late():
    """At the first iteration P = 0, so half of [W, P] is dropped by construction;
    late_drops counts only iterations after the first."""
    H, _ = kinetic_like(seed=8)
    _, _, info = lobpcg(lambda x: H @ x, 400, 3, tol=0.0, max_iter=1, return_info=True)
    assert int(info.iterations) == 1
    assert int(info.late_drops) == 0


def test_unreachable_tolerance_ends_by_stagnation():
    from support.tolerances import STAGNATION_MAX_ITERS

    H, ref = kinetic_like(seed=9)
    diag = jnp.real(jnp.diag(H))
    precond = lambda r, eps: r / (diag[:, None] - eps[None, :] + 1.0)
    eps, vecs, info = lobpcg(lambda x: H @ x, 400, 3, tol=0.0, preconditioner=precond, return_info=True)
    assert bool(info.stagnated)
    assert int(info.iterations) < STAGNATION_MAX_ITERS
    np.testing.assert_allclose(eps, ref[:3], atol=1e-12)
    assert np.all(residuals(H, eps, vecs) < 1e-11)
