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
