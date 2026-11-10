"""Block LOBPCG for the lowest eigenpairs of a Hermitian operator (Knyazev 2001;
theory post 2a, 'The eigensolver').

Each iteration performs a Rayleigh-Ritz step on span[X, W, P]: the current block X, the
(preconditioned) residuals W and the previous search directions P. The fragile part is
the conditioning of that block, so W and P are orthogonalised against X twice, scaled
to unit columns, and orthonormalised through the eigendecomposition of their Gram
matrix, dropping near-dependent directions (they get a large Ritz value and are never
selected). Shapes are fixed, so the solver runs inside jit, vmap and lax.while_loop.

It need not be differentiable: forces stop the gradient on converged orbitals.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

# Directions whose Gram eigenvalue is below this fraction of the largest are dropped.
_DROP = 1e-12


def _orth_against(X, V):
    for _ in range(2):
        V = V - X @ (X.conj().T @ V)
    return V


def _gram_orthonormalise(V):
    """Orthonormal basis of span(V) with fixed shape: (Q, keep). Dropped columns are 0."""
    norms = jnp.linalg.norm(V, axis=0)
    V = V / jnp.where(norms > 0, norms, 1.0)
    G = V.conj().T @ V
    lam, U = jnp.linalg.eigh(G)
    keep = lam > _DROP * jnp.maximum(jnp.max(lam), 1e-300)
    scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)
    return V @ (U * scale), keep


def lobpcg(
    apply: Callable[[jax.Array], jax.Array],
    dim: int,
    n_bands: int,
    *,
    x0=None,
    preconditioner=None,
    tol: float = 1e-10,
    max_iter: int = 300,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Lowest ``n_bands`` eigenpairs of ``apply`` on C^dim: eigenvalues (n_bands,)
    ascending and orthonormal eigenvectors (dim, n_bands).

    ``x0`` (dim, n_bands) warm-starts the block; by default a fixed-seed random block is
    used, so every symmetry component is present (a symmetric start can converge to a
    saddle point). ``preconditioner(r, eps)`` maps residuals (dim, n) to search
    directions. Stops when every residual norm ||A x - eps x|| is below ``tol``
    (absolute, in the units of A) or after ``max_iter`` iterations; the caller checks
    the residuals.
    """
    k = int(n_bands)
    if x0 is None:
        key = jax.random.PRNGKey(seed)
        kr, ki = jax.random.split(key)
        x0 = jax.random.normal(kr, (dim, k)) + 1j * jax.random.normal(ki, (dim, k))
    X, _ = jnp.linalg.qr(jnp.asarray(x0, dtype=jnp.complex128))
    AX = apply(X)
    T = X.conj().T @ AX
    eps, C = jnp.linalg.eigh(0.5 * (T + T.conj().T))
    X, AX = X @ C, AX @ C
    P = jnp.zeros_like(X)

    def residual(X, AX, eps):
        R = AX - X * eps[None, :]
        return R, jnp.linalg.norm(R, axis=0)

    _, rnorm = residual(X, AX, eps)

    def cond(state):
        _, _, _, _, rnorm, it = state
        return (jnp.max(rnorm) >= tol) & (it < max_iter)

    def body(state):
        X, AX, P, eps, _, it = state
        R = AX - X * eps[None, :]
        W = R if preconditioner is None else preconditioner(R, eps)
        Z = _orth_against(X, jnp.concatenate([W, P], axis=1))
        Q, keep = _gram_orthonormalise(Z)
        Q = _orth_against(X, Q)  # second pass: twice is enough
        Q, keep2 = _gram_orthonormalise(Q)
        keep = keep & keep2
        Q = jnp.where(keep[None, :], Q, 0)
        AQ = apply(Q)
        S = jnp.concatenate([X, Q], axis=1)
        AS = jnp.concatenate([AX, AQ], axis=1)
        T = S.conj().T @ AS
        T = 0.5 * (T + T.conj().T)
        scale = jnp.max(jnp.abs(T)) + 1.0
        dropped = jnp.concatenate([jnp.zeros(k, dtype=bool), ~keep])
        T = T + jnp.diag(jnp.where(dropped, 1e3 * scale, 0.0))
        _, C = jnp.linalg.eigh(T)
        C = C[:, :k]
        X_new = S @ C
        AX_new = AS @ C
        P_new = Q @ C[k:]
        # Re-orthonormalise X against rounding drift (keeps AX consistent).
        Xq, Rq = jnp.linalg.qr(X_new)
        Rinv = jnp.linalg.inv(Rq)
        X_new, AX_new = Xq, AX_new @ Rinv
        T2 = X_new.conj().T @ AX_new
        eps_new, C2 = jnp.linalg.eigh(0.5 * (T2 + T2.conj().T))
        X_new, AX_new = X_new @ C2, AX_new @ C2
        _, rnorm = residual(X_new, AX_new, eps_new)
        return X_new, AX_new, P_new, eps_new, rnorm, it + 1

    state = (X, AX, P, eps, rnorm, jnp.asarray(0))
    X, AX, P, eps, rnorm, _ = jax.lax.while_loop(cond, body, state)
    return eps, X
