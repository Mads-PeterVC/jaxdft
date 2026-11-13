"""Block LOBPCG for the lowest eigenpairs of a Hermitian operator (Knyazev 2001;
theory post 2a, 'The eigensolver'; theory post 2c).

Each iteration performs a Rayleigh-Ritz step on span[X, W, P]: the current block X, the
(preconditioned) residuals W and the previous search directions P. The fragile part is
the conditioning of that block, so W and P are orthogonalised against X twice, scaled
to unit columns, and orthonormalised through the eigendecomposition of their Gram
matrix, dropping near-dependent directions. Shapes are fixed, so a dropped direction
is a zero column; it is kept out of the new block by a penalty on its diagonal entry
of the Rayleigh-Ritz matrix (``dropped_penalty``). Shapes are fixed throughout, so the
solver runs inside jit, vmap and lax.while_loop.

Robust stopping (theory post 2c): the stopping test covers the leading ``n_converge``
bands only (buffer bands are carried, not converged); the run also ends when the best
residual has stopped improving (stagnation), and the best block seen is returned,
because past its attainable residual LOBPCG can climb back by orders of magnitude.

It need not be differentiable: forces stop the gradient on converged orbitals.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

from jaxdft.solvers.eigen import EigenInfo

# Directions whose Gram eigenvalue is below this fraction of the largest are dropped.
_DROP = 1e-12
# The stagnation stop is active only once the best residual is within this factor of
# eps_mach ||H||, where the attainable residual lies (theory post 2c: about 10 eps ||H||).
# Far from it, LOBPCG can wait 7 (preconditioned) to 23 (unpreconditioned) iterations
# for a 20 % gain while converging normally (engineering log, 2c implementation).
_STAGNATION_GATE = 1e4
_EPS = float(jnp.finfo(jnp.float64).eps)


def dropped_penalty(eps: jax.Array) -> jax.Array:
    """Rayleigh-Ritz diagonal entry (Ha) of a dropped direction, given the current
    ascending Ritz values ``eps`` of the block. The block is part of the Rayleigh-Ritz
    space, so by Cauchy interlacing its k lowest eigenvalues are at most eps_k; any
    larger value keeps dropped (zero) directions out of the new block. The margin
    (eps_k - eps_1) + 1 Ha covers rounding and coinciding Ritz values, and keeps the
    matrix within the spectrum it already spans (theory post 2c, 'The Rayleigh-Ritz
    penalty')."""
    return eps[-1] + (eps[-1] - eps[0]) + 1.0


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
    n_converge: int | None = None,
    max_iter: int = 300,
    stagnation_window: int = 5,
    stagnation_factor: float = 0.8,
    seed: int = 0,
    return_info: bool = False,
):
    """Lowest ``n_bands`` eigenpairs of ``apply`` on C^dim: eigenvalues (n_bands,)
    ascending and orthonormal eigenvectors (dim, n_bands).

    ``x0`` (dim, n_bands) warm-starts the block; by default a fixed-seed random block is
    used, so every symmetry component is present (a symmetric start can converge to a
    saddle point). ``preconditioner(r, eps)`` maps residuals (dim, n) to search
    directions.

    Stopping: the run ends when the residual norms ||A x - eps x|| of the first
    ``n_converge`` bands (default: all) are below ``tol`` (absolute, in the units of
    A); or when the best such maximum has not fallen below ``stagnation_factor`` times
    its value ``stagnation_window`` iterations earlier, once that residual is within
    a factor 1e4 of eps_mach ||H|| (||H|| estimated by the largest Rayleigh-Ritz
    matrix element seen); or after ``max_iter`` iterations. The block with the smallest maximum residual seen is returned; the
    caller checks the residuals. With ``return_info`` an ``EigenInfo`` is also
    returned.
    """
    k = int(n_bands)
    m = k if n_converge is None else int(n_converge)
    if not 1 <= m <= k:
        raise ValueError(f"n_converge={n_converge} must be between 1 and n_bands={k}")
    window = int(stagnation_window)
    if window < 1:
        raise ValueError(f"stagnation_window must be >= 1, got {stagnation_window}")
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

    def worst(X, AX, eps):
        R = AX - X * eps[None, :]
        return jnp.max(jnp.linalg.norm(R[:, :m], axis=0))

    r0 = worst(X, AX, eps)
    # recent[j % window] holds the best residual after j iterations.
    recent = jnp.full(window, jnp.inf).at[0].set(r0)
    best = (X, eps, r0)
    h_norm = jnp.max(jnp.abs(T))

    def cond(state):
        *_, best, _, it, stagnated, _, _ = state
        return (best[2] >= tol) & (it < max_iter) & ~stagnated

    def body(state):
        X, AX, P, eps, best, recent, it, _, drops, h_norm = state
        R = AX - X * eps[None, :]
        W = R if preconditioner is None else preconditioner(R, eps)
        Z = _orth_against(X, jnp.concatenate([W, P], axis=1))
        Q, keep = _gram_orthonormalise(Z)
        Q = _orth_against(X, Q)  # second pass: twice is enough
        Q, keep2 = _gram_orthonormalise(Q)
        keep = keep & keep2
        Q = jnp.where(keep[None, :], Q, 0)
        # At the first iteration P = 0, so k directions are dropped by construction.
        drops = drops + jnp.where(it > 0, jnp.sum(~keep), 0)
        AQ = apply(Q)
        S = jnp.concatenate([X, Q], axis=1)
        AS = jnp.concatenate([AX, AQ], axis=1)
        T = S.conj().T @ AS
        T = 0.5 * (T + T.conj().T)
        h_norm = jnp.maximum(h_norm, jnp.max(jnp.abs(T)))
        dropped = jnp.concatenate([jnp.zeros(k, dtype=bool), ~keep])
        T = T + jnp.diag(jnp.where(dropped, dropped_penalty(eps), 0.0))
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
        r = worst(X_new, AX_new, eps_new)
        better = r < best[2]
        best = jax.tree_util.tree_map(lambda new, old: jnp.where(better, new, old), (X_new, eps_new, r), best)
        it = it + 1
        slot = it % window
        near_floor = best[2] < _STAGNATION_GATE * _EPS * h_norm
        stagnated = near_floor & (best[2] > stagnation_factor * recent[slot])  # recent[slot]: `window` its ago
        recent = recent.at[slot].set(best[2])
        return X_new, AX_new, P_new, eps_new, best, recent, it, stagnated, drops, h_norm

    state = (X, AX, P, eps, best, recent, jnp.asarray(0), jnp.asarray(False), jnp.asarray(0), h_norm)
    *_, best, _, it, stagnated, drops, _ = jax.lax.while_loop(cond, body, state)
    X, eps, _ = best
    if return_info:
        return eps, X, EigenInfo(it, stagnated, drops)
    return eps, X
