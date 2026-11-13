"""Eigensolvers for Hermitian linear operators, and the diagnostics they report.

Every eigensolver has the signature ``solver(apply, dim, n_bands, *, x0=None,
preconditioner=None, tol=..., n_converge=None, return_info=False)`` (engineering log,
'2026-11-12: Milestone 2c public API'). With ``return_info`` it also returns an
``EigenInfo``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp


class EigenInfo(NamedTuple):
    """Diagnostics of one eigensolver call, as scalar arrays (traceable).

    ``iterations``: iterations taken (0 for a direct solver). ``stagnated``: the run
    ended by the stagnation stop. ``late_drops``: directions dropped as nearly
    dependent after the first iteration, summed over iterations (theory post 02c,
    'The Rayleigh-Ritz penalty')."""

    iterations: jax.Array
    stagnated: jax.Array
    late_drops: jax.Array


def _no_iterations() -> EigenInfo:
    return EigenInfo(jnp.asarray(0), jnp.asarray(False), jnp.asarray(0))


def dense_eigensolver(
    apply: Callable[[jax.Array], jax.Array],
    dim: int,
    n_bands: int,
    *,
    x0=None,
    preconditioner=None,
    tol=None,
    n_converge=None,
    column_batch_size: int | None = 64,
    return_info: bool = False,
):
    """Lowest ``n_bands`` eigenpairs of the Hermitian operator ``apply`` on C^dim.

    Builds the dense matrix by applying the operator to the identity, then uses
    ``eigh``. Returns eigenvalues (n_bands,) ascending and orthonormal eigenvectors
    (dim, n_bands) as columns. ``x0``, ``preconditioner``, ``tol`` and ``n_converge``
    are accepted for the common eigensolver interface and ignored: the result is
    exact to rounding.

    The identity is applied in chunks of ``column_batch_size`` columns
    (``lax.map``), so the operator's temporaries (one wavefunction on the FFT grid
    per column) do not grow with ``dim``; ``None`` applies it to all columns at once.
    With ``return_info`` it also returns an ``EigenInfo`` of zeros (not iterative).
    """
    if column_batch_size is None or column_batch_size >= dim:
        H = apply(jnp.eye(dim, dtype=jnp.complex128))
    else:

        def column(i):
            return apply(jax.nn.one_hot(i, dim, dtype=jnp.complex128)[:, None])[:, 0]

        H = jax.lax.map(column, jnp.arange(dim), batch_size=int(column_batch_size)).T
    H = 0.5 * (H + H.conj().T)  # remove rounding-level anti-Hermitian part
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    out = (eigenvalues[:n_bands], eigenvectors[:, :n_bands])
    return (*out, _no_iterations()) if return_info else out
