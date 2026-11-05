"""Eigensolvers for Hermitian linear operators."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def dense_eigensolver(
    apply: Callable[[jax.Array], jax.Array], dim: int, n_bands: int
) -> tuple[jax.Array, jax.Array]:
    """Lowest ``n_bands`` eigenpairs of the Hermitian operator ``apply`` on C^dim.

    Builds the dense matrix by applying the operator to the identity, then uses
    ``eigh``. Returns eigenvalues (n_bands,) ascending and orthonormal eigenvectors
    (dim, n_bands) as columns.
    """
    H = apply(jnp.eye(dim, dtype=jnp.complex128))
    H = 0.5 * (H + H.conj().T)  # remove rounding-level anti-Hermitian part
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    return eigenvalues[:n_bands], eigenvectors[:, :n_bands]
