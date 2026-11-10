"""Eigensolver preconditioners built from a prepared Hamiltonian (theory post 2a,
'The eigensolver').

A preconditioner factory takes a ``Hamiltonian`` at one k-point and returns
``precondition(r, eps) -> w``, mapping residuals (n_pad, n) and the current Ritz
values (n,) to search directions. The SCF builds one per k-point and SCF step, under
``jit``; the factory is a static argument, so it must be hashable (a plain function
is).
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxdft.hamiltonian.operator import Hamiltonian


def diagonal_preconditioner(ham: Hamiltonian):
    """w = r / (max(D - eps, 0) + 1 Ha), with D the diagonal of the prepared terms
    (the kinetic energy): damps the high-|G| components of the residual, in the
    spirit of Teter, Payne and Allan."""
    diag = ham.diagonal()

    def precondition(r, eps):
        return r / (jnp.maximum(diag[:, None] - eps[None, :], 0.0) + 1.0)

    return precondition
