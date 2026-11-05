"""Matrix-free Hamiltonian (theory post 1a, 'The kinetic operator', 'Padding must be
exactly decoupled').

Term protocol. A term is a pytree with

    prepare(lattice, basis_k) -> prepared term

where the prepared term is a pytree with ``apply(psi) -> H_term psi`` acting on
blocks ``psi`` of shape (n_pad, n). Terms never see padding: ``apply_H`` masks
their input and output and puts a sentinel energy on padded components, so a
new term cannot leak into the physical block by forgetting a mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.structure.lattice import Lattice

# Padded components get the eigenvalue SENTINEL_FACTOR * ecut: above the kinetic
# spectrum (||T_phys|| <= ecut), but only a small multiple of it, so iterative
# solvers do not see a spectrum spread over many orders of magnitude.
SENTINEL_FACTOR = 2.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Hamiltonian:
    """H at a single k-point: prepared terms plus the padding mask."""

    terms: tuple
    mask: jax.Array
    sentinel: float = field(metadata=dict(static=True))


def hamiltonian(lattice: Lattice, basis_k: PlaneWaveBasis, terms: tuple) -> Hamiltonian:
    """Prepare every term once for the single-k basis ``basis_k``."""
    prepared = tuple(term.prepare(lattice, basis_k) for term in terms)
    return Hamiltonian(terms=prepared, mask=basis_k.mask, sentinel=SENTINEL_FACTOR * basis_k.ecut)


def apply_H(ham: Hamiltonian, psi: jax.Array) -> jax.Array:
    """H psi for a block psi of shape (n_pad, n).

    Physical components get the sum of the terms applied to the masked input;
    padded components get sentinel * psi, with no coupling in either direction.
    """
    real = ham.mask[:, None]
    x = jnp.where(real, psi, 0)
    hx = sum(term.apply(x) for term in ham.terms)
    return jnp.where(real, hx, ham.sentinel * psi)


def dense_matrix(ham: Hamiltonian) -> jax.Array:
    """The dense (n_pad, n_pad) matrix of H, built by applying the operator to the identity."""
    n = ham.mask.shape[0]
    return apply_H(ham, jnp.eye(n, dtype=jnp.complex128))
