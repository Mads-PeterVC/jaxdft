"""Matrix-free Hamiltonian (theory post 1a, 'The kinetic operator', 'Padding must be
exactly decoupled').

Term protocol. A term is a pytree with

    prepare(structure, basis_k) -> prepared term

where the prepared term is a pytree with ``apply(psi) -> H_term psi`` acting on
blocks ``psi`` of shape (n_pad, n). A prepared term that is diagonal in the basis
may also expose ``diagonal`` (n_pad,), which ``Hamiltonian.diagonal`` sums for
preconditioners. Terms never see padding: ``apply_H`` masks
their input and output and puts a sentinel energy on padded components, so a
new term cannot leak into the physical block by forgetting a mask.

Optionally a term may define ``bind(structure) -> term``. ``band_structure`` calls it on
the host before tracing, to validate the term against the structure and to precompute
data that fixes array shapes (e.g. coupling vectors). ``prepare`` must then do only
traced work.

Energy terms (``jaxdft.hamiltonian.energy_terms``: ``energy(structure, basis, density)``,
no ``prepare``) enter H through one real-space potential, the sum of their autodiff
potentials at ``density``, applied with one FFT pair per block of bands (theory post 2a).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.hamiltonian.energy_terms import is_energy_term, total_potential
from jaxdft.structure.structure import Structure

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

    def diagonal(self) -> jax.Array:
        """Sum of the ``diagonal`` of the prepared terms that expose one (n_pad,): the
        kinetic energy for the standard terms. Not the full diagonal of H, which would
        also need the mean of the grid potential."""
        n_pad = self.mask.shape[0]
        return sum((t.diagonal for t in self.terms if hasattr(t, "diagonal")), jnp.zeros(n_pad))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GridPotential:
    """A prepared local potential V(r) on the FFT grid: psi -> FFT[V IFFT[psi]] restricted
    to the basis. Exact for the grid coefficients of V (theory post 2a, claim 2)."""

    potential: jax.Array
    basis_k: PlaneWaveBasis

    def apply(self, psi):
        b = self.basis_k
        return b.from_grid(self.potential[..., None] * b.to_grid(psi))


def hamiltonian(
    structure: Structure, basis_k: PlaneWaveBasis, terms: tuple, density=None, potential=None
) -> Hamiltonian:
    """Prepare every term once for the single-k basis ``basis_k``.

    Operator terms are prepared directly. Energy terms contribute their summed potential
    at ``density`` (None: zero density); ``potential`` may be passed precomputed, since
    it does not depend on k.
    """
    operators = tuple(t for t in terms if not is_energy_term(t))
    prepared = tuple(term.prepare(structure, basis_k) for term in operators)
    if len(operators) < len(terms):
        if potential is None:
            potential = total_potential(terms, structure, basis_k, density)
        prepared = (*prepared, GridPotential(potential, basis_k))
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
