"""Band structures: eigenvalues of H(k) over a batch of k-points (theory post 1a)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.hamiltonian.energy_terms import total_potential
from jaxdft.hamiltonian.kinetic import Kinetic
from jaxdft.hamiltonian.operator import apply_H, hamiltonian
from jaxdft.solvers.eigen import dense_eigensolver
from jaxdft.structure.structure import Structure


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Bands:
    """Result of ``band_structure``.

    eigenvalues:  (n_k, n_bands) Hartree, ascending per k.
    eigenvectors: (n_k, n_pad, n_bands) plane-wave coefficients, zero on padding.
    kpoints:      (n_k, 3) fractional.
    basis:        the PlaneWaveBasis used.
    """

    eigenvalues: jax.Array
    eigenvectors: jax.Array
    kpoints: jax.Array
    basis: PlaneWaveBasis


def bind_terms(structure: Structure, terms) -> tuple:
    """Host-side, before tracing: let each term validate itself against ``structure``
    and precompute shape-fixing data (the optional ``bind`` hook)."""
    return tuple(term.bind(structure) if hasattr(term, "bind") else term for term in terms)


def band_structure(
    structure: Structure,
    basis: PlaneWaveBasis,
    n_bands: int,
    terms: tuple = (Kinetic(),),
    solver: Callable = dense_eigensolver,
    density=None,
) -> Bands:
    """Lowest ``n_bands`` eigenpairs of H(k) at every k-point of ``basis``.

    k-points are a batch dimension: one jitted, vmapped computation. Requires
    n_bands <= min_k N_PW so that no padding eigenvalue is returned. Only
    eigenvalues below ``basis.ecut`` are guaranteed to lie below the padding
    sentinel when ``terms`` contains more than the kinetic energy.

    Energy terms (local pseudopotential, Hartree, XC) contribute their potentials at the
    fixed ``density`` on the basis grid (None: zero density), e.g. a converged SCF
    density for non-self-consistent bands.
    """
    n_bands = int(n_bands)
    min_pw = int(jnp.min(basis.n_pw))
    if not 1 <= n_bands <= min_pw:
        raise ValueError(f"n_bands={n_bands} must be between 1 and min_k N_PW={min_pw}")
    terms = bind_terms(structure, terms)
    if density is not None:
        density = jnp.asarray(density, dtype=jnp.float64)
    eigenvalues, eigenvectors = _solve(structure, basis, terms, n_bands, solver, density)
    return Bands(eigenvalues, eigenvectors, basis.kpoints, basis)


@partial(jax.jit, static_argnums=(3, 4))
def _solve(structure, basis, terms, n_bands, solver, density):
    potential = total_potential(terms, structure, basis, density)

    def per_k(basis_k):
        ham = hamiltonian(structure, basis_k, terms, potential=potential)
        eps, vecs = solver(lambda psi: apply_H(ham, psi), dim=basis_k.n_pad, n_bands=n_bands)
        return eps, jnp.where(basis_k.mask[:, None], vecs, 0)

    return jax.vmap(per_k)(basis)
