"""Kinetic energy -1/2 nabla^2: diagonal, 1/2 |k+G|^2, in the plane-wave basis."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.structure.structure import Structure


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiagonalTerm:
    """A prepared term that is diagonal in the basis."""

    diagonal: jax.Array

    def apply(self, psi: jax.Array) -> jax.Array:
        return self.diagonal[:, None] * psi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Kinetic:
    """The kinetic-energy term. Has no parameters."""

    def prepare(self, structure: Structure, basis_k: PlaneWaveBasis) -> DiagonalTerm:
        q = (basis_k.kpoints + basis_k.miller) @ structure.lattice.reciprocal  # Cartesian k+G
        return DiagonalTerm(0.5 * jnp.sum(q**2, axis=-1))
