"""Atomic structure: any periodic configuration of atoms (theory post 1b,
'The diamond structure'; 1b API decisions).

A bulk crystal, a slab or a molecule in a vacuum box are all a Structure: a lattice,
fractional positions and per-atom weights. Positions are always *stored* fractional:
G . tau = 2 pi m . f holds only for fractional coordinates.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice
from jaxdft.units import convert

_LENGTH_UNITS = ("Bohr", "Angstrom")


@jax.tree_util.register_pytree_node_class
class Structure:
    """Lattice + fractional positions (n_atoms, 3) + weights (n_atoms,) + species labels.

    Leaves: the lattice vectors, ``positions`` and ``weights``. ``species`` is static.
    Weights lambda scale each atom's contribution (CLAUDE.md, per-atom weights); their
    physical range is [0, 1] but it is not enforced, so finite differences can step past 1.
    """

    __slots__ = ("lattice", "positions", "species", "weights")

    def __init__(self, lattice: Lattice, positions, species: Sequence[str], weights=None):
        positions = jnp.asarray(positions, dtype=jnp.float64)
        if positions.size == 0:
            positions = positions.reshape(0, 3)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"positions must have shape (n_atoms, 3), got {positions.shape}")
        species = tuple(str(s) for s in species)
        if len(species) != positions.shape[0]:
            raise ValueError(f"{len(species)} species labels for {positions.shape[0]} atoms")
        n = positions.shape[0]
        weights = jnp.ones(n, dtype=jnp.float64) if weights is None else jnp.asarray(weights, dtype=jnp.float64)
        if weights.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {weights.shape}")
        for name, value in zip(self.__slots__, (lattice, positions, species, weights)):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("Structure is immutable")

    def __repr__(self):
        return f"Structure(n_atoms={self.n_atoms}, species={self.species})"

    @classmethod
    def from_cartesian(cls, cell, positions, species: Sequence[str], unit: str, weights=None) -> Structure:
        """Build from Cartesian cell vectors (rows) and positions, both in ``unit``
        ("Bohr" or "Angstrom"). Converts once; Cartesian positions are never stored."""
        if unit not in _LENGTH_UNITS:
            raise ValueError(f"unit must be one of {_LENGTH_UNITS}, got {unit!r}")
        A = convert(cell, unit, "Bohr")
        r = jnp.asarray(convert(positions, unit, "Bohr")).reshape(-1, 3)
        return cls(Lattice(A), r @ jnp.linalg.inv(A), species, weights)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

    @property
    def cartesian_positions(self) -> jax.Array:
        """Positions in Bohr, tau = f A."""
        return self.positions @ self.lattice.vectors

    def tree_flatten(self):
        return (self.lattice, self.positions, self.weights), self.species

    @classmethod
    def tree_unflatten(cls, species, children):
        obj = object.__new__(cls)
        lattice, positions, weights = children
        for name, value in zip(cls.__slots__, (lattice, positions, species, weights)):
            object.__setattr__(obj, name, value)
        return obj
