"""Direct and reciprocal lattice (theory post 1a, 'Direct lattice', 'Reciprocal lattice').

Named lattices (fcc, ...) live in ``jaxdft.structure.builders``."""

from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class Lattice:
    """A Bravais lattice. Primitive vectors are the rows of ``vectors`` (Bohr).

    A frozen pytree whose only leaf is ``vectors``, so functions of a ``Lattice``
    can be jitted, vmapped and differentiated with respect to the cell.
    """

    __slots__ = ("vectors",)

    def __init__(self, vectors):
        vectors = jnp.asarray(vectors, dtype=jnp.float64)
        if vectors.shape != (3, 3):
            raise ValueError(f"lattice vectors must have shape (3, 3), got {vectors.shape}")
        object.__setattr__(self, "vectors", vectors)

    def __setattr__(self, name, value):
        raise AttributeError("Lattice is immutable")

    def __repr__(self):
        return f"Lattice(vectors={self.vectors!r})"

    @property
    def reciprocal(self) -> jax.Array:
        """Reciprocal primitive vectors as rows: B = 2 pi (A^-1)^T, so A B^T = 2 pi I."""
        return 2.0 * jnp.pi * jnp.linalg.inv(self.vectors).T

    @property
    def volume(self) -> jax.Array:
        """Primitive cell volume |det A| (Bohr^3)."""
        return jnp.abs(jnp.linalg.det(self.vectors))

    def tree_flatten(self):
        return (self.vectors,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        # Bypass __init__: JAX unflattens with tracers, cotangents and placeholders.
        obj = object.__new__(cls)
        object.__setattr__(obj, "vectors", children[0])
        return obj
