"""The empty lattice: a Structure with no atoms (Milestone 1a)."""

import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice
from jaxdft.structure.structure import Structure


def empty(lattice: Lattice) -> Structure:
    return Structure(lattice, jnp.zeros((0, 3)), (), jnp.zeros(0))
