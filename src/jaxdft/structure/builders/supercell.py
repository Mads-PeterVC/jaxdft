"""Supercells: n_1 x n_2 x n_3 repetitions of a structure (theory post 2b,
'Equivalence to a supercell')."""

import itertools

import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice
from jaxdft.structure.structure import Structure


def supercell(structure: Structure, repeats) -> Structure:
    """Lattice rows a_i n_i; the images of the cell translation by translation
    (translations outer, atoms inner), with species and weights tiled."""
    repeats = tuple(int(n) for n in repeats)
    if len(repeats) != 3 or min(repeats) < 1:
        raise ValueError(f"repeats must be three positive integers, got {repeats}")
    n = jnp.asarray(repeats, dtype=jnp.float64)
    A = structure.lattice.vectors * n[:, None]
    shifts = jnp.asarray(list(itertools.product(*[range(r) for r in repeats])), dtype=jnp.float64)
    f = (shifts[:, None, :] + structure.positions[None, :, :]) / n
    count = shifts.shape[0]
    return Structure(
        Lattice(A),
        f.reshape(-1, 3),
        structure.species * count,
        jnp.tile(structure.weights, count),
    )
