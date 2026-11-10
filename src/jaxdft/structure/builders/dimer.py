"""A diatomic molecule in a cubic box (Milestone 2a)."""

from __future__ import annotations

import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice
from jaxdft.structure.structure import Structure
from jaxdft.units import convert

_LENGTH_UNITS = ("Bohr", "Angstrom")


def dimer_in_box(species, bond_length: float, box: float, axis: int = 2, unit: str = "Bohr") -> Structure:
    """Two atoms along Cartesian ``axis`` in a cube of side ``box``, with the bond
    centre at the box centre, fractional (1/2, 1/2, 1/2). On an even FFT grid that
    centre is a grid point, so the molecule's D4h operations map the grid to itself.

    ``species`` is one label (homonuclear) or a pair. ``bond_length`` and ``box`` are
    in ``unit`` and converted once.
    """
    if unit not in _LENGTH_UNITS:
        raise ValueError(f"unit must be one of {_LENGTH_UNITS}, got {unit!r}")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    pair = (species, species) if isinstance(species, str) else tuple(species)
    if len(pair) != 2:
        raise ValueError(f"a dimer needs two species labels, got {pair}")
    d = float(convert(bond_length, unit, "Bohr"))
    L = float(convert(box, unit, "Bohr"))
    if not 0 < d < L:
        raise ValueError(f"bond length {bond_length} must be positive and shorter than the box {box}")
    half = 0.5 * d / L
    f = jnp.full((2, 3), 0.5).at[0, axis].add(-half).at[1, axis].add(half)
    return Structure(Lattice(L * jnp.eye(3)), f, pair)
