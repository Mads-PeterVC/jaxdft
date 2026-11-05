"""Face-centred cubic lattice."""

import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice
from jaxdft.units import convert

_FCC = jnp.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def fcc_lattice(a, *, unit: str) -> Lattice:
    """fcc primitive vectors (rows) a/2 (0,1,1), a/2 (1,0,1), a/2 (1,1,0); ``a`` is the
    conventional cubic lattice constant in ``unit`` ("Bohr" or "Angstrom")."""
    if unit not in ("Bohr", "Angstrom"):
        raise ValueError(f"unit must be 'Bohr' or 'Angstrom', got {unit!r}")
    return Lattice(0.5 * convert(a, unit, "Bohr") * _FCC)
