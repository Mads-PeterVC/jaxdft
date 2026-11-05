"""Diamond structure: fcc with a two-atom basis, origin at the bond centre."""

from jaxdft.structure.builders.fcc import fcc_lattice
from jaxdft.structure.structure import Structure


def diamond(a, species: str, *, unit: str) -> Structure:
    """Atoms at +-(1/8, 1/8, 1/8) fractional, i.e. +-(a/8)(1,1,1): the Cohen-Bergstresser
    origin, an inversion centre. ``a`` is the cubic lattice constant in ``unit``."""
    return Structure(fcc_lattice(a, unit=unit), [[0.125] * 3, [-0.125] * 3], [species, species])
