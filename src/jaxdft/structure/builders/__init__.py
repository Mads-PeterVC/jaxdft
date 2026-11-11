"""Factories for common lattices and structures, one module each."""

from jaxdft.structure.builders.diamond import diamond
from jaxdft.structure.builders.dimer import dimer_in_box
from jaxdft.structure.builders.empty import empty
from jaxdft.structure.builders.fcc import fcc_lattice
from jaxdft.structure.builders.supercell import supercell

__all__ = ["diamond", "dimer_in_box", "empty", "fcc_lattice", "supercell"]
