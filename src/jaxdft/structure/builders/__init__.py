"""Factories for common lattices and structures, one module each."""

from jaxdft.structure.builders.diamond import diamond
from jaxdft.structure.builders.empty import empty
from jaxdft.structure.builders.fcc import fcc_lattice

__all__ = ["diamond", "empty", "fcc_lattice"]
