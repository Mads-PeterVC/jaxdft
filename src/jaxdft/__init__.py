"""jaxdft: a GPU-first plane-wave density functional theory code in JAX."""

# Must be imported before anything else creates a JAX array.
from jaxdft import _config  # noqa: F401
from jaxdft.analytic.free_electron import free_electron_bands
from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.calculations.bands import Bands, band_structure
from jaxdft.hamiltonian.kinetic import Kinetic
from jaxdft.hamiltonian.local import LocalPotential
from jaxdft.hamiltonian.operator import Hamiltonian, apply_H, dense_matrix, hamiltonian
from jaxdft.precision import assert_double_precision, is_double_precision
from jaxdft.solvers.eigen import dense_eigensolver
from jaxdft.structure.builders import diamond, empty, fcc_lattice
from jaxdft.structure.kpoints import KPath, kpath, special_points_fcc
from jaxdft.structure.lattice import Lattice
from jaxdft.structure.structure import Structure

__all__ = [
    "Bands",
    "Hamiltonian",
    "KPath",
    "Kinetic",
    "Lattice",
    "LocalPotential",
    "PlaneWaveBasis",
    "Structure",
    "apply_H",
    "assert_double_precision",
    "band_structure",
    "dense_eigensolver",
    "dense_matrix",
    "diamond",
    "empty",
    "fcc_lattice",
    "free_electron_bands",
    "hamiltonian",
    "is_double_precision",
    "kpath",
    "special_points_fcc",
]
