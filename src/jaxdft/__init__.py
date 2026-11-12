"""jaxdft: a GPU-first plane-wave density functional theory code in JAX."""

# Must be imported before anything else creates a JAX array.
from jaxdft import _config  # noqa: F401

import logging as _logging

from jaxdft.output.log import LOGGER_NAME as _LOGGER_NAME
from jaxdft.output.log import log_to_file, log_to_stream

# A library emits log records but does not decide where they go.
_logging.getLogger(_LOGGER_NAME).addHandler(_logging.NullHandler())

from jaxdft.analytic.free_electron import free_electron_bands
from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.calculations.bands import Bands, band_structure
from jaxdft.hamiltonian.atomic_local import AtomicLocal
from jaxdft.hamiltonian.atomic_nonlocal import AtomicNonlocal
from jaxdft.hamiltonian.ewald import Ewald
from jaxdft.hamiltonian.hartree import Hartree
from jaxdft.hamiltonian.kinetic import Kinetic
from jaxdft.hamiltonian.local import LocalPotential
from jaxdft.hamiltonian.model import lda_terms
from jaxdft.hamiltonian.operator import Hamiltonian, apply_H, dense_matrix, hamiltonian
from jaxdft.hamiltonian.xc import Xc
from jaxdft.precision import assert_double_precision, is_double_precision
from jaxdft.scf.mixing import Anderson
from jaxdft.scf.scf import SCFError, SCFOptions, SCFResult, energy, scf, total_energy
from jaxdft.solvers.eigen import dense_eigensolver
from jaxdft.solvers.lobpcg import lobpcg
from jaxdft.structure.builders import diamond, dimer_in_box, empty, fcc_lattice, supercell
from jaxdft.structure.kpoints import KPath, kpath, monkhorst_pack, special_points_fcc
from jaxdft.structure.lattice import Lattice
from jaxdft.structure.structure import Structure

__all__ = [
    "Anderson",
    "AtomicLocal",
    "AtomicNonlocal",
    "Bands",
    "Ewald",
    "Hartree",
    "SCFError",
    "SCFOptions",
    "SCFResult",
    "Xc",
    "dimer_in_box",
    "energy",
    "lda_terms",
    "lobpcg",
    "log_to_file",
    "log_to_stream",
    "scf",
    "total_energy",
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
    "monkhorst_pack",
    "supercell",
    "special_points_fcc",
]
