"""The public API: subpackage layout and top-level re-exports (1a + 1b)."""

import jaxdft


def test_public_names_live_in_subpackages_and_are_reexported():
    from jaxdft.analytic.free_electron import free_electron_bands
    from jaxdft.basis.planewave import PlaneWaveBasis
    from jaxdft.calculations.bands import Bands, band_structure
    from jaxdft.hamiltonian.kinetic import Kinetic
    from jaxdft.hamiltonian.local import LocalPotential
    from jaxdft.hamiltonian.operator import Hamiltonian, apply_H, dense_matrix, hamiltonian
    from jaxdft.solvers.eigen import dense_eigensolver
    from jaxdft.structure.builders.diamond import diamond
    from jaxdft.structure.builders.empty import empty
    from jaxdft.structure.builders.fcc import fcc_lattice
    from jaxdft.structure.kpoints import KPath, kpath, special_points_fcc
    from jaxdft.structure.lattice import Lattice
    from jaxdft.structure.structure import Structure

    for obj in [
        free_electron_bands, PlaneWaveBasis, Bands, band_structure, Kinetic, LocalPotential,
        Hamiltonian, apply_H, dense_matrix, hamiltonian, dense_eigensolver, diamond, empty,
        fcc_lattice, KPath, kpath, special_points_fcc, Lattice, Structure,
    ]:
        assert getattr(jaxdft, obj.__name__) is obj


def test_builders_subpackage_exports():
    from jaxdft.structure import builders

    assert builders.diamond is jaxdft.diamond
    assert builders.empty is jaxdft.empty
    assert builders.fcc_lattice is jaxdft.fcc_lattice


def test_lattice_has_only_the_general_constructor():
    """Named lattices moved to structure/builders (1b API decision)."""
    assert not hasattr(jaxdft.Lattice, "fcc")


def test_non_reexported_modules():
    """Pseudopotentials, units and analysis are reached through their modules."""
    from jaxdft import units  # noqa: F401
    from jaxdft.analysis.bands import band_minimum_on_segment, valence_band_maximum  # noqa: F401
    from jaxdft.analysis.symmetry import group_levels, inversion_parities  # noqa: F401
    from jaxdft.pseudo.cohen_bergstresser import CohenBergstresser, silicon  # noqa: F401
