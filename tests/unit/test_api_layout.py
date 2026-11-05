"""The 1a public API: subpackage layout and top-level re-exports."""

import jaxdft


def test_public_names_live_in_subpackages_and_are_reexported():
    from jaxdft.analytic.free_electron import free_electron_bands
    from jaxdft.basis.planewave import PlaneWaveBasis
    from jaxdft.calculations.bands import Bands, band_structure
    from jaxdft.hamiltonian.kinetic import Kinetic
    from jaxdft.hamiltonian.operator import Hamiltonian, apply_H, dense_matrix, hamiltonian
    from jaxdft.solvers.eigen import dense_eigensolver
    from jaxdft.structure.kpoints import KPath, kpath, special_points_fcc
    from jaxdft.structure.lattice import Lattice

    for obj in [
        free_electron_bands, PlaneWaveBasis, Bands, band_structure, Kinetic, Hamiltonian,
        apply_H, dense_matrix, hamiltonian, dense_eigensolver, KPath, kpath,
        special_points_fcc, Lattice,
    ]:
        assert getattr(jaxdft, obj.__name__) is obj
