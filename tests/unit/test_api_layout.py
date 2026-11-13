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


# --- Milestone 2a (engineering log, '2026-11-06: Milestone 2a public API') ----------


def test_2a_public_names_live_in_subpackages_and_are_reexported():
    from jaxdft.hamiltonian.atomic_local import AtomicLocal
    from jaxdft.hamiltonian.ewald import Ewald
    from jaxdft.hamiltonian.hartree import Hartree
    from jaxdft.hamiltonian.model import lda_terms
    from jaxdft.hamiltonian.xc import Xc
    from jaxdft.scf.mixing import Anderson
    from jaxdft.scf.scf import SCFError, SCFOptions, SCFResult, energy, scf, total_energy
    from jaxdft.solvers.lobpcg import lobpcg
    from jaxdft.structure.builders.dimer import dimer_in_box

    for obj in [
        AtomicLocal, Ewald, Hartree, lda_terms, Xc, Anderson, SCFError, SCFOptions,
        SCFResult, energy, scf, total_energy, lobpcg, dimer_in_box,
    ]:
        assert getattr(jaxdft, obj.__name__) is obj


def test_2a_builders_subpackage_exports():
    from jaxdft.structure import builders

    assert builders.dimer_in_box is jaxdft.dimer_in_box


def test_2a_non_reexported_modules():
    """Pseudopotentials, functionals and SCF internals are reached through modules."""
    from jaxdft.hamiltonian.energy_terms import term_potential  # noqa: F401
    from jaxdft.hamiltonian.ewald import ewald_energy  # noqa: F401
    from jaxdft.pseudo.gth import GTH, hgh_lda, hgh_lda_path, parse_hgh  # noqa: F401
    from jaxdft.scf.density import clip_and_renormalise, electron_density  # noqa: F401
    from jaxdft.scf.occupations import aufbau_occupations  # noqa: F401
    from jaxdft.scf.solve import FixedPointResult, fixed_point  # noqa: F401
    from jaxdft.xc.lda import PW92_UNPOLARISED, LDA, lda_pw92, lda_x  # noqa: F401


def test_every_eigensolver_accepts_warm_start_and_preconditioner_keywords():
    """SCF passes x0 and preconditioner to whichever eigensolver it is given
    (engineering log: the 1a signature plus optional keywords)."""
    import inspect

    for solver in (jaxdft.dense_eigensolver, jaxdft.lobpcg):
        params = inspect.signature(solver).parameters
        assert list(params)[:3] == ["apply", "dim", "n_bands"]
        assert "x0" in params and "preconditioner" in params


# --- observability (engineering log, '2026-11-10: SCF observability') -----------------


def test_logging_helpers_are_reexported_and_the_package_is_silent_by_default():
    import logging

    from jaxdft.output.log import LOGGER_NAME, log_to_file, log_to_stream

    assert jaxdft.log_to_file is log_to_file and jaxdft.log_to_stream is log_to_stream
    handlers = logging.getLogger(LOGGER_NAME).handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


def test_every_eigensolver_can_report_its_diagnostics():
    """The SCF records eigensolver diagnostics: ``return_info=True`` appends an
    ``EigenInfo`` (iterations, stagnated, late_drops). It replaced
    ``return_iterations`` (engineering log, '2026-11-12: Milestone 2c public API')."""
    import inspect

    import jax.numpy as jnp

    from jaxdft.solvers.eigen import EigenInfo

    H = jnp.diag(jnp.arange(1.0, 7.0))
    for solver in (jaxdft.dense_eigensolver, jaxdft.lobpcg):
        assert "return_iterations" not in inspect.signature(solver).parameters
        _, vecs, info = solver(lambda x: H @ x, 6, 2, return_info=True)
        assert isinstance(info, EigenInfo)
        assert vecs.shape == (6, 2) and int(info.iterations) >= 0
        assert int(info.late_drops) >= 0 and bool(info.stagnated) in (True, False)
        assert len(solver(lambda x: H @ x, 6, 2)) == 2
    _, _, info = jaxdft.dense_eigensolver(lambda x: H @ x, 6, 2, return_info=True)
    assert int(info.iterations) == 0 and int(info.late_drops) == 0 and not bool(info.stagnated)


# --- Milestone 2c (engineering log, '2026-11-12: Milestone 2c public API') -------------


def test_every_eigensolver_accepts_n_converge():
    import inspect

    for solver in (jaxdft.dense_eigensolver, jaxdft.lobpcg):
        p = inspect.signature(solver).parameters["n_converge"]
        assert p.default is None


def test_2c_eigensolver_keywords():
    import inspect

    lob = inspect.signature(jaxdft.lobpcg).parameters
    assert lob["stagnation_window"].default == 5
    assert lob["stagnation_factor"].default == 0.8
    assert lob["max_iter"].default == 300
    dense = inspect.signature(jaxdft.dense_eigensolver).parameters
    assert dense["column_batch_size"].default == 64


def test_2c_scf_options():
    o = jaxdft.SCFOptions()
    assert o.eigensolver_tol_ratio > 0  # adaptive by default; the value is measured
    assert o.eigensolver_tol_max == 5e-3
    # One batch was agreed until measured; 64 cost nothing on the V100, was 14 %
    # faster on CPUs and cut memory to a third (engineering log, 2c cluster runs).
    assert o.k_batch_size == 64
    # 2b left the default to a V100 measurement; 3 was fastest there and costs
    # nothing on the laptop CPU (engineering log, 2c cluster runs).
    assert o.buffer_bands == 3


def test_2c_floor_constant_replaces_the_absolute_floor():
    import importlib

    # The package re-exports the function scf, which shadows the module of that name.
    scf_module = importlib.import_module("jaxdft.scf.scf")

    assert scf_module.EIGEN_FLOOR_FACTOR == 50
    assert not hasattr(scf_module, "EIGEN_FLOOR")


def test_2c_band_structure_signatures():
    import inspect

    from jaxdft.analysis.bands import band_minimum_on_segment

    assert inspect.signature(jaxdft.band_structure).parameters["k_batch_size"].default is None
    assert inspect.signature(band_minimum_on_segment).parameters["k_batch_size"].default == 8


# --- Milestone 2b (engineering log, '2026-11-11: Milestone 2b public API') -------------


def test_2b_public_names_live_in_subpackages_and_are_reexported():
    from jaxdft.hamiltonian.atomic_nonlocal import AtomicNonlocal
    from jaxdft.structure.builders.supercell import supercell
    from jaxdft.structure.kpoints import monkhorst_pack

    for obj in [AtomicNonlocal, supercell, monkhorst_pack]:
        assert getattr(jaxdft, obj.__name__) is obj


def test_2b_builders_subpackage_exports():
    from jaxdft.structure import builders

    assert builders.supercell is jaxdft.supercell


def test_2b_non_reexported_modules():
    from jaxdft.pseudo.gth import HGHChannel  # noqa: F401
    from jaxdft.scf.occupations import grid_occupations  # noqa: F401


def test_2b_signatures():
    import inspect

    from jaxdft.analysis.bands import band_minimum_on_segment

    build = inspect.signature(jaxdft.PlaneWaveBasis.build).parameters
    assert "kweights" in build and "fft_multiple" in build
    assert "density" in inspect.signature(band_minimum_on_segment).parameters
    assert "kweights" not in inspect.signature(jaxdft.total_energy).parameters
