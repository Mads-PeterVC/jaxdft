"""jaxdft against DFTK on the same discrete problem (theory post 02b, 'Against
references' and 'Tolerances').

Both codes use the same HGH file, Slater + PW92, the same cell, cutoff and 24^3 grid,
Gamma-centred k-grids without symmetry reduction and zero temperature: the ideal
geometry on the 2^3 test grid, and the displaced geometry on 3^3 (where the gap over
the grid stays open, tests/support/silicon.py). The comparison is term by term and
k-point by k-point. jaxdft's grid is its own monkhorst_pack in [0, 1); DFTK lists the
same points in [-1/2, 1/2), so k-points are matched modulo 1 and the agreement also
tests that relabelling k by a reciprocal lattice vector leaves the problem unchanged.
DFTK reports the G = 0 part of the local pseudopotential separately as PspCorrection
and keeps it out of the Hamiltonian, so its eigenvalues are higher than ours by
-sum_j alpha_j / Omega, asserted explicitly.
"""

import hashlib

import numpy as np
import pytest

from jaxdft.pseudo.gth import hgh_lda_path
from support.reference import dftk_si
from support.silicon import (
    A_SI,
    ECUT_TEST,
    KGRID_DISPLACED,
    KGRID_TEST,
    N_TEST,
    basis_for,
    displaced_si,
    forces,
    run,
    si,
    silicon,
)
from support.tolerances import (
    DFTK_EIGENVALUE_ATOL,
    DFTK_ENERGY_ATOL,
    DFTK_FORCE_ATOL,
    PSP_CORRECTION_ATOL,
)

pytestmark = pytest.mark.slow

REF = dftk_si()
SYSTEMS = {s["name"]: s for s in REF["systems"]}
STRUCTURES = {"ideal": (si, KGRID_TEST), "displaced": (displaced_si, KGRID_DISPLACED)}
N_BANDS = 5  # the bands DFTK converged: 4 occupied and the lowest conduction band
TOL = 1e-10  # our SCF, tighter than the acceptance default


def k_order(system, basis) -> np.ndarray:
    """For each of our k-points, the index of the DFTK k-point equal to it modulo 1."""
    ours = np.asarray(basis.kpoints)
    theirs = np.asarray(system["kpoints_fractional"])
    d = ours[:, None, :] - theirs[None, :, :]
    match = np.abs(d - np.rint(d)).max(axis=-1) < 1e-12
    assert np.all(match.sum(axis=1) == 1) and np.all(match.sum(axis=0) == 1)
    return np.argmax(match, axis=1)


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def case(request):
    system = SYSTEMS[request.param]
    build, kshape = STRUCTURES[request.param]
    s = build()
    b = basis_for(s, kshape=kshape)
    return system, s, b, run(s, b, tol=TOL)


def alpha_shift(system):
    """sum_j alpha_j / Omega (Ha): the G = 0 local potential DFTK leaves out of H."""
    omega = abs(np.linalg.det(np.asarray(system["lattice_bohr"])))
    return len(system["positions_fractional"]) * silicon().alpha / omega


def test_reference_identifies_the_same_problem():
    assert REF["functional"] == ["lda_x", "lda_c_pw"]
    assert REF["pseudopotential"]["sha256"] == hashlib.sha256(hgh_lda_path("Si").read_bytes()).hexdigest()
    assert set(SYSTEMS) == set(STRUCTURES)
    for name, system in SYSTEMS.items():
        build, kshape = STRUCTURES[name]
        s = build()
        np.testing.assert_allclose(system["lattice_bohr"], np.asarray(s.lattice.vectors), atol=1e-14, rtol=0)
        np.testing.assert_allclose(system["positions_fractional"], np.asarray(s.positions), atol=1e-15, rtol=0)
        assert np.asarray(s.lattice.vectors)[0, 1] == A_SI / 2
        assert system["ecut_hartree"] == ECUT_TEST
        assert system["fft_size"] == [N_TEST] * 3
        assert tuple(system["kgrid"]) == kshape
        # No symmetry reduction, so every grid point is listed with weight 1/n_k.
        assert system["n_symmetries"] == 1
        n_k = int(np.prod(kshape))
        assert len(system["kpoints_fractional"]) == n_k
        np.testing.assert_allclose(system["kweights"], 1.0 / n_k, rtol=1e-15)
        # Zero temperature: the insulating filling at every k-point.
        assert all(occ == [2.0, 2.0, 2.0, 2.0, 0.0] for occ in system["occupations"])


def test_same_basis(case):
    system, _, b, _ = case
    order = k_order(system, b)
    assert tuple(b.fft_shape) == tuple(system["fft_size"])
    np.testing.assert_array_equal(np.asarray(b.n_pw), np.asarray(system["n_plane_waves"])[order])
    np.testing.assert_allclose(np.asarray(b.kweights), np.asarray(system["kweights"])[order], rtol=1e-15)


def test_psp_correction_is_the_alpha_term():
    """DFTK's PspCorrection = N_el sum_j alpha_j / Omega (-0.294634 Ha, theory post),
    pinning the convention by measurement."""
    for system in SYSTEMS.values():
        expected = 8.0 * alpha_shift(system)
        assert abs(system["energies_hartree"]["PspCorrection"] - expected) < PSP_CORRECTION_ATOL
        assert abs(expected - (-0.294634)) < 1e-6


@pytest.mark.parametrize("term", ["Kinetic", "AtomicLocal", "AtomicNonlocal", "Hartree", "Xc", "Ewald"])
def test_energy_terms(case, term):
    system, _, _, result = case
    dftk = system["energies_hartree"]
    expected = dftk[term] + (dftk["PspCorrection"] if term == "AtomicLocal" else 0.0)
    assert abs(float(result.energies[term]) - expected) < DFTK_ENERGY_ATOL


def test_total_energy(case):
    system, _, _, result = case
    assert abs(float(result.total_energy) - system["total_energy_hartree"]) < DFTK_ENERGY_ATOL


def test_eigenvalues_with_the_alpha_shift(case):
    """Ours include sum_j alpha_j / Omega (Janak's theorem holds exactly); DFTK's do
    not. The shift is +0.036829 Ha (theory post), asserted at every k-point."""
    system, _, b, result = case
    assert abs(-alpha_shift(system) - 0.036829) < 1e-6
    ours = np.asarray(result.eigenvalues)[:, :N_BANDS]
    dftk = np.asarray(system["eigenvalues_hartree"])[k_order(system, b)]
    np.testing.assert_allclose(dftk, ours - alpha_shift(system), atol=DFTK_EIGENVALUE_ATOL, rtol=0)


def test_forces(case):
    system, s, b, _ = case
    np.testing.assert_allclose(
        forces(s, b, tol=TOL), np.asarray(system["forces_hartree_per_bohr"]), atol=DFTK_FORCE_ATOL, rtol=0
    )
