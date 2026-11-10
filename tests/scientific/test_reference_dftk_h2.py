"""jaxdft against DFTK on the same discrete problem (theory post 02a, 'Against
references' and 'Tolerances'; decisions 3, 5 and 11).

Both codes use the same HGH file, Slater + PW92, the same box, cutoff and 30^3 grid,
Gamma only and no symmetrisation, at a symmetric and a distorted, off-centre geometry.
The comparison is term by term. DFTK reports the G = 0 part of the local
pseudopotential separately as PspCorrection and keeps it out of the Hamiltonian, so
its eigenvalues are higher than ours by -sum_j alpha_j / Omega, asserted explicitly.
"""

import hashlib

import jax
import numpy as np
import pytest

from jaxdft import Lattice, PlaneWaveBasis, SCFOptions, Structure, energy, scf
from jaxdft.pseudo.gth import hgh_lda_path
from support.h2 import h2_terms, hydrogen
from support.reference import dftk_h2
from support.tolerances import (
    DFTK_EIGENVALUE_ATOL,
    DFTK_ENERGY_ATOL,
    DFTK_FORCE_ATOL,
    PSP_CORRECTION_ATOL,
)

pytestmark = pytest.mark.slow

REF = dftk_h2()
SYSTEMS = {s["name"]: s for s in REF["systems"]}
TOL = 1e-10  # our SCF, tighter than the acceptance default


def setup(system):
    lattice = Lattice(np.asarray(system["lattice_bohr"]))
    s = Structure(lattice, np.asarray(system["positions_fractional"]), ["H", "H"])
    b = PlaneWaveBasis.build(
        lattice, np.zeros((1, 3)), system["ecut_hartree"], fft_shape=tuple(system["fft_size"])
    )
    return s, b


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def case(request):
    system = SYSTEMS[request.param]
    s, b = setup(system)
    return system, s, b, scf(s, b, h2_terms(), SCFOptions(tol=TOL))


def alpha_shift(system):
    """sum_j alpha_j / Omega (Ha): the G = 0 local potential DFTK leaves out of H."""
    omega = abs(np.linalg.det(np.asarray(system["lattice_bohr"])))
    return len(system["positions_fractional"]) * hydrogen().alpha / omega


def test_reference_identifies_the_same_problem():
    assert REF["functional"] == ["lda_x", "lda_c_pw"]
    assert REF["pseudopotential"]["sha256"] == hashlib.sha256(hgh_lda_path("H").read_bytes()).hexdigest()
    assert set(SYSTEMS) == {"symmetric", "distorted"}
    for system in SYSTEMS.values():
        assert system["occupations"] == [2.0, 0.0]
        assert system["energies_hartree"]["AtomicNonlocal"] == 0.0


def test_same_basis(case):
    system, _, b, _ = case
    assert int(b.n_pw[0]) == system["n_plane_waves"]
    assert tuple(b.fft_shape) == tuple(system["fft_size"])


def test_psp_correction_is_the_alpha_term():
    """DFTK's PspCorrection = N_el sum_j alpha_j / Omega (-5.19e-6 Ha, theory post),
    pinning the convention by measurement."""
    for system in SYSTEMS.values():
        expected = 2.0 * alpha_shift(system)
        assert abs(system["energies_hartree"]["PspCorrection"] - expected) < PSP_CORRECTION_ATOL
        assert abs(expected - (-5.19e-6)) < 0.01e-6


@pytest.mark.parametrize("term", ["Kinetic", "AtomicLocal", "Hartree", "Xc", "Ewald"])
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
    not. The shift (+2.6e-6 Ha) is only 2.6 times the tolerance, so it is explicit."""
    system, _, _, result = case
    ours = np.asarray(result.eigenvalues[0, :2])
    dftk = np.asarray(system["eigenvalues_hartree"])
    np.testing.assert_allclose(dftk, ours - alpha_shift(system), atol=DFTK_EIGENVALUE_ATOL, rtol=0)


def test_forces(case):
    system, s, b, _ = case
    g = jax.grad(lambda st: energy(st, b, h2_terms(), SCFOptions(tol=TOL)))(s)
    A = np.asarray(s.lattice.vectors)
    forces = -np.asarray(g.positions) @ np.linalg.inv(A).T
    np.testing.assert_allclose(forces, np.asarray(system["forces_hartree_per_bohr"]), atol=DFTK_FORCE_ATOL, rtol=0)
