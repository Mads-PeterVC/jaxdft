"""Milestone 1b acceptance tests: Cohen-Bergstresser silicon against the paper
(validation ladder rung 2) and an independent NumPy implementation of the same model.
See docs/theory/01b_cohen_bergstresser.qmd. Reference data:
reference/cohen_bergstresser_1966/si_table1.json. Tolerances: tests/support/tolerances.py.
"""

import numpy as np
import pytest

from jaxdft import (
    Kinetic,
    LocalPotential,
    PlaneWaveBasis,
    Structure,
    assert_double_precision,
    band_structure,
    diamond,
    kpath,
    special_points_fcc,
)
from jaxdft.analysis.bands import band_minimum_on_segment
from jaxdft.analysis.symmetry import inversion_parities
from jaxdft.pseudo.cohen_bergstresser import silicon
from support import cohen_bergstresser as cbref
from support.free_electron import group_levels
from support.reference import cb1966_si_table1
from support.tolerances import (
    CB_TABLE_TOL_EV,
    EIGENVALUE_ATOL,
    INTERIOR_MARGIN,
    LEVEL_ATOL,
    SPLIT_MIN,
)

TEST_ECUT = 6.0  # Ha (theory post: all Table I splittings converged to ~1 meV)
HA_EV = 27.211386245981
SI = diamond(5.43, "Si", unit="Angstrom")
TERMS = (Kinetic(), LocalPotential({"Si": silicon()}))
POINTS = {k: np.asarray(v) for k, v in special_points_fcc().items()}
T = 2 * np.pi / cbref.A_CB


def cart_to_frac(c):
    """Fractional k for Cartesian k in units of 2 pi / a."""
    return np.linalg.solve(np.asarray(SI.lattice.reciprocal).T, T * np.asarray(c, dtype=float))


def eigenvalues_at(kfrac, ecut=TEST_ECUT):
    basis = PlaneWaveBasis.build(SI.lattice, np.atleast_2d(kfrac), ecut)
    return np.asarray(band_structure(SI, basis, n_bands=int(basis.n_pw[0]), terms=TERMS).eigenvalues[0])


def degeneracies(e, n):
    return [d for _, d in group_levels(e, LEVEL_ATOL)][:n]


@pytest.fixture(scope="module")
def levels():
    return {p: eigenvalues_at(POINTS[p]) for p in ("G", "X", "L")}


@pytest.fixture(scope="module")
def delta_minimum():
    """(t, energy) of band 5 (index 4) along Gamma -> X, via the library."""
    return band_minimum_on_segment(SI, POINTS["G"], POINTS["X"], band=4, ecut=TEST_ECUT, terms=TERMS)


# --- level structure at the high-symmetry points (identification of Table I levels) ---


def test_gamma_degeneracies(levels):
    assert degeneracies(levels["G"], 4) == [1, 3, 3, 1]


def test_l_degeneracies(levels):
    assert degeneracies(levels["L"], 5) == [1, 1, 2, 1, 2]


def test_x_levels_have_even_degeneracy(levels):
    assert all(d % 2 == 0 for d in degeneracies(levels["X"], 6))


@pytest.mark.parametrize("z", [0.1, 0.25, 0.4, 0.5])
def test_bands_paired_along_x_w(z):
    """Non-symmorphic operation + time reversal pair the bands on X-W (theory post)."""
    e = eigenvalues_at(cart_to_frac([0.0, 1.0, z]))[:8]
    np.testing.assert_allclose(e[1::2], e[0::2], atol=LEVEL_ATOL)


def test_generic_face_point_is_not_paired():
    """Negative control: the pairing is not a property of the whole square face."""
    e = eigenvalues_at(cart_to_frac([0.13, 1.0, 0.21]))[:8]
    assert np.min(np.abs(e[1::2] - e[0::2])) > SPLIT_MIN


def test_inversion_parities():
    """Gamma1(+) Gamma25'(+) Gamma15(-) Gamma2'(-); L2'(-) L1(+) L3'(-) L1(+) L3(+)."""
    expected = {"G": [(1, 1), (3, 1), (3, -1), (1, -1)], "L": [(1, -1), (1, 1), (2, -1), (1, 1), (2, 1)]}
    for p, want in expected.items():
        basis = PlaneWaveBasis.build(SI.lattice, POINTS[p][None], TEST_ECUT)
        bands = band_structure(SI, basis, n_bands=12, terms=TERMS)
        got = inversion_parities(bands, k_index=0, atol=LEVEL_ATOL)[: len(want)]
        assert [d for _, d, _ in got] == [d for d, _ in want]
        np.testing.assert_allclose([par for _, _, par in got], [par for _, par in want], atol=1e-8)


# --- the indirect gap: the Delta minimum, not the level at X ---------------------------


def test_delta_minimum_is_interior_and_below_x(levels, delta_minimum):
    t, e_min = delta_minimum
    e_x1 = levels["X"][4]
    assert 1e-3 < t < 1 - 1e-3
    assert e_x1 - e_min > INTERIOR_MARGIN


# --- Table I ----------------------------------------------------------------------------


def computed_splittings(levels, delta_minimum):
    G, X, L = levels["G"], levels["X"], levels["L"]
    vbm = G[3]
    gap_indirect_delta = delta_minimum[1] - vbm
    return {
        "Gamma25p-Gamma2p": (G[7] - vbm) * HA_EV,
        "Gamma25p-Gamma15": (G[4] - vbm) * HA_EV,
        "Gamma25p-L1": (L[4] - vbm) * HA_EV,
        "Gamma25p-X1(Delta1)": gap_indirect_delta * HA_EV,
        "L3p-L1": (L[4] - L[2]) * HA_EV,
        "L3p-L3": (L[5] - L[2]) * HA_EV,
        "X4-X1": (X[4] - X[2]) * HA_EV,
    }


@pytest.mark.parametrize("row", cb1966_si_table1()["splittings"], ids=lambda r: r["id"])
def test_table1_splittings(levels, delta_minimum, row):
    got = computed_splittings(levels, delta_minimum)[row["id"]]
    assert abs(got - row["calculated"]) <= CB_TABLE_TOL_EV[row["tolerance_class"]], f"{row['id']}: {got:.3f} eV"


# --- against the independent NumPy model -------------------------------------------------


@pytest.mark.parametrize("k", ["G", "X", "L", "generic"])
def test_matches_independent_numpy_model(k):
    kfrac = POINTS.get(k, np.array([0.11, 0.23, -0.31]))
    np.testing.assert_allclose(eigenvalues_at(kfrac, ecut=3.0), cbref.eigenvalues(kfrac, 3.0), atol=EIGENVALUE_ATOL)


# --- the band structure along the path -------------------------------------------------


@pytest.fixture(scope="module")
def path_bands():
    path = kpath(SI.lattice, special_points_fcc(), [["L", "G", "X", "U"], ["K", "G"]], n_points=80)
    basis = PlaneWaveBasis.build(SI.lattice, path.kpoints, TEST_ECUT)
    return path, band_structure(SI, basis, n_bands=8, terms=TERMS)


def test_gap_over_the_path(path_bands):
    """VBM at Gamma, a gap between bands 4 and 5 everywhere, CBM on Gamma-X (which the
    path covers, so the test is meaningful)."""
    path, bands = path_bands
    e = np.asarray(bands.eigenvalues)
    i_gamma = int(np.flatnonzero(np.all(np.abs(np.asarray(path.kpoints)) < 1e-14, axis=1))[0])
    assert e[:, 3].max() == pytest.approx(e[i_gamma, 3], abs=EIGENVALUE_ATOL)
    assert e[:, 4].min() > e[:, 3].max()
    d = np.asarray(path.distance)
    x_gamma, x_x = np.asarray(path.tick_positions)[1:3]
    i_cbm = int(np.argmin(e[:, 4]))
    assert x_gamma < d[i_cbm] <= x_x


def test_vmapped_path_matches_per_k(path_bands):
    path, bands = path_bands
    for i in [0, 17, 40, 79]:
        e = eigenvalues_at(np.asarray(path.kpoints[i]))[:8]
        np.testing.assert_allclose(np.asarray(bands.eigenvalues[i]), e, atol=EIGENVALUE_ATOL)


def test_path_results_are_double_precision(path_bands):
    assert_double_precision(path_bands[1])


# --- general structure factor: supercell folding -------------------------------------


def test_conventional_cell_folds_gamma_and_x():
    """The 8-atom conventional cell at Gamma contains the primitive cell's Gamma and its
    three X points (the cosets of the fcc reciprocal lattice in the simple-cubic one).
    Same cutoff, so the bases coincide and the spectra must agree exactly. Tests the
    general 1/Omega, per-atom structure factor with 8 atoms, not CB's two-atom form."""
    a = 5.43
    fcc = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    conv = Structure.from_cartesian(
        a * np.eye(3), np.concatenate([fcc + 0.125, fcc - 0.125]) * a, ["Si"] * 8, unit="Angstrom"
    )
    ecut = 2.9  # Ha; between simple-cubic shells (|G|^2 = 15 and 16 in (2 pi / a)^2)
    basis = PlaneWaveBasis.build(conv.lattice, np.zeros((1, 3)), ecut)
    e_conv = np.asarray(band_structure(conv, basis, n_bands=int(basis.n_pw[0]), terms=TERMS).eigenvalues[0])
    prim = np.concatenate([eigenvalues_at(cart_to_frac(c), ecut) for c in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]])
    assert len(prim) == len(e_conv)
    np.testing.assert_allclose(e_conv, np.sort(prim), atol=EIGENVALUE_ATOL)
