"""Special points and k-paths (theory post: 'High-symmetry points', 'Band-structure path')."""

import numpy as np
import pytest

from jaxdft import KPath, Lattice, assert_double_precision, kpath, special_points_fcc, fcc_lattice
from support.free_electron import SPECIAL_POINTS_FCC

SI_PATH = [["L", "G", "X", "U"], ["K", "G"]]


def test_special_points_fcc():
    points = special_points_fcc()
    assert set(points) == set(SPECIAL_POINTS_FCC)
    for label, f in SPECIAL_POINTS_FCC.items():
        np.testing.assert_allclose(points[label], f, atol=0)


@pytest.fixture
def path():
    return kpath(fcc_lattice(10.26, unit="Bohr"), special_points_fcc(), SI_PATH, n_points=101)


def _index_of(path, f):
    hits = np.flatnonzero(np.all(np.abs(np.asarray(path.kpoints) - np.asarray(f)) < 1e-12, axis=1))
    assert len(hits) >= 1
    return hits


def test_kpath_shape_and_type(path):
    assert isinstance(path, KPath)
    assert path.kpoints.shape == (101, 3)
    assert path.distance.shape == (101,)
    assert_double_precision((path.kpoints, path.distance))


def test_kpath_visits_corners_in_order(path):
    pts = special_points_fcc()
    np.testing.assert_allclose(path.kpoints[0], pts["L"], atol=1e-14)
    np.testing.assert_allclose(path.kpoints[-1], pts["G"], atol=1e-14)
    i_X = _index_of(path, pts["X"])[0]
    i_U = _index_of(path, pts["U"])[0]
    i_K = _index_of(path, pts["K"])[0]
    assert i_X < i_U and i_K == i_U + 1


def test_kpath_distance(path):
    lat = fcc_lattice(10.26, unit="Bohr")
    pts = special_points_fcc()
    B = np.asarray(lat.reciprocal)

    def seg(a, b):
        return np.linalg.norm((np.asarray(pts[b]) - np.asarray(pts[a])) @ B)

    d = np.asarray(path.distance)
    assert d[0] == 0
    assert np.all(np.diff(d) >= 0)
    total = seg("L", "G") + seg("G", "X") + seg("X", "U") + seg("K", "G")
    np.testing.assert_allclose(d[-1], total, rtol=1e-12)
    # The U|K break does not advance the distance.
    i_U = _index_of(path, pts["U"])[0]
    assert d[i_U + 1] == d[i_U]
    np.testing.assert_allclose(
        path.tick_positions,
        np.cumsum([0, seg("L", "G"), seg("G", "X"), seg("X", "U"), seg("K", "G")]),
        rtol=1e-12,
    )
    assert list(path.tick_labels) == ["L", "Γ", "X", "U|K", "Γ"]


def test_kpath_points_spread_by_length(path):
    """Points are spaced uniformly within a segment and similarly across segments."""
    d = np.asarray(path.distance)
    steps = np.diff(d)
    steps = steps[steps > 0]
    mean = steps.mean()
    assert np.all(steps > 0.5 * mean) and np.all(steps < 2.0 * mean)
