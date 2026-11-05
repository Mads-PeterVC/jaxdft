"""Independent NumPy reference for the empty lattice.

Deliberately shares no code with ``jaxdft``. It uses plain NumPy and a brute-force
integer search over a box twice the proven bound (theory post, 'Enumerating the
basis'), so that it stays an independent check of the package.
"""

import itertools
from fractions import Fraction

import numpy as np

# Si lattice constant in Bohr (5.431 Angstrom). The empty lattice is scale-free,
# so this only sets E0.
A_SI = 10.26


def fcc_vectors(a: float) -> np.ndarray:
    """Primitive fcc vectors as rows (theory post convention)."""
    return 0.5 * a * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])


def reciprocal(A: np.ndarray) -> np.ndarray:
    return 2.0 * np.pi * np.linalg.inv(A).T


def e0(a: float) -> float:
    """Natural energy unit 1/2 (2 pi / a)^2 in Hartree."""
    return 0.5 * (2.0 * np.pi / a) ** 2


# Fractional coordinates (k = f B) of the fcc high-symmetry points.
SPECIAL_POINTS_FCC = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.5, 0.0, 0.5),
    "L": (0.5, 0.5, 0.5),
    "W": (0.25, 0.5, 0.75),
    "K": (0.375, 0.375, 0.75),
    "U": (0.25, 0.625, 0.625),
}

# Cartesian coordinates in units of 2 pi / a (theory post table).
SPECIAL_POINTS_FCC_CART = {
    "G": (0.0, 0.0, 0.0),
    "X": (0.0, 1.0, 0.0),
    "L": (0.5, 0.5, 0.5),
    "W": (1.0, 0.5, 0.0),
    "K": (0.75, 0.75, 0.0),
    "U": (1.0, 0.25, 0.25),
}

F = Fraction
# Hand-derived lowest levels in units of E0, as (level, degeneracy). From the theory post.
DEGENERACY_TABLE = {
    "G": [(F(0), 1), (F(3), 8), (F(4), 6), (F(8), 12), (F(11), 24), (F(12), 8)],
    "X": [(F(1), 2), (F(2), 4), (F(5), 8), (F(6), 8), (F(9), 10), (F(10), 8)],
    "L": [(F(3, 4), 2), (F(11, 4), 6), (F(19, 4), 6), (F(27, 4), 8), (F(35, 4), 12), (F(43, 4), 6)],
    "W": [(F(5, 4), 4), (F(13, 4), 4), (F(21, 4), 8), (F(29, 4), 12), (F(37, 4), 4), (F(45, 4), 12)],
    "K": [(F(9, 8), 3), (F(17, 8), 2), (F(25, 8), 1), (F(33, 8), 4), (F(41, 8), 2), (F(49, 8), 8)],
    "U": [(F(9, 8), 3), (F(17, 8), 2), (F(25, 8), 1), (F(33, 8), 4), (F(41, 8), 2), (F(49, 8), 8)],
}


def miller_bound(A: np.ndarray, kfrac, gmax: float) -> np.ndarray:
    """Proven bound |m_i| <= (Gmax + |k|) |a_i| / 2 pi."""
    k = np.asarray(kfrac) @ reciprocal(A)
    return np.ceil((gmax + np.linalg.norm(k)) * np.linalg.norm(A, axis=1) / (2 * np.pi)).astype(int)


def _box(n: np.ndarray) -> np.ndarray:
    return np.array(list(itertools.product(*[range(-int(x), int(x) + 1) for x in n])), dtype=np.int64)


def basis_set(A: np.ndarray, kfrac, ecut: float) -> set[tuple[int, int, int]]:
    """Miller indices with 1/2|k+G|^2 <= ecut, from a box twice the proven bound.

    Asserts that nothing between the proven bound and the doubled box qualifies,
    which checks the bound itself.
    """
    gmax = np.sqrt(2 * ecut)
    bound = miller_bound(A, kfrac, gmax)
    m = _box(2 * bound)
    k = np.asarray(kfrac) @ reciprocal(A)
    inside = 0.5 * np.sum((k + m @ reciprocal(A)) ** 2, axis=1) <= ecut
    assert np.all(np.abs(m[inside]) <= bound), "proven Miller bound violated"
    return {tuple(int(x) for x in row) for row in m[inside]}


def free_electron_levels(A: np.ndarray, kfrac, emax: float) -> np.ndarray:
    """All 1/2|k+G|^2 <= emax, sorted, with multiplicity (Hartree)."""
    gmax = np.sqrt(2 * emax)
    m = _box(2 * miller_bound(A, kfrac, gmax))
    k = np.asarray(kfrac) @ reciprocal(A)
    e = 0.5 * np.sum((k + m @ reciprocal(A)) ** 2, axis=1)
    return np.sort(e[e <= emax])


def lowest_free_electron_levels(A: np.ndarray, kfrac, n: int) -> np.ndarray:
    """The n lowest folded free-electron eigenvalues at k (Hartree)."""
    emax = 1.0
    while True:
        e = free_electron_levels(A, kfrac, emax)
        if len(e) >= n:
            return e[:n]
        emax *= 2.0


def group_levels(eigenvalues, atol: float) -> list[tuple[float, int]]:
    """Group sorted eigenvalues into (mean level, degeneracy) with gap > atol."""
    e = np.sort(np.asarray(eigenvalues))
    groups: list[list[float]] = [[e[0]]]
    for x in e[1:]:
        if x - groups[-1][-1] <= atol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(float(np.mean(g)), len(g)) for g in groups]


def ecut_between_shells(A: np.ndarray, kfracs, target: float) -> float:
    """A cutoff near ``target`` (Ha) placed mid-gap between shells at every k.

    Considers the union of shell energies 1/2|k+G|^2 over all k-points, and returns the
    midpoint of the gap containing ``target`` (or the nearest gap above it).
    """
    kfracs = np.atleast_2d(np.asarray(kfracs, dtype=float))
    levels = np.unique(np.concatenate([free_electron_levels(A, k, 2 * target + 1) for k in kfracs]))
    above = levels[levels > target]
    below = levels[levels <= target]
    lo = below[-1] if len(below) else 0.0
    hi = above[0]
    ecut = 0.5 * (lo + hi)
    assert min(ecut - lo, hi - ecut) >= 1e-8 * ecut or len(below) == 0
    return float(ecut)


def oh_operations() -> list[np.ndarray]:
    """The 48 signed permutation matrices of O_h (Cartesian, cubic frame)."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            S = np.zeros((3, 3))
            for i, p in enumerate(perm):
                S[i, p] = signs[i]
            ops.append(S)
    return ops
