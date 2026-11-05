"""Independent NumPy implementation of the Cohen-Bergstresser Si model.

Shares no code with the package. It follows the theory post's equations directly,
in the paper's own form: V(G) = cos(G.tau) V_S(|G|^2), atoms at +-(a/8)(1,1,1), V_S
in Ry on shells 3, 8, 11 in units of (2 pi/a)^2. Dense matrices; small cutoffs only.
"""

import itertools

import numpy as np

from support.free_electron import fcc_vectors, reciprocal

BOHR_IN_ANGSTROM = 0.529177210544  # CODATA 2022 (theory post, 'Units at the boundary')
HARTREE_IN_EV = 27.211386245981  # CODATA 2022
A_CB_ANGSTROM = 5.43
A_CB = A_CB_ANGSTROM / BOHR_IN_ANGSTROM  # Bohr
V_S_RY = {3: -0.21, 8: 0.04, 11: 0.08}  # Table II, Si
TAU_FRAC = np.array([1.0, 1.0, 1.0]) / 8.0


def cubic_indices(miller: np.ndarray, A: np.ndarray, a: float) -> np.ndarray:
    """(h, k, l) with G = (2 pi / a)(h, k, l), as floats."""
    return miller @ reciprocal(A) / (2 * np.pi / a)


def v_tilde(miller_diff: np.ndarray, a: float = A_CB) -> np.ndarray:
    """V(G) in Ha for Miller-index differences (..., 3), symmetric-origin diamond Si."""
    A = fcc_vectors(a)
    hkl = cubic_indices(miller_diff, A, a)
    g2 = np.sum(hkl**2, axis=-1)
    vs = np.zeros(g2.shape)
    for shell, v in V_S_RY.items():
        vs = np.where(np.abs(g2 - shell) < 1e-9, 0.5 * v, vs)
    return np.cos(2 * np.pi * miller_diff @ TAU_FRAC) * vs


def basis(kfrac, ecut: float, a: float = A_CB) -> np.ndarray:
    A = fcc_vectors(a)
    B = reciprocal(A)
    n = int(np.ceil((np.sqrt(2 * ecut) + np.linalg.norm(np.asarray(kfrac) @ B)) * np.linalg.norm(A, axis=1).max() / (2 * np.pi)))
    m = np.array(list(itertools.product(range(-n, n + 1), repeat=3)))
    return m[0.5 * np.sum(((np.asarray(kfrac) + m) @ B) ** 2, axis=1) <= ecut]


def hamiltonian(kfrac, ecut: float, a: float = A_CB):
    """(Miller indices, dense H in Ha) at fractional k."""
    B = reciprocal(fcc_vectors(a))
    m = basis(kfrac, ecut, a)
    kin = 0.5 * np.sum(((np.asarray(kfrac) + m) @ B) ** 2, axis=1)
    return m, np.diag(kin) + v_tilde(m[:, None, :] - m[None, :, :], a)


def eigenvalues(kfrac, ecut: float, a: float = A_CB) -> np.ndarray:
    return np.linalg.eigvalsh(hamiltonian(kfrac, ecut, a)[1])


def table1_splittings(ecut: float, delta_samples: int = 201) -> dict[str, float]:
    """The seven Table I splittings (eV) of the model, identified by degeneracy/order."""
    G = eigenvalues([0, 0, 0], ecut)
    X = eigenvalues([0.5, 0.0, 0.5], ecut)
    L = eigenvalues([0.5, 0.5, 0.5], ecut)
    vbm = G[3]
    xs = np.linspace(0.0, 1.0, delta_samples)
    cb = [eigenvalues(x * np.array([0.5, 0.0, 0.5]), ecut)[4] for x in xs]
    i = int(np.argmin(cb))
    lo, hi = xs[max(i - 1, 0)], xs[min(i + 1, len(xs) - 1)]
    for _ in range(40):  # golden-section refinement on the bracket
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        f1, f2 = (eigenvalues(x * np.array([0.5, 0.0, 0.5]), ecut)[4] for x in (m1, m2))
        lo, hi = (lo, m2) if f1 < f2 else (m1, hi)
    cbm = eigenvalues(0.5 * (lo + hi) * np.array([0.5, 0.0, 0.5]), ecut)[4]
    ev = HARTREE_IN_EV
    return {
        "Gamma25p-Gamma2p": (G[7] - vbm) * ev,
        "Gamma25p-Gamma15": (G[4] - vbm) * ev,
        "Gamma25p-L1": (L[4] - vbm) * ev,
        "Gamma25p-X1(Delta1)": (cbm - vbm) * ev,
        "L3p-L1": (L[4] - L[2]) * ev,
        "L3p-L3": (L[5] - L[2]) * ev,
        "X4-X1": (X[4] - X[2]) * ev,
    }
