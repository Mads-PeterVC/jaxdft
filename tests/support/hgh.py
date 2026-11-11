"""Independent transcriptions of the HGH projector formulas (theory post 02b, 'The
separable nonlocal pseudopotential'), for tests. Written from the post, not from the
package, so that a transcription error in either shows up as a disagreement.
"""

import math

import numpy as np
from scipy import special

# (l, i) covered by the theory post's table (i is 1-based).
TABLE = ((0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2))


def real_space_projector(l, i, r_l, r):
    """p_i^l(r) of HGH 1998, normalised so that int p^2 r^2 dr = 1."""
    a = l + (4 * i - 1) / 2
    return math.sqrt(2) * r ** (l + 2 * (i - 1)) * np.exp(-(r**2) / (2 * r_l**2)) / (r_l**a * math.sqrt(math.gamma(a)))


def quadrature_projector(l, i, r_l, q, n=600):
    """p_hat_i^l(q) = 4 pi int p(r) j_l(q r) r^2 dr by Gauss-Legendre quadrature."""
    x, w = np.polynomial.legendre.leggauss(n)
    R = 16.0 * r_l
    r = 0.5 * R * (x + 1)
    w = 0.5 * R * w
    return 4 * math.pi * np.sum(w * real_space_projector(l, i, r_l, r) * special.spherical_jn(l, q * r) * r**2)


def reduced_projector(l, i, r_l, q2):
    """p_hat_i^l(q) / q^l from the theory post's table, as a function of q^2."""
    y = np.asarray(q2, dtype=float) * r_l**2
    pref = math.pi**1.25 * np.exp(-y / 2)
    table = {
        (0, 1): 4 * math.sqrt(2) * r_l**1.5 * np.ones_like(y),
        (0, 2): 8 * math.sqrt(2 / 15) * r_l**1.5 * (3 - y),
        (0, 3): 16 / 3 * math.sqrt(2 / 105) * r_l**1.5 * (15 - 10 * y + y**2),
        (1, 1): 8 / math.sqrt(3) * r_l**2.5 * np.ones_like(y),
        (1, 2): 16 / math.sqrt(105) * r_l**2.5 * (5 - y),
        (1, 3): 32 / (3 * math.sqrt(1155)) * r_l**2.5 * (35 - 14 * y + y**2),
        (2, 1): 8 * math.sqrt(2 / 15) * r_l**3.5 * np.ones_like(y),
        (2, 2): 16 / 3 * math.sqrt(2 / 105) * r_l**3.5 * (7 - y),
    }
    return pref * table[(l, i)]


def legendre_nonlocal_matrix(structure, pseudos, kpoint, millers) -> np.ndarray:
    """<k+G|V_nl|k+G'> from the Legendre formula of the theory post, in NumPy:

        (1/Omega) sum_a lambda_a e^{-i(G-G').tau_a} sum_l sum_ij
            p_i(q) h_ij p_j(q') (2l+1)/(4 pi) P_l(q^.q'^),

    with p_i(q) p_j(q') P_l written through the reduced projectors and the solid
    form of P_l, so q = 0 needs no special case."""
    A = np.asarray(structure.lattice.vectors)
    B = 2 * np.pi * np.linalg.inv(A).T
    omega = abs(np.linalg.det(A))
    m = np.asarray(millers, dtype=float)
    q = (np.asarray(kpoint)[None, :] + m) @ B
    q2 = np.sum(q**2, axis=1)
    dot = q @ q.T
    angular = {
        0: np.ones_like(dot),
        1: dot,
        2: 0.5 * (3 * dot**2 - np.outer(q2, q2)),
    }
    f = np.asarray(structure.positions)
    w = np.asarray(structure.weights)
    V = np.zeros((len(m), len(m)), dtype=complex)
    for a, species in enumerate(structure.species):
        phase = np.exp(-2j * np.pi * (m @ f[a]))
        S = np.outer(phase, phase.conj())
        for ch in getattr(pseudos[species], "channels", ()):
            h = np.asarray(ch.h, dtype=float)
            P = np.stack([reduced_projector(ch.l, i + 1, ch.r, q2) for i in range(h.shape[0])], axis=1)
            radial = P @ h @ P.T
            V += w[a] * S * (2 * ch.l + 1) / (4 * np.pi) * angular[ch.l] * radial
    return V / omega
