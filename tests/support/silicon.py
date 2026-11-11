"""Shared silicon setups for Milestone 2b (theory post 02b, 'Test size',
'Decisions from review'; engineering log, '2026-11-11: Milestone 2b public API').

TEST size: a = 10.263 Bohr (the experimental 5.431 Angstrom), Ecut = 12 Ha (544 plane
waves at the largest k, a 24^3 grid, so N is even) and the Gamma-centred 2^3 k-grid.
Converged runs are cached for the whole session, since several modules need them.

The space-group helpers are test geometry, not package code: the package does not
detect symmetry (engineering log: the caller states ``fft_multiple``).
"""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np

from jaxdft import (
    PlaneWaveBasis,
    SCFOptions,
    Structure,
    diamond,
    energy,
    lda_terms,
    monkhorst_pack,
    scf,
)
from jaxdft.pseudo.gth import GTH, hgh_lda

A_SI = 10.263  # Bohr
ECUT_TEST = 12.0  # Ha
KGRID_TEST = (2, 2, 2)
N_TEST = 24  # the rule grid at the test size
HARTREE_IN_EV = 27.211386245981

# The displaced geometry of the force test (theory post, 'Forces'): atom 0 moved by
# this fractional vector (0.11 Bohr). No symmetry is left, and no inversion centre on
# the grid. Its tests run on the 3^3 grid: on 2^3 the gap over the grid closes to
# 0.12 mHa, below min_gap, while on 3^3 it is 10.3 mHa (decided 2026-11-11 with the
# project lead; the theory post's force measurements were made on 3^3).
DISPLACEMENT = np.array([[0.012, -0.005, 0.008], [0.0, 0.0, 0.0]])
KGRID_DISPLACED = (3, 3, 3)


@functools.cache
def silicon():
    return hgh_lda("Si")


@functools.cache
def local_only_silicon() -> GTH:
    """The local part of HGH silicon with the projectors dropped: a semimetal on the
    2^3 grid at 12 Ha, the positive control of the gap guard."""
    si = silicon()
    return GTH(si.zion, si.rloc, si.c)


def si_terms(xc=True):
    terms = lda_terms({"Si": silicon()})
    if xc:
        return terms
    return tuple(t for t in terms if type(t).__name__ != "Xc")


def si(a=A_SI, positions=None, weights=None) -> Structure:
    s = diamond(a, "Si", unit="Bohr")
    positions = s.positions if positions is None else jnp.asarray(positions, dtype=float)
    w = None if weights is None else jnp.asarray(weights, dtype=float)
    return Structure(s.lattice, positions, s.species, w)


def displaced_si(weights=None) -> Structure:
    s = si()
    return si(positions=np.asarray(s.positions) + DISPLACEMENT, weights=weights)


def with_weights(structure, weights) -> Structure:
    return Structure(structure.lattice, structure.positions, structure.species, jnp.asarray(weights, dtype=float))


def with_positions(structure, positions) -> Structure:
    return Structure(structure.lattice, jnp.asarray(positions, dtype=float), structure.species, structure.weights)


def basis_for(structure, ecut=ECUT_TEST, kshape=KGRID_TEST, shift=(0, 0, 0), **build):
    kpts, w = monkhorst_pack(kshape, shift=shift)
    return PlaneWaveBasis.build(structure.lattice, kpts, ecut, kweights=w, **build)


def run(structure, basis, terms=None, tol=None, **options):
    from support.tolerances import SCF_TOL

    terms = si_terms() if terms is None else terms
    return scf(structure, basis, terms, SCFOptions(tol=tol or SCF_TOL, **options))


@functools.cache
def converged_test_size():
    s = si()
    b = basis_for(s)
    return s, b, run(s, b)


@functools.cache
def converged_displaced():
    s = displaced_si()
    b = basis_for(s, kshape=KGRID_DISPLACED)
    return s, b, run(s, b)


def grid_gap(eigenvalues, n_occupied=4) -> float:
    """CBM - VBM over all k-points (Ha): band n_occupied against band n_occupied - 1."""
    e = np.asarray(eigenvalues)
    return float(e[:, n_occupied].min() - e[:, n_occupied - 1].max())


# --- derivatives ---------------------------------------------------------------------


def position_gradient(structure, basis, terms=None, tol=None, side=-1, **options) -> np.ndarray:
    """dE/df (Ha per unit fractional coordinate), (n_atoms, 3). ``options`` go to
    SCFOptions."""
    from support.tolerances import SCF_TOL

    terms = si_terms() if terms is None else terms
    opts = SCFOptions(tol=tol or SCF_TOL, **options)
    g = jax.grad(lambda st: energy(st, basis, terms, opts, side=side))(structure)
    return np.asarray(g.positions)


def forces(structure, basis, terms=None, tol=None) -> np.ndarray:
    """-dE/dtau in Ha/Bohr, (n_atoms, 3): dE/dtau = (dE/df) A^-T."""
    A = np.asarray(structure.lattice.vectors)
    return -position_gradient(structure, basis, terms, tol) @ np.linalg.inv(A).T


# --- the diamond space group (test geometry) ----------------------------------------


def cubic_operations() -> list[np.ndarray]:
    """The 48 orthogonal matrices of the cubic group O_h (Cartesian)."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                m[row, col] = sign
            ops.append(m)
    return ops


def space_group(structure, atol=1e-9) -> list[tuple[np.ndarray, np.ndarray]]:
    """All (R, t) with fractional rows f -> f R + t (mod 1) mapping the atoms onto
    atoms of the same species, R from the cubic group. R is an integer matrix."""
    A = np.asarray(structure.lattice.vectors)
    f = np.asarray(structure.positions)
    species = structure.species
    out = []
    for C in cubic_operations():
        R = A @ C.T @ np.linalg.inv(A)
        Ri = np.rint(R)
        if np.abs(R - Ri).max() > atol:
            continue
        mapped = f @ Ri
        for b in range(len(f)):
            if species[b] != species[0]:
                continue
            t = np.mod(f[b] - mapped[0], 1.0)
            image = np.mod(mapped + t, 1.0)
            ok = True
            for a in range(len(f)):
                d = np.mod(image[a] - f + 0.5, 1.0) - 0.5
                match = [c for c in range(len(f)) if np.abs(d[c]).max() < atol and species[c] == species[a]]
                if not match:
                    ok = False
                    break
            if ok:
                t = np.where(np.abs(t - 1.0) < atol, 0.0, t)
                if not any(np.array_equal(Ri, R2) and np.allclose(t, t2, atol=atol) for R2, t2 in out):
                    out.append((Ri.astype(int), t))
    return out


def maps_grid(op, shape, atol=1e-9) -> bool:
    """Whether f -> f R + t maps the FFT grid j/N onto itself (t N integral)."""
    _R, t = op
    tn = np.asarray(t) * np.asarray(shape)
    return bool(np.abs(tn - np.rint(tn)).max() < atol)


def transform_grid(values, op) -> np.ndarray:
    """values evaluated at the image points: out[j] = values[(j R + t N) mod N]. A
    density invariant under the operation satisfies out == values."""
    R, t = op
    values = np.asarray(values)
    shape = np.asarray(values.shape)
    j = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1).reshape(-1, 3)
    # Fractional point j/N maps to (j/N) R + t, i.e. grid index j R' with R' acting on
    # j/N: index_i = sum_k j_k N_i R_ki / N_k + t_i N_i.
    image = (j / shape) @ R * shape + np.asarray(t) * shape
    idx = np.mod(np.rint(image).astype(int), shape)
    return values[idx[:, 0], idx[:, 1], idx[:, 2]].reshape(values.shape)
