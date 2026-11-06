"""Shared H2-in-a-box setups for Milestone 2a (theory post 02a, 'Decisions from review').

TEST size (L = 10 Bohr, Ecut = 10 Ha, 1503 plane waves, 30^3 grid) is the acceptance
size. SMALL size (L = 8 Bohr, Ecut = 6 Ha) is for invariants that hold at any size.
Converged runs are cached for the whole session, since several test modules need the
same SCF.
"""

import functools

import jax.numpy as jnp
import numpy as np

from jaxdft import PlaneWaveBasis, SCFOptions, Structure, dimer_in_box, lda_terms, scf
from jaxdft.pseudo.gth import hgh_lda

L_TEST, ECUT_TEST, D_TEST = 10.0, 10.0, 1.45
L_SMALL, ECUT_SMALL = 8.0, 6.0
GAMMA = np.zeros((1, 3))


@functools.cache
def hydrogen():
    return hgh_lda("H")


def h2_terms():
    return lda_terms({"H": hydrogen()})


def h2(box=L_TEST, d=D_TEST, axis=2, weights=None) -> Structure:
    s = dimer_in_box("H", d, box, axis=axis)
    if weights is not None:
        s = Structure(s.lattice, s.positions, s.species, jnp.asarray(weights, dtype=float))
    return s


def basis_for(structure, ecut=ECUT_TEST, fft_shape=None, allow_aliasing=False):
    return PlaneWaveBasis.build(
        structure.lattice, GAMMA, ecut, fft_shape=fft_shape, allow_aliasing=allow_aliasing
    )


def with_positions(structure, cartesian) -> Structure:
    """Same lattice, species and weights, new Cartesian positions (Bohr)."""
    A = np.asarray(structure.lattice.vectors)
    f = np.asarray(cartesian, dtype=float) @ np.linalg.inv(A)
    return Structure(structure.lattice, f, structure.species, structure.weights)


def translated(structure, shift) -> Structure:
    r = np.asarray(structure.cartesian_positions)
    return with_positions(structure, r + np.asarray(shift, dtype=float))


def displaced(structure, atom, vector) -> Structure:
    r = np.array(structure.cartesian_positions)
    r[atom] += np.asarray(vector, dtype=float)
    return with_positions(structure, r)


def distorted_off_centre(box=L_TEST, d=1.2) -> Structure:
    """Bond tilted off every axis, centre off every grid point: no symmetry left."""
    s = h2(box, d)
    u = np.array([0.2, 0.1, 1.0]) / np.linalg.norm([0.2, 0.1, 1.0])
    c = np.full(3, box / 2) + np.array([0.13, -0.07, 0.21])
    return with_positions(s, [c - d / 2 * u, c + d / 2 * u])


def run(structure, basis, tol=None, **options):
    from support.tolerances import SCF_TOL

    return scf(structure, basis, h2_terms(), SCFOptions(tol=tol or SCF_TOL, **options))


@functools.cache
def converged_test_size():
    s = h2()
    b = basis_for(s)
    return s, b, run(s, b)


@functools.cache
def converged_small():
    s = h2(L_SMALL)
    b = basis_for(s, ECUT_SMALL)
    return s, b, run(s, b)
