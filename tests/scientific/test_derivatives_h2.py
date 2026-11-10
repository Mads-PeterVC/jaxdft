"""Forces, weight derivatives and cell gradients from jax.grad(energy) (theory post 02a,
'Forces and other derivatives', 'Per-atom weights'; decisions 9 and 11).

energy() stops the gradient on the converged orbitals, so the derivative is the exact
derivative of the discrete energy; finite differences of converged energies must agree
to the truncation and SCF errors derived in support/tolerances.py.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Lattice, SCFOptions, Structure, energy
from support.h2 import (
    ECUT_SMALL,
    L_SMALL,
    basis_for,
    converged_test_size,
    displaced,
    distorted_off_centre,
    h2,
    h2_terms,
    run,
)
from support.tolerances import (
    FD_STEP,
    FORCE_ATOL,
    NET_FORCE_MAX,
    SCF_TOL,
    SCF_TOL_TIGHT,
    WEIGHT_ATOL,
    WEIGHT_GAP_ATOL,
    WEIGHT_STEP,
)

pytestmark = pytest.mark.slow


def forces(structure, basis, tol=SCF_TOL):
    """-dE/dtau in Ha/Bohr, (n_atoms, 3), from the gradient with respect to fractional
    positions: dE/dtau = (dE/df) A^-T."""
    g = jax.grad(lambda s: energy(s, basis, h2_terms(), SCFOptions(tol=tol)))(structure)
    A = np.asarray(structure.lattice.vectors)
    return -np.asarray(g.positions) @ np.linalg.inv(A).T


def fd_force(structure, basis, atom, axis, h=FD_STEP, tol=SCF_TOL):
    e = []
    for sign in (+1, -1):
        v = np.zeros(3)
        v[axis] = sign * h
        e.append(float(run(displaced(structure, atom, v), basis, tol=tol).total_energy))
    return -(e[0] - e[1]) / (2 * h)


def test_forces_symmetric_molecule():
    s, b, _ = converged_test_size()
    F = forces(s, b)
    np.testing.assert_allclose(F[1, 2], fd_force(s, b, 1, 2), atol=FORCE_ATOL)
    np.testing.assert_allclose(F[:, :2], 0.0, atol=1e-10)  # by symmetry
    np.testing.assert_allclose(F[0], -F[1], atol=1e-10)


def test_forces_distorted_off_centre_all_components():
    s = distorted_off_centre()
    b = basis_for(s)
    F = forces(s, b)
    for axis in range(3):
        np.testing.assert_allclose(F[0, axis], fd_force(s, b, 0, axis), atol=FORCE_ATOL)


def test_net_force_off_grid_is_eggbox_noise():
    """Only XC quadrature breaks translation invariance (review: 2.5e-11 Ha/Bohr)."""
    s = distorted_off_centre()
    F = forces(s, basis_for(s), tol=1e-10)
    assert np.linalg.norm(F.sum(axis=0)) < NET_FORCE_MAX


@pytest.fixture(scope="module")
def weight_setup():
    s = h2()
    b = basis_for(s)
    return s, b, run(s, b, tol=SCF_TOL_TIGHT)


def weight_grad(s, b, side):
    opts = SCFOptions(tol=SCF_TOL_TIGHT)
    return np.asarray(jax.grad(lambda st: energy(st, b, h2_terms(), opts, side=side))(s).weights)


def one_sided_fd(s, b, e0, side):
    """Second-order one-sided stencil (-3 E0 + 4 E(d) - E(2d)) / (2 d) in the direction
    `side`, from separate SCF runs at lambda_0 = 1 + side d, 1 + 2 side d."""
    d = WEIGHT_STEP[side]
    e = []
    for k in (1, 2):
        w = jnp.asarray([1.0 + side * k * d, 1.0])
        e.append(float(run(Structure(s.lattice, s.positions, s.species, w), b, tol=SCF_TOL_TIGHT).total_energy))
    return side * (-3 * e0 + 4 * e[0] - e[1]) / (2 * d)


@pytest.mark.parametrize("side", [-1, +1], ids=["homo-side", "lumo-side"])
def test_one_sided_weight_derivative(weight_setup, side):
    s, b, res = weight_setup
    g = weight_grad(s, b, side)
    np.testing.assert_allclose(g[0], g[1], atol=1e-10)  # identical atoms
    np.testing.assert_allclose(g[0], one_sided_fd(s, b, float(res.total_energy), side), atol=WEIGHT_ATOL)


def test_weight_derivative_sides_differ_by_z_times_gap(weight_setup):
    """Left = explicit + Z eps_HOMO, right = explicit + Z eps_LUMO (Janak)."""
    s, b, res = weight_setup
    gap = float(res.eigenvalues[0, 1] - res.eigenvalues[0, 0])
    diff = weight_grad(s, b, +1)[0] - weight_grad(s, b, -1)[0]
    np.testing.assert_allclose(diff, 1.0 * gap, atol=WEIGHT_GAP_ATOL)


def test_cell_gradient_matches_fixed_basis_finite_difference():
    """At fixed Miller sets and grid shape the energy is a smooth function of the cell;
    nothing physical is concluded (no Pulay stress)."""
    s = h2(L_SMALL)
    b = basis_for(s, ECUT_SMALL)
    opts = SCFOptions(tol=SCF_TOL)
    g = np.asarray(jax.grad(lambda st: energy(st, b, h2_terms(), opts))(s).lattice.vectors)
    assert np.all(np.isfinite(g))
    A = np.asarray(s.lattice.vectors)
    h = FD_STEP
    for i, j in ((0, 0), (2, 1)):
        e = []
        for sign in (+1, -1):
            Ap = A.copy()
            Ap[i, j] += sign * h
            e.append(float(run(Structure(Lattice(Ap), s.positions, s.species, s.weights), b).total_energy))
        np.testing.assert_allclose(g[i, j], (e[0] - e[1]) / (2 * h), atol=FORCE_ATOL)


def test_displaced_geometry_helper_moves_one_atom():
    s = h2()
    r0 = np.asarray(s.cartesian_positions)
    r1 = np.asarray(displaced(s, 1, [0, 0, 0.01]).cartesian_positions)
    np.testing.assert_allclose(r1 - r0, [[0, 0, 0], [0, 0, 0.01]], atol=1e-14)
