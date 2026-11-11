"""Forces, cell gradients and one-sided weight derivatives of silicon from
jax.grad(energy) (theory post 02b, 'Forces and the cell', 'Fractional electron counts
and the weight derivatives'; decision 1).

energy() stops the gradient on the converged orbitals; finite differences of
converged energies must agree to the truncation and SCF errors in
``support/tolerances.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Lattice, SCFOptions, Structure, energy
from support.silicon import (
    DISPLACEMENT,
    KGRID_DISPLACED,
    basis_for,
    converged_displaced,
    displaced_si,
    position_gradient,
    run,
    si,
    si_terms,
    silicon,
    with_positions,
    with_weights,
)
from support.tolerances import (
    CROSS_CHECK_SI_DEDF,
    CROSS_CHECK_SI_DEDF_ATOL,
    FD_FRACTIONAL_ATOL,
    FD_FRACTIONAL_STEP,
    NET_FORCE_NO_XC_MAX,
    NET_FORCE_XC_MAX,
    SCF_TOL,
    SCF_TOL_TIGHT,
    SI_WEIGHT_ATOL,
    SI_WEIGHT_ATOL_DEGENERATE,
    SI_WEIGHT_GAP_ATOL,
    SI_WEIGHT_STEP,
)

pytestmark = pytest.mark.slow
Z_SI = 4.0


def fd_position(structure, basis, atom, axis, h=FD_FRACTIONAL_STEP, terms=None):
    e = []
    for sign in (+1, -1):
        f = np.array(structure.positions)
        f[atom, axis] += sign * h
        e.append(float(run(with_positions(structure, f), basis, terms=terms).total_energy))
    return (e[0] - e[1]) / (2 * h)


# --- forces -------------------------------------------------------------------------


def test_position_gradient_matches_finite_differences():
    """All components of the displaced atom, nonlocal term included."""
    s, b, _ = converged_displaced()
    g = position_gradient(s, b)
    for axis in range(3):
        np.testing.assert_allclose(g[0, axis], fd_position(s, b, 0, axis), atol=FD_FRACTIONAL_ATOL)


def test_cross_check_of_the_force_on_the_three_cubed_grid():
    """dE/df_0x at the displaced geometry, 3^3 grid, 12 Ha: the throwaway check and the
    review agree to nine digits."""
    s, b, _ = converged_displaced()
    assert b.kpoints.shape[0] == 27
    np.testing.assert_allclose(position_gradient(s, b)[0, 0], CROSS_CHECK_SI_DEDF, atol=CROSS_CHECK_SI_DEDF_ATOL)


@pytest.mark.parametrize(("xc", "bound"), [(True, NET_FORCE_XC_MAX), (False, NET_FORCE_NO_XC_MAX)], ids=["with-xc", "without-xc"])
def test_net_force_at_the_displaced_geometry(xc, bound):
    """The acceptance pair of the theory post: the net force is the egg-box effect,
    caused by the XC quadrature alone; without XC it vanishes at self-consistency.

    Silicon without XC is a semimetal (gap over the 3^3 grid -0.035 Ha). The identity
    tested holds at any self-consistent fixed point with fixed occupations, ground state
    or not, so the XC-off run disables the gap guard (decided 2026-11-11 with the
    project lead)."""
    s, b, _ = converged_displaced()
    A = np.asarray(s.lattice.vectors)
    options = {} if xc else {"min_gap": -np.inf}
    F = -position_gradient(s, b, terms=si_terms(xc=xc), **options) @ np.linalg.inv(A).T
    assert np.linalg.norm(F.sum(axis=0)) < bound


def test_cell_gradient_matches_fixed_basis_finite_difference():
    """Now with the nonlocal projectors, which depend on the cell through q and
    Omega^-1/2. Fixed Miller sets and grid: nothing physical is concluded."""
    s, b, _ = converged_displaced()
    opts = SCFOptions(tol=SCF_TOL)
    g = np.asarray(jax.grad(lambda st: energy(st, b, si_terms(), opts))(s).lattice.vectors)
    assert np.all(np.isfinite(g))
    A = np.asarray(s.lattice.vectors)
    h = 1e-3
    for i, j in ((0, 0), (2, 1)):
        e = []
        for sign in (+1, -1):
            Ap = A.copy()
            Ap[i, j] += sign * h
            e.append(float(run(Structure(Lattice(Ap), s.positions, s.species, s.weights), b).total_energy))
        np.testing.assert_allclose(g[i, j], (e[0] - e[1]) / (2 * h), atol=1e-6)


# --- one-sided weight derivatives ------------------------------------------------------


def weight_gradient(s, b, side):
    opts = SCFOptions(tol=SCF_TOL_TIGHT)
    return np.asarray(jax.grad(lambda st: energy(st, b, si_terms(), opts, side=side))(s).weights)


def frontier_set(res, side, n_ref=8):
    """States (k, n) whose occupation differs from the band filling of N_ref."""
    f = np.asarray(res.occupations)
    ref = np.zeros_like(f)
    ref[:, : n_ref // 2] = 2.0
    changed = np.abs(f - ref) > 1e-12
    return {tuple(x) for x in np.argwhere(changed)}


def frontier_level(res, side, tol=1e-6):
    """The states the one-sided derivative at integer N moves: the VBM level (side
    -1) or the CBM level (side +1) over all k-points."""
    e = np.asarray(res.eigenvalues)
    band = 3 if side < 0 else 4
    edge = e[:, band].max() if side < 0 else e[:, band].min()
    level = set()
    for k in range(e.shape[0]):
        for n in range(e.shape[1]):
            occupied = n < 4
            if occupied == (side < 0) and abs(e[k, n] - edge) <= tol:
                level.add((k, n))
    return level


def one_sided_fd(s, b, res0, side, delta=SI_WEIGHT_STEP):
    """(-3 E0 + 4 E(d) - E(2d)) / (2 d) in the direction `side`, with lambda_0 = 1 +
    side d, 1 + side 2d. Asserts the frontier set is the same at both points and equal
    to the level the derivative at lambda = 1 moves (review, open question 1)."""
    e, sets = [], []
    for k in (1, 2):
        w = np.asarray(s.weights).copy()
        w[0] = 1.0 + side * k * delta
        r = run(with_weights(s, w), b, tol=SCF_TOL_TIGHT)
        e.append(float(r.total_energy))
        sets.append(frontier_set(r, side))
    assert sets[0] == sets[1] == frontier_level(res0, side), (sets, frontier_level(res0, side))
    return side * (-3 * float(res0.total_energy) + 4 * e[0] - e[1]) / (2 * delta)


@pytest.fixture(scope="module")
def displaced_tight():
    s = displaced_si()
    b = basis_for(s, kshape=KGRID_DISPLACED)
    return s, b, run(s, b, tol=SCF_TOL_TIGHT)


@pytest.fixture(scope="module")
def ideal_tight():
    s = si()
    b = basis_for(s)
    return s, b, run(s, b, tol=SCF_TOL_TIGHT)


def test_weight_derivative_sides_differ_by_z_times_the_gap_over_the_grid(displaced_tight):
    """Left = explicit + Z eps_VBM, right = explicit + Z eps_CBM, frontier levels over
    all k-points. The 2a occupations gave Z times the Brillouin-zone average of a band
    instead (theory post: Z x 0.112 Ha against a 0.010 Ha gap)."""
    s, b, res = displaced_tight
    e = np.asarray(res.eigenvalues)
    gap = float(e[:, 4].min() - e[:, 3].max())
    diff = weight_gradient(s, b, +1) - weight_gradient(s, b, -1)
    np.testing.assert_allclose(diff, Z_SI * gap, atol=SI_WEIGHT_GAP_ATOL)


@pytest.mark.parametrize("side", [-1, +1], ids=["vbm-side", "cbm-side"])
def test_one_sided_weight_derivative_non_degenerate_frontier(displaced_tight, side):
    """The displaced geometry has no spatial symmetry. On the 3^3 grid the VBM is at
    Gamma, time-reversal invariant, so its level is a single state. The CBM lies at a
    k-point that is not, so its level is the pair (k, -k), held together by time
    reversal at every lambda; the set is asserted unchanged over the step."""
    s, b, res = displaced_tight
    level = frontier_level(res, side)
    if side < 0:
        assert level == {(0, 3)}
    else:
        assert len(level) == 2
        (k1, n1), (k2, n2) = sorted(level)
        kp = np.asarray(b.kpoints)
        assert n1 == n2 == 4
        np.testing.assert_allclose(np.mod(kp[k1] + kp[k2] + 1e-9, 1.0), 0.0, atol=1e-8)
    g = weight_gradient(s, b, side)
    np.testing.assert_allclose(g[0], one_sided_fd(s, b, res, side), atol=SI_WEIGHT_ATOL)


def test_one_sided_weight_derivative_degenerate_frontier(ideal_tight):
    """The VBM at Gamma is a triplet, and stays one when lambda breaks the equivalence
    of the two atoms (zincblende keeps Gamma_15). Charge is shared within the level."""
    s, b, res = ideal_tight
    level = frontier_level(res, -1)
    assert len(level) == 3 and {k for k, _ in level} == {0}
    g = weight_gradient(s, b, -1)
    np.testing.assert_allclose(g[0], g[1], atol=1e-10)  # equivalent atoms
    np.testing.assert_allclose(g[0], one_sided_fd(s, b, res, -1), atol=SI_WEIGHT_ATOL_DEGENERATE)


def test_nonlocal_energy_is_linear_and_homogeneous_in_the_weights(displaced_tight):
    """V_nl is linear in each lambda_a, so at fixed orbitals sum_a lambda_a
    dE_nl/dlambda_a = E_nl (Euler): the weights reach the projectors."""
    from jaxdft.scf.scf import total_energy

    s, b, res = displaced_tight
    terms = tuple(t for t in si_terms() if type(t).__name__ == "AtomicNonlocal")

    def e_nl(st):
        return total_energy(st, b, terms, res.orbitals, res.occupations)[0]

    g = np.asarray(jax.grad(e_nl)(s).weights)
    np.testing.assert_allclose(float(np.dot(np.asarray(s.weights), g)), float(res.energies["AtomicNonlocal"]), atol=1e-10)
    assert np.all(g > 0)


def test_displacement_helper():
    s0, s1 = si(), displaced_si()
    np.testing.assert_allclose(np.asarray(s1.positions) - np.asarray(s0.positions), DISPLACEMENT, atol=1e-15)
    assert silicon().ionic_charge == Z_SI
    assert jnp.asarray(s1.weights).shape == (2,)
