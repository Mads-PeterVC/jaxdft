"""Ewald invariants for any cell and charges (theory post 02a, 'Electrostatics')."""

import jax.numpy as jnp
import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from jaxdft import Lattice, Structure
from jaxdft.hamiltonian.ewald import ewald_energy
from support.budget import examples
from support.strategies import cells
from support.tolerances import EXACT_ATOL

positions = hnp.arrays(float, (3, 3), elements=st.floats(0, 1))
charges = hnp.arrays(float, (3,), elements=st.floats(0.2, 4.0))


def separated(A, f, minimum=0.5):
    """Point charges must not coincide (the energy is infinite): minimum image distance."""
    df = f[:, None, :] - f[None, :, :]
    df -= np.round(df)
    images = np.array(np.meshgrid(*[np.arange(-1, 2)] * 3, indexing="ij")).reshape(3, -1).T
    d = np.linalg.norm((df[:, :, None, :] + images) @ A, axis=-1)
    d[np.arange(3), np.arange(3), 13] = np.inf  # self, home cell
    return d.min() > minimum


def make(A, f, q):
    return Structure(Lattice(A), f, ["X", "Y", "Z"]), jnp.asarray(q)


@settings(max_examples=examples(10))
@given(cells(), positions, charges, st.floats(0.5, 2.0))
def test_eta_independence(A, f, q, factor):
    assume(separated(A, f))
    s, qq = make(A, f, q)
    e0 = float(ewald_energy(s, qq))
    scale = abs(np.linalg.det(A)) ** (-1 / 3)
    e1 = float(ewald_energy(s, qq, eta=factor * 3.0 * scale))
    np.testing.assert_allclose(e1, e0, atol=EXACT_ATOL * max(1.0, abs(e0)))


@settings(max_examples=examples(10))
@given(cells(), positions, charges, hnp.arrays(float, (3,), elements=st.floats(-2, 2)))
def test_translation_and_permutation_invariance(A, f, q, t):
    assume(separated(A, f))
    s, qq = make(A, f, q)
    e0 = float(ewald_energy(s, qq))
    st_, _ = make(A, f + t, q)
    np.testing.assert_allclose(float(ewald_energy(st_, qq)), e0, atol=EXACT_ATOL * max(1.0, abs(e0)))
    perm = [2, 0, 1]
    sp, qp = make(A, f[perm], q[perm])
    np.testing.assert_allclose(float(ewald_energy(sp, qp)), e0, atol=EXACT_ATOL * max(1.0, abs(e0)))
