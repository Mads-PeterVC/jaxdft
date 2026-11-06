"""Grid invariants for any cell and cutoff (theory post 02a, 'The FFT grid')."""

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft import Lattice, PlaneWaveBasis
from support.budget import examples
from support.strategies import cells, kfracs


def is_smooth_235(n):
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@settings(max_examples=examples(10))
@given(cells(), st.floats(1.0, 6.0))
def test_grid_shape_holds_every_basis_difference_without_wrapping(A, ecut):
    b = PlaneWaveBasis.build(Lattice(A), np.zeros((1, 3)), ecut)
    N = np.asarray(b.fft_shape)
    assert all(is_smooth_235(int(n)) for n in N)
    m = np.asarray(b.miller[0])[np.asarray(b.mask[0])]
    span = m.max(axis=0) - m.min(axis=0)  # largest |m_i - m'_i| that occurs
    assert np.all(N >= 2 * span + 1)


@settings(max_examples=examples(10))
@given(cells(), st.floats(1.0, 4.0), kfracs(), st.integers(1, 3))
def test_round_trip_any_cell_and_k(A, ecut, k, n):
    b = PlaneWaveBasis.build(Lattice(A), np.atleast_2d(k), ecut)[0]
    rng = np.random.default_rng(0)
    c = rng.normal(size=(b.n_pad, n)) + 1j * rng.normal(size=(b.n_pad, n))
    c = jnp.asarray(np.where(np.asarray(b.mask)[:, None], c, 0))
    np.testing.assert_allclose(b.from_grid(b.to_grid(c)), c, atol=1e-12)
