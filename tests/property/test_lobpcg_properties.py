"""LOBPCG invariants of Milestone 2c (theory post 02c, 'The Rayleigh-Ritz penalty';
engineering log, '2026-11-12: Milestone 2c public API').

Dropped directions are zero columns, so the Rayleigh-Ritz matrix is block diagonal
and the penalty only has to lie above the k lowest kept eigenvalues; by Cauchy
interlacing those are at most the current Ritz values. The penalty is the bound
sigma = eps_k + (eps_k - eps_1) + 1 Ha, exposed as ``dropped_penalty`` (module level,
not re-exported) so that it can be tested.
"""

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from jaxdft import lobpcg
from jaxdft.solvers.lobpcg import dropped_penalty
from support.budget import examples


def hermitian(n, seed, spread=20.0):
    rng = np.random.default_rng(seed)
    d = np.sort(rng.uniform(0, spread, n))
    X = 0.1 * (rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n)))
    return np.diag(d) + (X + X.conj().T) / 2


@settings(max_examples=examples(20))
@given(
    eps=st.lists(st.floats(-50, 50, allow_nan=False), min_size=1, max_size=12).map(sorted),
)
def test_penalty_is_the_interlacing_bound(eps):
    e = jnp.asarray(eps, dtype=jnp.float64)
    sigma = float(dropped_penalty(e))
    assert sigma == float(e[-1] + (e[-1] - e[0]) + 1.0)
    assert sigma > eps[-1]


@settings(max_examples=examples(10))
@given(n=st.integers(20, 60), k=st.integers(1, 4), seed=st.integers(0, 2**32 - 1))
def test_first_iteration_never_selects_a_dropped_direction(n, k, seed):
    """At the first iteration P = 0, so k directions are dropped. Selecting one would
    put a zero column into the new block: it would not be orthonormal, or not finite.
    Rayleigh-Ritz on a space containing the old block can only lower the Ritz values."""
    H = hermitian(n, seed)
    rng = np.random.default_rng(seed + 1)
    x0 = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    Q, _ = np.linalg.qr(x0)
    before = np.linalg.eigvalsh(Q.conj().T @ H @ Q)
    Hj = jnp.asarray(H)
    eps, vecs, info = lobpcg(lambda x: Hj @ x, n, k, x0=x0, tol=0.0, max_iter=1, return_info=True)
    eps, vecs = np.asarray(eps), np.asarray(vecs)
    assert int(info.iterations) == 1
    assert np.all(np.isfinite(eps)) and np.all(np.isfinite(vecs))
    np.testing.assert_allclose(vecs.conj().T @ vecs, np.eye(k), atol=1e-12)
    assert np.all(eps <= before + 1e-10)
