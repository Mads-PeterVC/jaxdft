"""The fixed-point solver and Anderson mixing (theory post 02a, 'Self-consistency as a
fixed point'): solve(F, x0) with mixing in the solver, traceable, never silently
unconverged."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Anderson
from jaxdft.scf.density import clip_and_renormalise
from jaxdft.scf.solve import FixedPointResult, fixed_point


def linear_map(n=12, radius=0.95, seed=0):
    """F(x) = M x + b with spectral radius 0.95: plain iteration converges slowly."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    M = Q @ np.diag(np.linspace(-radius, radius, n)) @ Q.T
    b = rng.normal(size=n)
    x_star = np.linalg.solve(np.eye(n) - M, b)
    return jnp.asarray(M), jnp.asarray(b), x_star


def test_anderson_converges_to_the_fixed_point():
    M, b, x_star = linear_map()
    res = fixed_point(lambda x: M @ x + b, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-12, max_iter=200)
    assert isinstance(res, FixedPointResult)
    assert bool(res.converged)
    assert float(res.residual) < 1e-12
    np.testing.assert_allclose(res.x, x_star, atol=1e-10)


def test_anderson_beats_simple_mixing_on_a_slow_contraction():
    M, b, _ = linear_map()
    F = lambda x: M @ x + b
    fast = fixed_point(F, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-10, max_iter=500)
    # Simple mixing contracts the slowest mode by 1 - 0.7 * 0.05 = 0.965 per step: ~715
    # steps to 1e-10, so the cap must allow that.
    slow = fixed_point(F, jnp.zeros(12), Anderson(depth=0, beta=0.7), tol=1e-10, max_iter=2000)
    assert bool(fast.converged) and bool(slow.converged)
    assert int(fast.n_iter) < int(slow.n_iter) / 3


def test_depth_zero_is_simple_mixing():
    """Anderson(depth=0, beta) is x <- x + beta (F(x) - x), exactly."""
    M, b, _ = linear_map()
    F = lambda x: M @ x + b
    x = jnp.zeros(12)
    for _ in range(5):
        x = x + 0.3 * (F(x) - x)
    res = fixed_point(F, jnp.zeros(12), Anderson(depth=0, beta=0.3), tol=0.0, max_iter=5)
    np.testing.assert_allclose(res.x, x, atol=1e-14)
    assert int(res.n_iter) == 5
    assert not bool(res.converged)


def test_unconverged_is_flagged():
    M, b, _ = linear_map()
    res = fixed_point(lambda x: M @ x + b, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-12, max_iter=3)
    assert not bool(res.converged)
    assert int(res.n_iter) == 3


def test_solver_is_traceable():
    """The SCF loop runs under jit and grad (engineering log: energy runs SCF inside)."""
    M, b, x_star = linear_map()

    @jax.jit
    def solve(b):
        return fixed_point(lambda x: M @ x + b, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-12, max_iter=200).x

    np.testing.assert_allclose(solve(b), x_star, atol=1e-10)


def test_project_is_applied_to_every_mixed_iterate():
    """The projection hook (used for clip-then-renormalise) keeps every iterate on the
    constraint set; here: components sum to 1."""
    seen = []

    def F(x):
        jax.debug.callback(lambda y: seen.append(np.asarray(y).sum()), x)
        return 0.5 * x + jnp.asarray([0.5, 0.0, 0.0, 0.0])

    project = lambda x: x / jnp.sum(x)
    res = fixed_point(F, jnp.full(4, 0.25), Anderson(depth=3, beta=0.6), tol=1e-12, max_iter=100, project=project)
    assert bool(res.converged)
    np.testing.assert_allclose(seen, 1.0, atol=1e-14)


def test_anderson_parameters_validated():
    with pytest.raises(ValueError):
        Anderson(depth=-1, beta=0.5)
    with pytest.raises(ValueError):
        Anderson(depth=3, beta=0.0)


def test_clip_and_renormalise():
    """Negative grid densities from mixing are clipped and the integral restored to
    N_el (review 2.9: a non-neutral iterate brings back the G = 0 divergence)."""
    rho = jnp.asarray([[[0.5, -0.1], [0.2, 0.0]], [[0.3, -1e-3], [0.1, 0.4]]])
    volume, n_el = 3.0, 2.0
    out = clip_and_renormalise(rho, n_el, volume)
    assert float(out.min()) >= 0.0
    np.testing.assert_allclose(volume / out.size * float(out.sum()), n_el, rtol=1e-15)
    pos = np.asarray(rho) > 0
    ratio = np.asarray(out)[pos] / np.asarray(rho)[pos]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-15)
