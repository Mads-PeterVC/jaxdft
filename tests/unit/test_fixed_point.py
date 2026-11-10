"""The fixed-point solver and Anderson mixing (theory post 02a, 'Self-consistency as a
fixed point'): solve(F, x0) with mixing in the solver, traceable, never silently
unconverged."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxdft import Anderson
from jaxdft.scf.density import clip_and_renormalise
from jaxdft.scf.solve import FixedPointResult, fixed_point
from support.mixers import AdaptiveLinear


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


def test_any_mixer_slots_in():
    """fixed_point knows only the Mixer protocol (init, update): a mixer with
    different state from Anderson's converges to the same fixed point."""
    M, b, x_star = linear_map()
    res = fixed_point(lambda x: M @ x + b, jnp.zeros(12), AdaptiveLinear(), tol=1e-10, max_iter=2000)
    assert bool(res.converged)
    np.testing.assert_allclose(res.x, x_star, atol=1e-8)


def test_mixer_sees_each_evaluated_iterate_once_and_projected():
    """update receives every input F was evaluated at, after projection, with its
    residual F(x) - x, and nothing else."""
    evaluated, offered = [], []

    @dataclass(frozen=True)
    class Recording:
        def init(self, x0):
            return jnp.asarray(0)

        def update(self, state, x, r):
            jax.debug.callback(lambda y, s: offered.append((np.asarray(y), np.asarray(s))), x, r)
            return x + 0.5 * r + 0.1, state + 1  # leaves the simplex; project restores it

    def F(x):
        jax.debug.callback(lambda y: evaluated.append(np.asarray(y)), x)
        return 0.5 * x + jnp.asarray([0.5, 0.0, 0.0, 0.0])

    project = lambda x: x / jnp.sum(x)
    fixed_point(F, jnp.full(4, 0.25), Recording(), tol=0.0, max_iter=4, project=project)
    assert len(evaluated) == 5 and len(offered) == 4
    for x_eval, (x_seen, r_seen) in zip(evaluated, offered):
        np.testing.assert_array_equal(x_seen, x_eval)
        np.testing.assert_allclose(x_seen.sum(), 1.0, atol=1e-15)
        np.testing.assert_allclose(r_seen, 0.5 * x_seen + np.array([0.5, 0, 0, 0]) - x_seen, atol=1e-15)


# --- history and the per-step hook (engineering log, 'SCF observability') -------------


def test_history_records_every_evaluated_iterate():
    M, b, _ = linear_map()
    F = lambda x: M @ x + b
    res = fixed_point(
        F, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-10, max_iter=50, record=lambda x, f, aux: {"x0": x[0]}
    )
    n = int(res.n_iter)
    h = {k: np.asarray(v) for k, v in res.history.items()}
    assert set(h) == {"residual", "projection", "x0"}
    assert all(v.shape == (51,) for v in h.values())
    assert np.all(np.isfinite(h["residual"][: n + 1])) and np.all(np.isnan(h["residual"][n + 1 :]))
    assert h["residual"][0] == pytest.approx(float(jnp.linalg.norm(F(jnp.zeros(12)))), rel=1e-14)
    assert h["residual"][n] == float(res.residual)
    assert h["x0"][n] == float(res.x[0])
    np.testing.assert_array_equal(h["projection"][: n + 1], 0.0)  # no projection given


def test_recording_does_not_change_the_iterates():
    M, b, _ = linear_map()
    F = lambda x: M @ x + b
    plain = fixed_point(F, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-10, max_iter=50)
    recorded = fixed_point(
        F, jnp.zeros(12), Anderson(depth=6, beta=0.7), tol=1e-10, max_iter=50, record=lambda x, f, aux: {}
    )
    assert plain.history is None
    np.testing.assert_array_equal(plain.x, recorded.x)
    assert int(plain.n_iter) == int(recorded.n_iter)


def test_projection_change_is_recorded():
    """Clipping a negative proposal shows up as a nonzero projection entry."""
    F = lambda x: jnp.array([1.2, -0.2, 0.5, 0.5])  # a target outside the feasible set
    project = lambda x: clip_and_renormalise(x, 2.0, 1.0 * x.size)
    res = fixed_point(
        F, jnp.full(4, 0.5), Anderson(depth=0, beta=1.0), tol=1e-12, max_iter=3, project=project, record=lambda *a: {}
    )
    assert float(np.nanmax(np.asarray(res.history["projection"]))) > 0.1


def test_on_step_sees_each_iterate_in_order():
    M, b, _ = linear_map()
    seen = []

    def on_step(i, values):
        jax.debug.callback(lambda i, r: seen.append((int(i), float(r))), i, values["residual"], ordered=True)

    res = jax.jit(
        lambda x0: fixed_point(lambda x: M @ x + b, x0, Anderson(depth=6, beta=0.7), 1e-10, 50, on_step=on_step)
    )(jnp.zeros(12))
    n = int(res.n_iter)
    assert [i for i, _ in seen] == list(range(n + 1))
    np.testing.assert_array_equal([r for _, r in seen], np.asarray(res.history["residual"])[: n + 1])


def test_recording_needs_a_static_max_iter():
    with pytest.raises(TypeError, match="static"):
        fixed_point(lambda x: x, jnp.zeros(2), Anderson(), 1e-10, jnp.asarray(5), record=lambda *a: {})
