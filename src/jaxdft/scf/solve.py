"""solve(F, x0): the fixed-point solver behind the SCF (CLAUDE.md, 'Differentiating
through SCF'; theory post 2a). Traceable: a lax.while_loop, so it runs under jit and
can later be wrapped in a custom_vjp for implicit differentiation."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxdft.scf.mixing import Anderson


class FixedPointResult(NamedTuple):
    """``x`` is the last input iterate, ``fx = F(x)`` and ``residual = norm(fx - x)``.
    ``n_iter`` counts mixing updates; ``converged`` is residual < tol. ``aux`` is the
    auxiliary state of an F that carries one (warm starts), else None."""

    x: jax.Array
    fx: jax.Array
    residual: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    aux: Any = None


def _l2(r):
    return jnp.sqrt(jnp.sum(r**2))


def fixed_point(F, x0, mixer: Anderson, tol, max_iter, project=None, norm=None, aux0=None) -> FixedPointResult:
    """Iterate to x = F(x) with Anderson mixing.

    Each step evaluates f = F(x) and r = f - x. If norm(r) < tol it stops with x;
    otherwise it mixes a new x from the history, applies ``project`` (e.g.
    clip-then-renormalise) and counts one update, up to ``max_iter`` updates.

    If ``aux0`` is given, F has the signature F(x, aux) -> (f, aux): the aux state
    (e.g. orbitals for warm starts) is carried along and is not mixed.
    """
    norm = _l2 if norm is None else norm
    project = (lambda x: x) if project is None else project
    carries_aux = aux0 is not None
    Fa = F if carries_aux else (lambda x, a: (F(x), a))
    x0 = project(jnp.asarray(x0))
    shape, dtype = x0.shape, x0.dtype
    m = max(int(mixer.depth), 1)
    size = x0.size
    beta = mixer.beta

    def evaluate(x, aux):
        f, aux = Fa(x, aux)
        return f, aux, norm(f - x)

    f0, a0, res0 = evaluate(x0, aux0 if carries_aux else 0)
    # History of the last m + 1 (x, r) pairs, newest last, flattened.
    xs = jnp.zeros((m + 1, size), dtype).at[-1].set(x0.ravel())
    rs = jnp.zeros((m + 1, size), dtype).at[-1].set((f0 - x0).ravel())
    state = (x0, f0, a0, res0, jnp.asarray(0), xs, rs, jnp.asarray(1))

    def cond(state):
        _, _, _, res, it, _, _, _ = state
        return (res >= tol) & (it < max_iter)

    def body(state):
        x, f, aux, _, it, xs, rs, count = state
        r = (f - x).ravel()
        if mixer.depth == 0:
            x_new = x.ravel() + beta * r
        else:
            dX = xs[1:] - xs[:-1]  # (m, size)
            dR = rs[1:] - rs[:-1]
            valid = jnp.arange(m) >= m - (count - 1)  # columns with two real entries
            dX = jnp.where(valid[:, None], dX, 0)
            dR = jnp.where(valid[:, None], dR, 0)
            gamma = jnp.linalg.lstsq(dR.T, r)[0]
            gamma = jnp.where(valid, gamma, 0)
            x_bar = x.ravel() - gamma @ dX
            r_bar = r - gamma @ dR
            x_new = x_bar + beta * r_bar
        x_new = project(x_new.reshape(shape))
        f_new, aux_new, res_new = evaluate(x_new, aux)
        xs = jnp.roll(xs, -1, axis=0).at[-1].set(x_new.ravel())
        rs = jnp.roll(rs, -1, axis=0).at[-1].set((f_new - x_new).ravel())
        return (x_new, f_new, aux_new, res_new, it + 1, xs, rs, jnp.minimum(count + 1, m + 1))

    x, f, aux, res, it, _, _, _ = jax.lax.while_loop(cond, body, state)
    return FixedPointResult(x, f, res, it, res < tol, aux if carries_aux else None)
