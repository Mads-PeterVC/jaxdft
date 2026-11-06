"""solve(F, x0): the fixed-point solver behind the SCF (CLAUDE.md, 'Differentiating
through SCF'; theory post 2a). Traceable: a lax.while_loop, so it runs under jit and
can later be wrapped in a custom_vjp for implicit differentiation."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxdft.scf.mixing import Mixer


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


def fixed_point(F, x0, mixer: Mixer, tol, max_iter, project=None, norm=None, aux0=None) -> FixedPointResult:
    """Iterate to x = F(x), with ``mixer`` proposing each next input.

    Each step evaluates f = F(x) and r = f - x. If norm(r) < tol it stops with x;
    otherwise ``mixer.update`` proposes a new x from (x, r) and its history,
    ``project`` (e.g. clip-then-renormalise) is applied, and one update is counted,
    up to ``max_iter`` updates.

    If ``aux0`` is given, F has the signature F(x, aux) -> (f, aux): the aux state
    (e.g. orbitals for warm starts) is carried along and is not mixed.
    """
    norm = _l2 if norm is None else norm
    project = (lambda x: x) if project is None else project
    carries_aux = aux0 is not None
    Fa = F if carries_aux else (lambda x, a: (F(x), a))
    x0 = project(jnp.asarray(x0))

    def evaluate(x, aux):
        f, aux = Fa(x, aux)
        return f, aux, norm(f - x)

    f0, a0, res0 = evaluate(x0, aux0 if carries_aux else 0)
    state = (x0, f0, a0, res0, jnp.asarray(0), mixer.init(x0))

    def cond(state):
        _, _, _, res, it, _ = state
        return (res >= tol) & (it < max_iter)

    def body(state):
        x, f, aux, _, it, mix_state = state
        x_new, mix_state = mixer.update(mix_state, x, f - x)
        x_new = project(x_new)
        f_new, aux_new, res_new = evaluate(x_new, aux)
        return (x_new, f_new, aux_new, res_new, it + 1, mix_state)

    x, f, aux, res, it, _ = jax.lax.while_loop(cond, body, state)
    return FixedPointResult(x, f, res, it, res < tol, aux if carries_aux else None)
