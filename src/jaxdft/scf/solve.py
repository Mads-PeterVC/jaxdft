"""solve(F, x0): the fixed-point solver behind the SCF (CLAUDE.md, 'Differentiating
through SCF'; theory post 2a). Traceable: a lax.while_loop, so it runs under jit and
can later be wrapped in a custom_vjp for implicit differentiation.

Observability (engineering log, 'SCF observability'): every evaluated iterate can be
recorded into a fixed-size history, returned with the result, and optionally passed
to a host callback as it happens. Recording does not change the iterates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from jaxdft.scf.mixing import Mixer


class FixedPointResult(NamedTuple):
    """``x`` is the last input iterate, ``fx = F(x)`` and ``residual = norm(fx - x)``.
    ``n_iter`` counts mixing updates; ``converged`` is residual < tol. ``aux`` is the
    auxiliary state of an F that carries one (warm starts), else None.

    ``history`` (when recording) maps names to arrays of length ``max_iter + 1``: entry
    i belongs to the i-th evaluated iterate (i = 0 is the initial guess), entries past
    ``n_iter`` are NaN. It always holds ``residual`` and ``projection`` (norm of the
    change made by ``project`` to the proposed input), plus the ``record`` values."""

    x: jax.Array
    fx: jax.Array
    residual: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    aux: Any = None
    history: dict | None = None


def _l2(r):
    return jnp.sqrt(jnp.sum(r**2))


def fixed_point(
    F,
    x0,
    mixer: Mixer,
    tol,
    max_iter,
    project=None,
    norm=None,
    aux0=None,
    record: Callable | None = None,
    on_step: Callable | None = None,
) -> FixedPointResult:
    """Iterate to x = F(x), with ``mixer`` proposing each next input.

    Each step evaluates f = F(x) and r = f - x. If norm(r) < tol it stops with x;
    otherwise ``mixer.update`` proposes a new x from (x, r) and its history,
    ``project`` (e.g. clip-then-renormalise) is applied, and one update is counted,
    up to ``max_iter`` updates.

    If ``aux0`` is given, F has the signature F(x, aux) -> (f, aux): the aux state
    (e.g. orbitals for warm starts) is carried along and is not mixed.

    Recording. If ``record`` or ``on_step`` is given, ``max_iter`` must be a Python int
    (it sizes the history) and a history is returned. ``record(x, f, aux)`` returns a
    dict of scalars for each evaluated iterate. ``on_step(i, values)`` is called inside
    the trace after each evaluation, with ``values`` holding ``residual``,
    ``projection`` and the ``record`` values; to reach the host, it issues a
    ``jax.debug.callback(..., ordered=True)``. It must not change the iterates.
    """
    norm = _l2 if norm is None else norm
    project = (lambda x: x) if project is None else project
    carries_aux = aux0 is not None
    Fa = F if carries_aux else (lambda x, a: (F(x), a))
    recording = record is not None or on_step is not None
    if recording and not isinstance(max_iter, int):
        raise TypeError("recording a history needs a static (Python int) max_iter")
    record = (lambda x, f, aux: {}) if record is None else record

    def evaluate(x, aux):
        f, aux = Fa(x, aux)
        return f, aux, norm(f - x)

    def entry(x, f, aux, res, moved):
        values = {"residual": res, "projection": moved}
        values.update({k: jnp.asarray(v, dtype=jnp.float64) for k, v in record(x, f, aux).items()})
        return values

    def log(hist, it, values):
        if on_step is not None:
            on_step(it, values)
        if hist is None:
            return None
        return {k: hist[k].at[it].set(values[k]) for k in hist}

    raw0 = jnp.asarray(x0)
    x0 = project(raw0)
    f0, a0, res0 = evaluate(x0, aux0 if carries_aux else 0)
    hist0 = None
    if recording:
        values0 = entry(x0, f0, a0, res0, norm(x0 - raw0))
        hist0 = {k: jnp.full(max_iter + 1, jnp.nan, dtype=jnp.float64) for k in values0}
        hist0 = log(hist0, 0, values0)
    state = (x0, f0, a0, res0, jnp.asarray(0), mixer.init(x0), hist0)

    def cond(state):
        _, _, _, res, it, _, _ = state
        return (res >= tol) & (it < max_iter)

    def body(state):
        x, f, aux, _, it, mix_state, hist = state
        proposed, mix_state = mixer.update(mix_state, x, f - x)
        x_new = project(proposed)
        f_new, aux_new, res_new = evaluate(x_new, aux)
        if recording:
            hist = log(hist, it + 1, entry(x_new, f_new, aux_new, res_new, norm(x_new - proposed)))
        return (x_new, f_new, aux_new, res_new, it + 1, mix_state, hist)

    x, f, aux, res, it, _, hist = jax.lax.while_loop(cond, body, state)
    return FixedPointResult(x, f, res, it, res < tol, aux if carries_aux else None, hist)
