"""Mixing schemes for the fixed-point solver (theory post 2a, 'Self-consistency as a
fixed point'): mixing belongs to the solver, not to the SCF map.

A mixer is any object satisfying ``Mixer``. The solver owns evaluation, projection
and the convergence test; the mixer only proposes the next input from the iterates it
has seen. ``Anderson`` is the one 2a uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import jax
import jax.numpy as jnp


class Mixer(Protocol):
    """A mixing scheme for ``fixed_point``.

    - ``init(x0)`` returns the initial mixer state; ``x0`` gives shape and dtype only.
    - ``update(state, x, r)`` is called once per evaluated iterate with the input
      ``x`` (already projected) and its residual ``r = F(x) - x``. It returns the
      next, unprojected input and the new state.

    Both run inside ``jax.lax.while_loop``: the state must be a pytree whose shapes
    and dtypes do not change between iterations. The mixer is a static argument of
    ``jit``, so it must be hashable (a frozen dataclass of Python scalars is).
    """

    def init(self, x0: jax.Array) -> Any: ...

    def update(self, state: Any, x: jax.Array, r: jax.Array) -> tuple[jax.Array, Any]: ...


@dataclass(frozen=True)
class Anderson:
    """Anderson acceleration (Pulay/DIIS mixing) over the last ``depth`` residual
    differences, with damping ``beta``. ``depth=0`` is simple linear mixing,
    x <- x + beta (F(x) - x)."""

    depth: int = 6
    beta: float = 0.7

    def __post_init__(self):
        if int(self.depth) != self.depth or self.depth < 0:
            raise ValueError(f"depth must be a non-negative integer, got {self.depth}")
        if not 0 < self.beta <= 1:
            raise ValueError(f"beta must lie in (0, 1], got {self.beta}")

    def init(self, x0):
        """History of the last depth + 1 (x, r) pairs, newest last, flattened, and
        the number of pairs recorded so far (capped at depth + 1)."""
        if self.depth == 0:
            return ()
        rows = (self.depth + 1, x0.size)
        return jnp.zeros(rows, x0.dtype), jnp.zeros(rows, x0.dtype), jnp.asarray(0)

    def update(self, state, x, r):
        if self.depth == 0:
            return x + self.beta * r, state
        m = self.depth
        xs, rs, count = state
        xf, rf = x.ravel(), r.ravel()
        xs = jnp.roll(xs, -1, axis=0).at[-1].set(xf)
        rs = jnp.roll(rs, -1, axis=0).at[-1].set(rf)
        count = jnp.minimum(count + 1, m + 1)

        dX = xs[1:] - xs[:-1]  # (m, size)
        dR = rs[1:] - rs[:-1]
        valid = jnp.arange(m) >= m - (count - 1)  # columns with two real entries
        dX = jnp.where(valid[:, None], dX, 0)
        dR = jnp.where(valid[:, None], dR, 0)
        gamma = jnp.linalg.lstsq(dR.T, rf)[0]
        gamma = jnp.where(valid, gamma, 0)
        x_bar = xf - gamma @ dX
        r_bar = rf - gamma @ dR
        x_new = x_bar + self.beta * r_bar
        return x_new.reshape(x.shape), (xs, rs, count)
