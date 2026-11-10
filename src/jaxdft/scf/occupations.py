"""Aufbau occupations with a one-sided slope at integer electron counts
(theory post 2a, 'Occupations', 'Per-atom weights')."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def _check(n_electrons, n_bands: int, side: int) -> None:
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 (HOMO side) or +1 (LUMO side), got {side}")
    if isinstance(n_electrons, jax.core.Tracer):
        return
    n = float(n_electrons)
    if n < 0 or n > 2 * n_bands:
        raise ValueError(f"{n} electrons do not fit in {n_bands} bands")


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def _aufbau(n_electrons, n_bands, side):
    n = jnp.arange(n_bands)
    return jnp.clip(n_electrons - 2.0 * n, 0.0, 2.0)


@_aufbau.defjvp
def _aufbau_jvp(n_bands, side, primals, tangents):
    (n_el,), (dn,) = primals, tangents
    n = jnp.arange(n_bands)
    lower = 2.0 * n
    if side < 0:  # the band that loses charge as N decreases: 2n < N <= 2n + 2
        filling = (n_el > lower) & (n_el <= lower + 2.0)
    else:  # the band that gains charge as N increases: 2n <= N < 2n + 2
        filling = (n_el >= lower) & (n_el < lower + 2.0)
    return _aufbau(n_el, n_bands, side), jnp.where(filling, 1.0, 0.0) * dn


def aufbau_occupations(n_electrons, n_bands: int, side: int = -1) -> jax.Array:
    """f_n = min(2, max(0, N - 2n)) for n = 0..n_bands-1 (spin included).

    Piecewise linear in N. At an integer kink the derivative is one-sided: ``side=-1``
    differentiates as charge leaves the highest occupied band (HOMO side), ``side=+1``
    as it enters the lowest unoccupied band (LUMO side). Away from kinks both agree.
    """
    _check(n_electrons, n_bands, side)
    return _aufbau(jnp.asarray(n_electrons, dtype=jnp.float64), int(n_bands), int(side))
