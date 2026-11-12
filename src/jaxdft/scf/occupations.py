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


# --- occupations over a k-grid (theory post 2b; engineering log, 2b API) ------------------


def reference_count(n_electrons):
    """N_ref: the even integer nearest to N (halves round up), as a traced float."""
    n = jax.lax.stop_gradient(jnp.asarray(n_electrons, dtype=jnp.float64))
    return 2.0 * jnp.floor(n / 2.0 + 0.5)


def _levels(key, w, candidate, tol):
    """Group candidate states (flat) into levels in ascending ``key``: a new level
    starts where the gap to the previous sorted key exceeds ``tol``. Non-candidates are
    pushed into one level above all candidates. Returns (order, level id per sorted
    state, level capacity 2 sum w, capacity of all lower levels), the last two per
    sorted state."""
    sentinel = jnp.max(jnp.where(candidate, jnp.abs(key), 0.0)) + 1e3
    key = jnp.where(candidate, key, sentinel)
    order = jnp.argsort(key, stable=True)
    ks, ws = key[order], w[order]
    new = jnp.concatenate([jnp.zeros(1, dtype=int), (jnp.diff(ks) > tol).astype(int)])
    level = jnp.cumsum(new)
    n = key.shape[0]
    cap_level = jax.ops.segment_sum(2.0 * ws, level, num_segments=n)
    before_level = jnp.cumsum(cap_level) - cap_level
    return order, cap_level[level], before_level[level]


def _fill(amount, cap, before):
    """Change of f per sorted state when ``amount`` electrons are distributed level by
    level, equally per state within a level (Delta f = fill / sum w = 2 fill / cap)."""
    fill = jnp.clip(amount - before, 0.0, cap)
    return jnp.where(cap > 0, 2.0 * fill / jnp.where(cap > 0, cap, 1.0), 0.0)


def _active(amount, cap, before, right_open):
    """d(Delta f)/d(amount) per sorted state: 2 / cap on the level being filled."""
    if right_open:
        on = (before <= amount) & (amount < before + cap)
    else:
        on = (before < amount) & (amount <= before + cap)
    return jnp.where(on & (cap > 0), 2.0 / jnp.where(cap > 0, cap, 1.0), 0.0)


def _unsort(values, order):
    return jnp.zeros_like(values).at[order].set(values)


def _grid_parts(eigenvalues, kweights, n_electrons, tol):
    e = jax.lax.stop_gradient(jnp.asarray(eigenvalues, dtype=jnp.float64))
    w = jax.lax.stop_gradient(jnp.asarray(kweights, dtype=jnp.float64))
    n_k, n_bands = e.shape
    n_ref = reference_count(n_electrons)
    m = n_ref / 2.0
    band = jnp.broadcast_to(jnp.arange(n_bands), (n_k, n_bands))
    occupied = (band < m).reshape(-1)
    flat_e = e.reshape(-1)
    flat_w = jnp.broadcast_to(w[:, None], (n_k, n_bands)).reshape(-1)
    delta = jnp.asarray(n_electrons, dtype=jnp.float64) - n_ref
    add = _levels(flat_e, flat_w, ~occupied, tol)
    rem = _levels(-flat_e, flat_w, occupied, tol)
    base = jnp.where(occupied, 2.0, 0.0)
    return (n_k, n_bands), base, delta, add, rem


@partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _grid_occupations(eigenvalues, kweights, n_electrons, side, tol):
    shape, base, delta, (oa, ca, ba), (orr, cr, br) = _grid_parts(eigenvalues, kweights, n_electrons, tol)
    up = _unsort(_fill(jnp.maximum(delta, 0.0), ca, ba), oa)
    down = _unsort(_fill(jnp.maximum(-delta, 0.0), cr, br), orr)
    return (base + up - down).reshape(shape)


@_grid_occupations.defjvp
def _grid_occupations_jvp(side, tol, primals, tangents):
    eigenvalues, kweights, n_el = primals
    dn = tangents[2]
    f = _grid_occupations(eigenvalues, kweights, n_el, side, tol)
    shape, _base, delta, (oa, ca, ba), (orr, cr, br) = _grid_parts(eigenvalues, kweights, n_el, tol)
    a, r = jnp.maximum(delta, 0.0), jnp.maximum(-delta, 0.0)
    if side > 0:  # the level that gains as N increases
        d_add = _unsort(_active(a, ca, ba, right_open=True), oa)
        d_rem = _unsort(_active(r, cr, br, right_open=False), orr)
        use_add = delta >= 0
    else:  # the level that loses as N decreases
        d_add = _unsort(_active(a, ca, ba, right_open=False), oa)
        d_rem = _unsort(_active(r, cr, br, right_open=True), orr)
        use_add = delta > 0
    d = jnp.where(use_add, d_add, d_rem).reshape(shape)
    return f, d * dn


def grid_occupations(eigenvalues, kweights, n_electrons, *, side: int = -1, tol: float = 1e-6) -> jax.Array:
    """Occupations f_nk (spin included) for a k-grid (theory post 2b, 'Occupations,
    the gap and the frontier over all k-points'; engineering log, 2b API).

    ``eigenvalues`` (n_k, n_bands), ascending per k; ``kweights`` (n_k,) summing to 1.
    The reference is band filling at N_ref, the even integer nearest to N (halves up):
    f = 2 for bands n < N_ref / 2 at every k. The remainder N - N_ref is added to the
    lowest empty states, or removed from the highest occupied ones, over all k-points,
    level by level; states within ``tol`` (Ha) of each other form a level and change
    by the same amount. sum_k w_k sum_n f_nk = N.

    The derivative with respect to N is one-sided at kinks: ``side=-1`` moves the
    level that loses charge as N decreases (the VBM level at even N), ``side=+1`` the
    level that gains as N increases (the CBM level). Eigenvalues and weights are
    treated as constants: no derivative flows through them.
    """
    e = jnp.asarray(eigenvalues)
    if e.ndim != 2:
        raise ValueError(f"eigenvalues must have shape (n_k, n_bands), got {e.shape}")
    _check(n_electrons, e.shape[1], side)
    return _grid_occupations(e, jnp.asarray(kweights, dtype=jnp.float64), jnp.asarray(n_electrons, dtype=jnp.float64), int(side), float(tol))


def frontier_gap(eigenvalues, n_electrons) -> tuple[jax.Array, jax.Array]:
    """(VBM, CBM) of the band filling at N_ref over all k-points: the highest
    eigenvalue of band N_ref/2 - 1 and the lowest of band N_ref/2. -inf for the VBM
    without occupied bands, NaN for the CBM without a band above (both traceable)."""
    e = jnp.asarray(eigenvalues)
    n_bands = e.shape[1]
    m = (reference_count(n_electrons) / 2.0).astype(int)
    vbm = jnp.where(m > 0, jnp.max(jnp.take(e, jnp.maximum(m - 1, 0), axis=1)), -jnp.inf)
    cbm = jnp.where(m < n_bands, jnp.min(jnp.take(e, jnp.minimum(m, n_bands - 1), axis=1)), jnp.nan)
    return vbm, cbm
