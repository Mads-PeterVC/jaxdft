"""Degenerate levels and inversion parities (theory post 1b, 'What the band structure
should look like')."""

from __future__ import annotations

import jax.numpy as jnp


def group_levels(eigenvalues, atol: float) -> list[tuple[float, int]]:
    """Group sorted eigenvalues into (mean energy, degeneracy); a new level starts where
    the gap to the previous eigenvalue exceeds ``atol``."""
    e = [float(x) for x in jnp.sort(jnp.asarray(eigenvalues))]
    groups: list[list[float]] = []
    for x in e:
        if groups and x - groups[-1][-1] <= atol:
            groups[-1].append(x)
        else:
            groups.append([x])
    return [(sum(g) / len(g), len(g)) for g in groups]


def inversion_parities(bands, k_index: int, atol: float = 1e-8) -> list[tuple[float, int, float]]:
    """(energy, degeneracy, parity) for the levels at a time-reversal-invariant k-point.

    Valid when the origin is an inversion centre of the structure. Inversion maps
    c(G) -> c(-G - 2k); 2k must be a reciprocal lattice vector. The parity of a level is
    Re tr(V^dagger P V) / degeneracy over its eigenvectors V: well defined inside a
    degenerate level, +-1 for a level of definite parity. The last level may be
    incomplete if ``n_bands`` cuts through it.
    """
    k = jnp.asarray(bands.kpoints[k_index])
    shift = jnp.rint(2 * k)
    if float(jnp.max(jnp.abs(2 * k - shift))) > 1e-10:
        raise ValueError(f"k = {k} is not time-reversal invariant (2k is not a reciprocal lattice vector)")
    basis = bands.basis[k_index]
    mask = [bool(x) for x in basis.mask]
    miller = [tuple(int(v) for v in m) for m in basis.miller]
    index = {m: i for i, (m, real) in enumerate(zip(miller, mask)) if real}
    s = tuple(int(v) for v in shift)
    perm = [index[tuple(-a - b for a, b in zip(m, s))] if real else i for i, (m, real) in enumerate(zip(miller, mask))]
    V = jnp.asarray(bands.eigenvectors[k_index])
    PV = V[jnp.asarray(perm)]
    out, start = [], 0
    for energy, deg in group_levels(bands.eigenvalues[k_index], atol):
        Vl, PVl = V[:, start : start + deg], PV[:, start : start + deg]
        out.append((energy, deg, float(jnp.real(jnp.trace(Vl.conj().T @ PVl))) / deg))
        start += deg
    return out
