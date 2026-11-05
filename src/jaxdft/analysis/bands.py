"""Band edges and extrema (theory post 1b, 'Reference values': the indirect gap is the
minimum along Delta, not the level at X)."""

from __future__ import annotations

import math

import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.calculations.bands import band_structure

_INV_PHI = (math.sqrt(5) - 1) / 2


def valence_band_maximum(bands, n_valence: int) -> float:
    """Highest energy of band ``n_valence`` (1-based count of valence bands) over all k."""
    return float(jnp.max(bands.eigenvalues[:, n_valence - 1]))


def band_minimum_on_segment(
    structure, k_start, k_end, band: int, ecut: float, terms, n_samples: int = 41, xtol: float = 1e-6
) -> tuple[float, float]:
    """Minimum of band ``band`` (0-based) along k(t) = k_start + t (k_end - k_start),
    t in [0, 1], fractional k. Returns (t, energy in Ha).

    Dense vmapped sampling, then golden-section refinement of the bracket around the best
    sample. Golden section needs no derivative, so it tolerates the tiny steps from the
    k-dependent basis. All refinement steps use one padded length, so they compile once.
    """
    k_start, k_end = jnp.asarray(k_start, dtype=jnp.float64), jnp.asarray(k_end, dtype=jnp.float64)
    lattice = structure.lattice
    ts = jnp.linspace(0.0, 1.0, n_samples)
    samples = PlaneWaveBasis.build(lattice, k_start + ts[:, None] * (k_end - k_start), ecut)
    e = band_structure(structure, samples, n_bands=band + 1, terms=terms).eigenvalues[:, band]
    n_pad = samples.n_pad + max(16, samples.n_pad // 20)

    def energy(t):
        nonlocal n_pad
        k = (k_start + t * (k_end - k_start))[None]
        try:
            basis = PlaneWaveBasis.build(lattice, k, ecut, n_pad=n_pad)
        except ValueError:  # this k needs a larger basis than any sample: grow once
            n_pad = PlaneWaveBasis.build(lattice, k, ecut).n_pad + 16
            basis = PlaneWaveBasis.build(lattice, k, ecut, n_pad=n_pad)
        return float(band_structure(structure, basis, n_bands=band + 1, terms=terms).eigenvalues[0, band])

    i = int(jnp.argmin(e))
    lo, hi = float(ts[max(i - 1, 0)]), float(ts[min(i + 1, n_samples - 1)])
    x1, x2 = hi - _INV_PHI * (hi - lo), lo + _INV_PHI * (hi - lo)
    f1, f2 = energy(x1), energy(x2)
    while hi - lo > xtol:
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - _INV_PHI * (hi - lo)
            f1 = energy(x1)
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + _INV_PHI * (hi - lo)
            f2 = energy(x2)
    candidates = [(x1, f1), (x2, f2)]
    for t_end in (0.0, 1.0):  # a minimum at an endpoint is found exactly
        if lo <= t_end <= hi or abs(t_end - lo) <= xtol or abs(t_end - hi) <= xtol:
            candidates.append((t_end, energy(t_end)))
    t, f = min(candidates, key=lambda c: c[1])
    return t, f
