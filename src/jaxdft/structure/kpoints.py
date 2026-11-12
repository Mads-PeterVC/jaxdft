"""High-symmetry points and band-structure paths (theory post 1a, 'Band-structure path').

All k-points are fractional, k = f B. Paths are built once on the host and
returned as arrays; they are inputs to the physics, not part of it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice

# Display labels for ASCII keys.
_DISPLAY = {"G": "Γ"}


def special_points_fcc() -> dict[str, jax.Array]:
    """Fractional coordinates of the fcc high-symmetry points. Key ``"G"`` is Γ."""
    points = {
        "G": (0.0, 0.0, 0.0),
        "X": (0.5, 0.0, 0.5),
        "L": (0.5, 0.5, 0.5),
        "W": (0.25, 0.5, 0.75),
        "K": (0.375, 0.375, 0.75),
        "U": (0.25, 0.625, 0.625),
    }
    return {label: jnp.asarray(f, dtype=jnp.float64) for label, f in points.items()}


@dataclass(frozen=True)
class KPath:
    """A sampled k-path.

    kpoints: (n, 3) fractional k-points.
    distance: (n,) cumulative Cartesian path length (Bohr^-1); does not advance across breaks.
    tick_positions, tick_labels: corner positions and display labels (breaks as "U|K").
    """

    kpoints: jax.Array
    distance: jax.Array
    tick_positions: jax.Array
    tick_labels: tuple[str, ...]


def _allocate(lengths: list[float], n_intervals: int) -> list[int]:
    """Split n_intervals over segments proportionally to length, at least one each
    (largest-remainder rounding, so the total is exact)."""
    if n_intervals < len(lengths):
        raise ValueError(f"need at least {len(lengths)} intervals for {len(lengths)} segments")
    total = sum(lengths)
    spare = n_intervals - len(lengths)
    raw = [spare * length / total for length in lengths]
    counts = [1 + math.floor(r) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in order[: n_intervals - sum(counts)]:
        counts[i] += 1
    return counts


def kpath(
    lattice: Lattice,
    points: dict[str, jax.Array],
    segments: list[list[str]],
    n_points: int,
) -> KPath:
    """Sample a path through labelled points.

    ``segments`` is a list of continuous runs, e.g. ``[["L", "G", "X", "U"], ["K", "G"]]``;
    the path jumps between runs without advancing the distance. Exactly ``n_points``
    points are distributed over the straight pieces in proportion to their Cartesian
    length; each corner appears exactly once per run.
    """
    if any(len(run) < 2 for run in segments):
        raise ValueError("every run needs at least two labels")
    B = lattice.reciprocal
    pieces = [(run[i], run[i + 1]) for run in segments for i in range(len(run) - 1)]
    lengths = [float(jnp.linalg.norm((points[b] - points[a]) @ B)) for a, b in pieces]
    counts = iter(_allocate(lengths, n_points - len(segments)))
    lengths_iter = iter(lengths)

    kpts, dist, ticks, labels = [], [], [], []
    d = 0.0
    for r, run in enumerate(segments):
        start_label = _DISPLAY.get(run[0], run[0])
        if r == 0:
            ticks.append(d)
            labels.append(start_label)
        else:
            labels[-1] = f"{labels[-1]}|{start_label}"
        for a, b in zip(run[:-1], run[1:]):
            n, length = next(counts), next(lengths_iter)
            t = jnp.arange(n, dtype=jnp.float64) / n
            kpts.append(points[a][None, :] + t[:, None] * (points[b] - points[a])[None, :])
            dist.append(d + t * length)
            d += length
            ticks.append(d)
            labels.append(_DISPLAY.get(b, b))
        kpts.append(points[run[-1]][None, :])
        dist.append(jnp.array([d]))

    return KPath(
        kpoints=jnp.concatenate(kpts),
        distance=jnp.concatenate(dist),
        tick_positions=jnp.asarray(ticks, dtype=jnp.float64),
        tick_labels=tuple(labels),
    )


def monkhorst_pack(shape, shift=(0.0, 0.0, 0.0)) -> tuple[jax.Array, jax.Array]:
    """Uniform k-grid (theory post 2b, 'Brillouin-zone sampling'): fractional
    k = (n + s) / N for n_i = 0..N_i - 1, wrapped into [0, 1), with uniform weights.

    ``shift`` s is in units of the grid spacing: 0 is the Gamma-centred grid used by
    the SCF, 1/2 the half-shifted grid (which, without density symmetrisation, breaks
    the symmetry of the density; theory post 2b). Returns (kpoints (n, 3), weights (n,)).
    """
    shape = tuple(int(n) for n in shape)
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError(f"k-grid shape must be three positive integers, got {shape}")
    shift = tuple(float(s) for s in shift)
    if len(shift) != 3:
        raise ValueError(f"shift must have three components, got {shift}")
    axes = [[((i + s) / n) % 1.0 for i in range(n)] for n, s in zip(shape, shift)]
    k = jnp.asarray([[a, b, c] for a in axes[0] for b in axes[1] for c in axes[2]], dtype=jnp.float64)
    n = math.prod(shape)
    return k, jnp.full(n, 1.0 / n, dtype=jnp.float64)
