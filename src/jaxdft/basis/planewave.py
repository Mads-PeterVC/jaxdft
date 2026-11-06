"""Plane-wave basis with a k-dependent kinetic-energy cutoff
(theory post 1a, 'Kinetic-energy cutoff', 'Enumerating the basis', 'Batching over k-points'),
and its FFT grid (theory post 2a, 'The FFT grid').

The basis is purely combinatorial: integer Miller indices and a mask per k-point,
plus the static shape of the FFT grid. Cartesian k+G, kinetic energies and the cell
volume are computed from a Lattice where needed, so the dependence of energies on the
cell stays differentiable. For the same reason the grid transforms are unnormalised
index operations: ``to_grid(c)[j] = sum_G c(G) exp(2 pi i m.j/N)``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxdft.structure.lattice import Lattice


def _smooth_235(n: int) -> int:
    """Smallest integer >= n whose prime factors are 2, 3 and 5."""
    while True:
        m = n
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def rule_fft_shape(lattice: Lattice, ecut: float) -> tuple[int, int, int]:
    """The minimal grid on which no difference G - G' of basis vectors wraps round:
    N_i >= 2 floor(2 Gmax |a_i| / 2 pi) + 1 (Cauchy-Schwarz), rounded up to 2-3-5."""
    gmax = math.sqrt(2.0 * ecut)
    norms = np.linalg.norm(np.asarray(lattice.vectors), axis=1)
    return tuple(_smooth_235(2 * math.floor(2 * gmax * float(a) / (2 * math.pi)) + 1) for a in norms)


def grid_miller(fft_shape) -> jax.Array:
    """Integer Miller indices of the grid in FFT order, shape (*fft_shape, 3)."""
    axes = [np.fft.fftfreq(n, 1.0 / n).astype(np.int64) for n in fft_shape]
    return jnp.asarray(np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1))


def nyquist_mask(fft_shape) -> jax.Array:
    """False on the Nyquist planes (index N/2 of an even axis), True elsewhere. Those
    planes hold -N/2 but not +N/2, so an analytic coefficient there has no conjugate
    partner (theory post 2a, 'Nyquist planes')."""
    keep = np.ones(fft_shape, dtype=bool)
    for axis, n in enumerate(fft_shape):
        if n % 2 == 0:
            index = [slice(None)] * 3
            index[axis] = n // 2
            keep[tuple(index)] = False
    return jnp.asarray(keep)


@jax.tree_util.register_pytree_node_class
class PlaneWaveBasis:
    """Plane-wave bases B_k = {G : 1/2 |k+G|^2 <= ecut} for a batch of k-points.

    Leaves (leading k axis in the batched form, absent for a single k-point):
        miller:  (n_k, n_pad, 3) integer Miller indices m, with G = m B.
        mask:    (n_k, n_pad) True for real basis functions, False for padding.
        kpoints: (n_k, 3) fractional k-points.
    Static: ``ecut`` (Ha), ``n_pad`` and ``fft_shape``.

    Real entries come first in each row. Padding entries have Miller index 0 and
    carry no physical meaning; the Hamiltonian decouples them.
    """

    __slots__ = ("miller", "mask", "kpoints", "ecut", "n_pad", "fft_shape")

    def __init__(self, miller, mask, kpoints, ecut: float, n_pad: int, fft_shape):
        values = (miller, mask, kpoints, float(ecut), int(n_pad), tuple(int(n) for n in fft_shape))
        for name, value in zip(self.__slots__, values):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("PlaneWaveBasis is immutable")

    def __repr__(self):
        return (
            f"PlaneWaveBasis(n_k={self.kpoints.shape[:-1]}, n_pad={self.n_pad}, ecut={self.ecut}, "
            f"fft_shape={self.fft_shape})"
        )

    @property
    def n_pw(self) -> jax.Array:
        """Number of real plane waves per k-point."""
        return self.mask.sum(axis=-1)

    def __getitem__(self, i) -> PlaneWaveBasis:
        """Slice the k axis; an integer index gives a single-k basis."""
        return jax.tree_util.tree_map(lambda x: x[i], self)

    # --- grid transforms (single-k basis) ------------------------------------------------

    def _grid_index(self):
        if self.miller.ndim != 2:
            raise ValueError("grid transforms need a single-k basis; index the k axis first")
        n = jnp.asarray(self.fft_shape)
        idx = jnp.mod(self.miller, n)
        return idx[:, 0], idx[:, 1], idx[:, 2]

    def to_grid(self, c: jax.Array) -> jax.Array:
        """Grid values sum_G c(G) exp(2 pi i m.j/N) of coefficients c with shape (n_pad,)
        or (n_pad, n), giving (*fft_shape) or (*fft_shape, n). Padding is ignored."""
        i, j, k = self._grid_index()
        c = jnp.asarray(c)
        mask = self.mask if c.ndim == 1 else self.mask[:, None]
        grid = jnp.zeros((*self.fft_shape, *c.shape[1:]), dtype=jnp.complex128)
        grid = grid.at[i, j, k].add(jnp.where(mask, c, 0))
        return jnp.fft.ifftn(grid, axes=(0, 1, 2)) * math.prod(self.fft_shape)

    def from_grid(self, f: jax.Array) -> jax.Array:
        """Basis coefficients (1/N) sum_j f(j) exp(-2 pi i m.j/N) of grid values f with
        shape (*fft_shape) or (*fft_shape, n); zero on padding. Inverse of ``to_grid``."""
        i, j, k = self._grid_index()
        f = jnp.asarray(f)
        coeffs = jnp.fft.fftn(f, axes=(0, 1, 2))[i, j, k] / math.prod(self.fft_shape)
        mask = self.mask if f.ndim == 3 else self.mask[:, None]
        return jnp.where(mask, coeffs, 0)

    def tree_flatten(self):
        return (self.miller, self.mask, self.kpoints), (self.ecut, self.n_pad, self.fft_shape)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, value in zip(cls.__slots__, (*children, *aux)):
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    def build(
        cls,
        lattice: Lattice,
        kpoints,
        ecut: float,
        n_pad: int | None = None,
        fft_shape=None,
        allow_aliasing: bool = False,
    ) -> PlaneWaveBasis:
        """Enumerate B_k for each fractional k-point (host side, before tracing).

        The integer search box |m_i| <= ceil((Gmax + |k|) |a_i| / 2 pi) is provably
        complete for any cell (G . a_i = 2 pi m_i and Cauchy-Schwarz).
        ``n_pad`` defaults to max_k N_PW; a smaller value raises ValueError.

        ``fft_shape`` defaults to ``rule_fft_shape``. An explicit shape may be larger.
        A shape below the rule aliases the density and raises unless
        ``allow_aliasing=True`` (the grid-rule positive control). A shape that cannot
        hold the basis itself (distinct basis vectors on one grid point) always raises.
        """
        kpoints = jnp.asarray(kpoints, dtype=jnp.float64)
        if kpoints.ndim != 2 or kpoints.shape[1] != 3:
            raise ValueError(f"kpoints must have shape (n_k, 3), got {kpoints.shape}")
        if not ecut > 0:
            raise ValueError(f"ecut must be positive, got {ecut}")

        B = lattice.reciprocal
        gmax = jnp.sqrt(2.0 * ecut)
        kmax = jnp.max(jnp.linalg.norm(kpoints @ B, axis=1))
        bound = jnp.ceil((gmax + kmax) * jnp.linalg.norm(lattice.vectors, axis=1) / (2 * jnp.pi)).astype(int)
        axes = [jnp.arange(-int(n), int(n) + 1) for n in bound]
        box = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        q = (kpoints[:, None, :] + box[None, :, :]) @ B  # (n_k, n_box, 3)
        inside = 0.5 * jnp.sum(q**2, axis=-1) <= ecut
        n_pw = inside.sum(axis=1)
        n_max = int(n_pw.max())
        if n_pad is None:
            n_pad = n_max
        if n_pad < n_max:
            raise ValueError(f"n_pad={n_pad} is smaller than the largest basis (N_PW={n_max})")

        # Real entries first, in box order (stable), then truncate/extend to n_pad.
        order = jnp.argsort(~inside, axis=1, stable=True)[:, :n_pad]
        miller = jnp.take(box, order, axis=0)
        mask = jnp.take_along_axis(inside, order, axis=1)
        if n_pad > order.shape[1]:
            extra = n_pad - order.shape[1]
            miller = jnp.pad(miller, ((0, 0), (0, extra), (0, 0)))
            mask = jnp.pad(mask, ((0, 0), (0, extra)))
        miller = jnp.where(mask[..., None], miller, 0)

        rule = rule_fft_shape(lattice, ecut)
        shape = rule if fft_shape is None else tuple(int(n) for n in fft_shape)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f"fft_shape must be three positive integers, got {fft_shape}")
        span = np.zeros(3, dtype=int)
        for m_k, mask_k in zip(np.asarray(miller), np.asarray(mask)):
            m = m_k[mask_k]
            span = np.maximum(span, m.max(axis=0) - m.min(axis=0))
        if np.any(np.asarray(shape) <= span):
            raise ValueError(f"fft_shape {shape} cannot hold the basis (Miller span {tuple(span)})")
        if any(n < r for n, r in zip(shape, rule)) and not allow_aliasing:
            raise ValueError(
                f"fft_shape {shape} is below the no-wraparound rule {rule}: the density would alias. "
                "Pass allow_aliasing=True only to test the rule itself."
            )
        return cls(miller, mask, kpoints, float(ecut), n_pad, shape)
