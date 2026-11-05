"""Plane-wave basis with a k-dependent kinetic-energy cutoff
(theory post 1a, 'Kinetic-energy cutoff', 'Enumerating the basis', 'Batching over k-points').

The basis is purely combinatorial: integer Miller indices and a mask per k-point.
Cartesian k+G and kinetic energies are computed from a Lattice where needed, so
the dependence of energies on the cell stays differentiable.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdft.structure.lattice import Lattice


@jax.tree_util.register_pytree_node_class
class PlaneWaveBasis:
    """Plane-wave bases B_k = {G : 1/2 |k+G|^2 <= ecut} for a batch of k-points.

    Leaves (leading k axis in the batched form, absent for a single k-point):
        miller:  (n_k, n_pad, 3) integer Miller indices m, with G = m B.
        mask:    (n_k, n_pad) True for real basis functions, False for padding.
        kpoints: (n_k, 3) fractional k-points.
    Static: ``ecut`` (Ha) and ``n_pad``.

    Real entries come first in each row. Padding entries have Miller index 0 and
    carry no physical meaning; the Hamiltonian decouples them.
    """

    __slots__ = ("miller", "mask", "kpoints", "ecut", "n_pad")

    def __init__(self, miller, mask, kpoints, ecut: float, n_pad: int):
        for name, value in zip(self.__slots__, (miller, mask, kpoints, float(ecut), int(n_pad))):
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("PlaneWaveBasis is immutable")

    def __repr__(self):
        return f"PlaneWaveBasis(n_k={self.kpoints.shape[:-1]}, n_pad={self.n_pad}, ecut={self.ecut})"

    @property
    def n_pw(self) -> jax.Array:
        """Number of real plane waves per k-point."""
        return self.mask.sum(axis=-1)

    def __getitem__(self, i) -> PlaneWaveBasis:
        """Slice the k axis; an integer index gives a single-k basis."""
        return jax.tree_util.tree_map(lambda x: x[i], self)

    def tree_flatten(self):
        return (self.miller, self.mask, self.kpoints), (self.ecut, self.n_pad)

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        for name, value in zip(cls.__slots__, (*children, *aux)):
            object.__setattr__(obj, name, value)
        return obj

    @classmethod
    def build(cls, lattice: Lattice, kpoints, ecut: float, n_pad: int | None = None) -> PlaneWaveBasis:
        """Enumerate B_k for each fractional k-point (host side, before tracing).

        The integer search box |m_i| <= ceil((Gmax + |k|) |a_i| / 2 pi) is provably
        complete for any cell (G . a_i = 2 pi m_i and Cauchy-Schwarz).
        ``n_pad`` defaults to max_k N_PW; a smaller value raises ValueError.
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
        return cls(miller, mask, kpoints, float(ecut), n_pad)
