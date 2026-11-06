"""Electron densities on the FFT grid (theory post 2a, 'The FFT grid', 'Vacuum')."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.structure.lattice import Lattice


def electron_density(lattice: Lattice, basis: PlaneWaveBasis, orbitals, occupations, kweights=None) -> jax.Array:
    """rho(r_j) = (1/Omega) sum_k w_k sum_n f_nk |sum_G c_nk(G) e^{iG.r_j}|^2, real, shape
    ``basis.fft_shape``.

    ``orbitals`` (n_k, n_pad, n_bands), ``occupations`` (n_k, n_bands) including spin,
    ``kweights`` (n_k,) summing to 1 (uniform by default). With the no-wraparound grid
    this is exact: no aliasing.
    """
    n_k = orbitals.shape[0]
    w = jnp.full(n_k, 1.0 / n_k) if kweights is None else jnp.asarray(kweights, dtype=jnp.float64)

    def per_k(basis_k, c, f):
        psi = basis_k.to_grid(c)  # (*N, n_bands)
        return jnp.sum(f * jnp.abs(psi) ** 2, axis=-1)

    rho_k = jax.vmap(per_k)(basis, orbitals, occupations)
    return jnp.tensordot(w, rho_k, axes=1) / lattice.volume


def clip_and_renormalise(rho: jax.Array, n_electrons, volume) -> jax.Array:
    """Clip negative grid densities (from mixing) to zero and rescale so that
    int rho = n_electrons. Keeps every SCF iterate neutral (review 2.9)."""
    rho = jnp.maximum(rho, 0.0)
    integral = volume / rho.size * jnp.sum(rho)
    return rho * (n_electrons / integral)
