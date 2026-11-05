"""Folded free-electron bands (theory post 1a, 'The folded free-electron bands')."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.structure.lattice import Lattice


def free_electron_bands(lattice: Lattice, kpoints, n_bands: int) -> jax.Array:
    """The ``n_bands`` lowest values of 1/2 |k+G|^2 over all G, per fractional k (Ha).

    Enumerates the cutoff sphere, growing it until every k-point has at least
    ``n_bands`` plane waves. Every value inside the sphere is included, so the lowest
    ``n_bands`` are exact. Shape (n_k, n_bands).
    """
    kpoints = jnp.asarray(kpoints, dtype=jnp.float64)
    # Start from the cutoff whose asymptotic basis size is about 2 * n_bands.
    ecut = 0.5 * float((12 * jnp.pi**2 * n_bands / lattice.volume) ** (2 / 3))
    while True:
        basis = PlaneWaveBasis.build(lattice, kpoints, ecut)
        if int(jnp.min(basis.n_pw)) >= n_bands:
            break
        ecut *= 2.0
    q = (basis.kpoints[:, None, :] + basis.miller) @ lattice.reciprocal
    energies = jnp.where(basis.mask, 0.5 * jnp.sum(q**2, axis=-1), jnp.inf)
    return jnp.sort(energies, axis=-1)[:, :n_bands]
