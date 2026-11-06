"""Hartree energy by FFT Poisson (theory post 2a, 'Electrostatics and the G = 0 terms').

    E_H = (Omega / 2) sum_{G != 0} 4 pi |rho~(G)|^2 / G^2,   V_H(G = 0) = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxdft.hamiltonian.energy_terms import density_coefficients, grid_g2


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Hartree:
    """The classical electrostatic energy of the electrons, with the G = 0 divergence
    dropped (it cancels against the ionic terms for a neutral cell)."""

    def energy(self, structure, basis, density):
        rho_g = density_coefficients(density)
        g2 = grid_g2(structure, basis)
        nonzero = g2 > 0
        kernel = jnp.where(nonzero, 4 * jnp.pi / jnp.where(nonzero, g2, 1.0), 0.0)
        return 0.5 * structure.lattice.volume * jnp.sum(kernel * jnp.abs(rho_g) ** 2)
