"""Exchange-correlation energy term (theory post 2a, 'Exchange and correlation').

    E_xc = (Omega / N_tot) sum_j rho_j eps_xc(rho_j)

a grid quadrature: the only inexact term of the discrete functional, and the only
source of the egg-box effect."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Xc:
    """XC energy for a functional with ``energy_density(rho)`` (e.g. ``lda_pw92``)."""

    functional: object = field(metadata=dict(static=True))

    def energy(self, structure, basis, density):
        e = self.functional.energy_density(density)
        return structure.lattice.volume / density.size * jnp.sum(e)
