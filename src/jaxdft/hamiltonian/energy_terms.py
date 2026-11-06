"""Energy terms: density functionals whose potential is the autodiff derivative of
their energy (engineering log, 2a API; theory post 2a, 'The local term moves to real
space').

An energy term is a pytree with ``energy(structure, basis, density) -> E`` (Ha), where
``density`` is real on ``basis.fft_shape``. Its potential on the grid is

    V(r_j) = (N_tot / Omega) dE / d rho_j,

because the discrete energies integrate with weight Omega / N_tot per grid point. The
potential is therefore the exact derivative of the discrete energy: Janak's theorem,
the G = 0 terms in H and direct = double-counting energies hold by construction.
A term whose energy does not depend on the density sets ``density_dependent = False``.

Roles. The SCF finds the terms it needs by what they declare, never by class name:

- An *ionic* term carries the point-ion charges: ``ionic_charges(structure)`` (lambda_j
  Z_j, traceable), ``static_ionic_charges(structure)`` (Z_j as Python floats, host
  side, for array shapes) and ``is_ionic`` (False when its models have no point ions,
  e.g. empirical pseudopotentials). It fixes the electron count.
- An *ion-ion* term (``ion_ion = True``) supplies the ion-ion energy, without which
  the total is not the energy of a neutral cell.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis, grid_miller
from jaxdft.structure.structure import Structure


def is_ionic_term(term) -> bool:
    return hasattr(term, "ionic_charges") and getattr(term, "is_ionic", True)


def is_ion_ion_term(term) -> bool:
    return bool(getattr(term, "ion_ion", False))


def is_energy_term(term) -> bool:
    return hasattr(term, "energy") and not hasattr(term, "prepare")


def potential_terms(terms) -> tuple:
    return tuple(t for t in terms if is_energy_term(t) and getattr(t, "density_dependent", True))


def total_potential(terms, structure: Structure, basis: PlaneWaveBasis, density) -> jax.Array:
    """Sum of the potentials of the density-dependent energy terms in ``terms``, on the
    grid (Ha). Zero if there are none. ``density=None`` means a zero density."""
    shape = basis.fft_shape
    rho = jnp.zeros(shape) if density is None else jnp.asarray(density, dtype=jnp.float64)
    active = potential_terms(terms)
    if not active:
        return jnp.zeros(shape)
    grad = jax.grad(lambda r: sum(t.energy(structure, basis, r) for t in active))(rho)
    return grad * (math.prod(shape) / structure.lattice.volume)


def term_potential(term, structure: Structure, basis: PlaneWaveBasis, density) -> jax.Array:
    """V(r) of one energy term at ``density`` (real, float64, shape ``fft_shape``)."""
    return total_potential((term,), structure, basis, density)


def density_coefficients(density) -> jax.Array:
    """rho~(G) = (1/Omega) int rho e^{-iG.r} = FFT[rho] / N_tot, in FFT order."""
    return jnp.fft.fftn(density) / density.size


def grid_g2(structure: Structure, basis: PlaneWaveBasis) -> jax.Array:
    """|G|^2 on the grid, in FFT order (differentiable in the cell)."""
    G = grid_miller(basis.fft_shape) @ structure.lattice.reciprocal
    return jnp.sum(G**2, axis=-1)
