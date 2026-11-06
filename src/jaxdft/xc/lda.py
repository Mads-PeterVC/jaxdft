"""Spin-unpolarised LDA: Slater exchange and Perdew-Wang 92 correlation (libxc
``lda_x`` + ``lda_c_pw``; theory post 2a).

A functional is defined by its energy per electron eps(rho). Calling it returns the
energy density rho eps and the potential d(rho eps)/d rho, the latter by autodiff, so
the two cannot disagree. Every function is finite, with finite derivatives, at rho = 0
(vacuum): the double-where guard keeps NaN out of masked branches.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

# PW92 Table I, unpolarised (p = 1), Hartree. lda_c_pw, not lda_c_pw_mod.
PW92_UNPOLARISED = {
    "A": 0.031091,
    "alpha1": 0.21370,
    "beta1": 7.5957,
    "beta2": 3.5876,
    "beta3": 1.6382,
    "beta4": 0.49294,
}

_CX = -0.75 * (3.0 / math.pi) ** (1.0 / 3.0)
# Below this density the functional is set to zero. At 1e-300 the derivative of
# r_s ~ rho^(-1/3) overflows; at 1e-40 everything is finite and rho eps ~ 1e-53 Ha/Bohr^3.
DENSITY_THRESHOLD = 1e-40


def _exchange(rho):
    return _CX * rho ** (1.0 / 3.0)


def _pw92(rho):
    p = PW92_UNPOLARISED
    rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    den = 2 * p["A"] * (p["beta1"] * jnp.sqrt(rs) + p["beta2"] * rs + p["beta3"] * rs**1.5 + p["beta4"] * rs**2)
    return -2 * p["A"] * (1 + p["alpha1"] * rs) * jnp.log1p(1.0 / den)


@dataclass(frozen=True)
class LDA:
    """Local density approximation with Slater exchange and, optionally, PW92
    correlation."""

    correlation: bool = True

    def eps(self, rho):
        """Energy per electron (Ha); 0 at and below ``DENSITY_THRESHOLD``."""
        rho = jnp.asarray(rho, dtype=jnp.float64)
        positive = rho > DENSITY_THRESHOLD
        safe = jnp.where(positive, rho, 1.0)
        e = _exchange(safe)
        if self.correlation:
            e = e + _pw92(safe)
        return jnp.where(positive, e, 0.0)

    def energy_density(self, rho):
        rho = jnp.asarray(rho, dtype=jnp.float64)
        return rho * self.eps(rho)

    def __call__(self, rho):
        """(energy density rho eps, potential d(rho eps)/d rho), elementwise."""
        rho = jnp.asarray(rho, dtype=jnp.float64)
        e, vjp = jax.vjp(self.energy_density, rho)
        (v,) = vjp(jnp.ones_like(e))
        return e, v


lda_x = LDA(correlation=False)
lda_pw92 = LDA(correlation=True)
