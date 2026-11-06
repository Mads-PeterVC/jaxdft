"""Cohen-Bergstresser (1966) empirical local pseudopotential (theory post 1b).

The paper tabulates V^S(|G|^2) in Ry on the shells |G|^2 = 3, 8, 11 (2 pi / a)^2 of the
diamond lattice at one lattice constant. With V^S = v_hat / Omega_at (the paper's
V_1(G) = (2 / Omega) int V_1(r) e^{-iG.r} d^3r), the atomic transform on a shell is
v_hat = V^S * Omega_at(a_CB).

The model is defined only at its own cell, for two reasons: v_hat is known at three
physical q only, and V^S carries 1/Omega. ``validate`` therefore compares physical q
(Bohr^-1) against the tabulated shells. Shell |G|^2 = 4 was never fitted (in diamond its
structure factor vanishes), so it counts as untabulated.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdft.units import convert

SHELL_RTOL = 1e-8  # relative tolerance on q^2 when matching a tabulated shell


class CohenBergstresser:
    """Local-only Cohen-Bergstresser pseudopotential for a homopolar diamond crystal.

    An empirical model without point ions: ``ionic_charge`` is None, and the G = 0
    limit ``alpha`` is 0 (the V0 = 0 convention)."""

    ionic_charge = None
    alpha = 0.0

    def __init__(self, lattice_constant_angstrom: float, form_factors_ry: dict[int, float]):
        # Publication units, stored as published.
        self.lattice_constant_angstrom = lattice_constant_angstrom
        self.form_factors_ry = dict(form_factors_ry)
        # Atomic units, converted once.
        a = float(convert(lattice_constant_angstrom, "Angstrom", "Bohr"))
        self.lattice_constant = a
        shells = sorted(self.form_factors_ry)
        t = 2 * math.pi / a
        self.shell_q = jnp.sqrt(jnp.asarray(shells, dtype=jnp.float64)) * t
        omega_at = a**3 / 8  # diamond: Omega / 2 with Omega = a^3 / 4
        self._v_hat = jnp.asarray(
            [float(convert(self.form_factors_ry[s], "Rydberg", "Hartree")) * omega_at for s in shells],
            dtype=jnp.float64,
        )
        # Between the last tabulated shell and the next lattice shell (|G|^2 = 12).
        self.local_support = math.sqrt(shells[-1] + 0.5) * t

    def __repr__(self):
        return f"CohenBergstresser(a={self.lattice_constant_angstrom} Å, V_S={self.form_factors_ry} Ry)"

    def _on_shell(self, q):
        q2 = jnp.asarray(q, dtype=jnp.float64)[..., None] ** 2
        s2 = self.shell_q**2
        return jnp.abs(q2 - s2) <= SHELL_RTOL * s2  # (..., n_shells)

    def local(self, q) -> jax.Array:
        """v_hat(q) in Ha Bohr^3: tabulated on the shells, 0 elsewhere (including q = 0,
        the V_0 = 0 convention). Off-table q are refused by ``validate``, not here, so
        that this stays traceable and NaN-free."""
        return jnp.sum(jnp.where(self._on_shell(q), self._v_hat, 0.0), axis=-1)

    def validate(self, q_values) -> None:
        q = jnp.atleast_1d(jnp.asarray(q_values, dtype=jnp.float64))
        used = (q > SHELL_RTOL * self.shell_q[0]) & (q <= self.local_support)
        bad = used & ~jnp.any(self._on_shell(q), axis=-1)
        if bool(jnp.any(bad)):
            t2 = (2 * math.pi / self.lattice_constant) ** 2
            g2 = sorted({round(float(x) ** 2 / t2, 6) for x in q[bad]})
            raise ValueError(
                "Cohen-Bergstresser form factors are only defined on the tabulated shells "
                f"|G|^2 = {sorted(self.form_factors_ry)} (2 pi / a_CB)^2 at a_CB = "
                f"{self.lattice_constant_angstrom} Å; this structure needs |G|^2 = {g2[:6]} "
                "(in the same units). The model is fixed-cell: see theory post 1b."
            )


def silicon() -> CohenBergstresser:
    """Si: a = 5.43 Å, V^S_3 = -0.21, V^S_8 = +0.04, V^S_11 = +0.08 Ry
    (Cohen & Bergstresser 1966, Tables I and II)."""
    return CohenBergstresser(5.43, {3: -0.21, 8: 0.04, 11: 0.08})
