"""Test-only pseudopotentials following the protocol (theory post 1b; CLAUDE.md):
``local(q) -> v_hat(q)`` in Ha Bohr^3, ``local_support`` in Bohr^-1 (or None), and an
optional ``validate(q_values)``."""

import jax.numpy as jnp


class GaussianPseudo:
    """v_hat(q) = -depth * exp(-(q w)^2 / 2), cut to zero beyond ``support``.

    Smooth, isotropic and defined at every q, so it can be used in any cell and with any
    weights. Not physical.
    """

    def __init__(self, depth=5.0, width=0.6, support=3.0):
        self.depth, self.width, self.local_support = depth, width, support

    def local(self, q):
        q = jnp.asarray(q)
        return jnp.where(q <= self.local_support, -self.depth * jnp.exp(-0.5 * (q * self.width) ** 2), 0.0)


class UnboundedPseudo:
    """A local part with no finite support, like a real -4 pi Z / q^2 tail."""

    local_support = None

    def local(self, q):
        q = jnp.asarray(q)
        return jnp.where(q > 0, -4 * jnp.pi / jnp.where(q > 0, q, 1.0) ** 2, 0.0)
