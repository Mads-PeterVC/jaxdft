"""A mixer that is not Anderson, for testing that the solver and the SCF accept any
``Mixer`` (jaxdft.scf.mixing): the protocol must not be Anderson-shaped."""

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class AdaptiveLinear:
    """Linear mixing x <- x + beta r with adaptive damping: beta grows by ``grow``
    when ||r|| falls and is halved when it rises, within [beta_min, beta_max]. Its
    state (beta, previous ||r||) is unlike Anderson's history."""

    beta: float = 0.3
    grow: float = 1.2
    beta_min: float = 0.05
    beta_max: float = 0.9

    def init(self, x0):
        return jnp.asarray(self.beta, x0.dtype), jnp.asarray(jnp.inf, x0.dtype)

    def update(self, state, x, r):
        beta, prev = state
        rn = jnp.sqrt(jnp.sum(r**2))
        beta = jnp.where(rn < prev, beta * self.grow, beta * 0.5)
        beta = jnp.clip(beta, self.beta_min, self.beta_max)
        return x + beta * r, (beta, rn)
