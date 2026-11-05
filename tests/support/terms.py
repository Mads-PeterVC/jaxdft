"""Hamiltonian terms defined outside the package, following the term protocol.

They exercise the extension point that 1b's local potential will use: a term is a
pytree with ``prepare(lattice, basis_k)``, returning a pytree with ``apply(psi)``.
"""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _DenseApply:
    matrix: jax.Array

    def apply(self, psi):
        return self.matrix @ psi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _ScalarApply:
    value: jax.Array

    def apply(self, psi):
        return self.value * psi


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ConstantShift:
    """V(r) = value everywhere: shifts every eigenvalue by ``value`` (Ha)."""

    value: float

    def prepare(self, lattice, basis_k):
        return _ScalarApply(jnp.asarray(self.value, dtype=jnp.float64))


def miller_coupling(d: np.ndarray, strength: float, width: float, phase: np.ndarray):
    """Hermitian coupling f(d) with f(-d) = conj f(d), for Miller differences d (..., 3)."""
    d = np.asarray(d, dtype=float)
    return strength * np.exp(-np.sum(d**2, -1) / (2 * width**2)) * np.exp(1j * d @ phase)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MillerCoupling:
    """A k-independent coupling <G|V|G'> = f(m - m'), like a local potential.

    Couples every pair of basis functions, so padding entries (whose Miller
    indices are arbitrary) would leak into the physical block unless the package
    decouples them centrally. Not physical: it exists to make H non-diagonal.

    Default strength and width give a Gershgorin bound on its norm of about
    0.01 * (2 pi)^{3/2} ~ 0.16 Ha, so it perturbs the kinetic spectrum by much more
    than the eigenvalue tolerance but stays small next to the cutoff.
    """

    strength: float = field(default=0.01, metadata=dict(static=True))
    width: float = field(default=1.0, metadata=dict(static=True))
    phase: tuple = field(default=(0.3, -0.7, 1.1), metadata=dict(static=True))

    def prepare(self, lattice, basis_k):
        m = basis_k.miller
        d = (m[:, None, :] - m[None, :, :]).astype(jnp.float64)
        phase = jnp.asarray(self.phase, dtype=jnp.float64)
        f = self.strength * jnp.exp(-jnp.sum(d**2, -1) / (2 * self.width**2)) * jnp.exp(1j * d @ phase)
        return _DenseApply(f)

    def reference_matrix(self, millers: np.ndarray) -> np.ndarray:
        """The same coupling on an explicit list of Miller indices, in NumPy."""
        d = millers[:, None, :] - millers[None, :, :]
        return miller_coupling(d, self.strength, self.width, np.asarray(self.phase))
