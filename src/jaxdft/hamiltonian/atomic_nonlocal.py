"""Separable nonlocal pseudopotential as an operator term (theory post 2b, 'Matrix
elements and the operator', 'The angular factor at k + G = 0').

    V_nl = B D B^dagger,
    B_{G,(a l m i)} = Omega^(-1/2) exp(-i G.tau_a) S_lm(q) p_hat_i^l(q) / q^l,   q = k + G
    D = block-diagonal, lambda_a h^l_ij for each atom a, channel l and m

S_lm are the real solid harmonics q^l Y_lm(q^), polynomials in the Cartesian
components of q, and the reduced projectors are functions of q^2 (the pseudopotential
protocol). Nothing here forms |q| or q^, so k + G = 0 needs no special case and every
derivative is finite. The factors i^l and exp(i k.tau) cancel between bra and ket.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.hamiltonian.atomic_local import check_species
from jaxdft.structure.structure import Structure


def solid_harmonics(l: int, q: jax.Array) -> jax.Array:
    """Real solid harmonics q^l Y_lm(q^), shape (2l + 1, *q.shape[:-1]), orthonormal
    on the sphere, so that sum_m S_lm(q) S_lm(q') = (2l+1)/(4 pi) q^l q'^l P_l(q^.q'^)."""
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    if l == 0:
        return jnp.full((1, *x.shape), 1.0 / math.sqrt(4 * math.pi))
    if l == 1:
        return math.sqrt(3 / (4 * math.pi)) * jnp.stack([x, y, z])
    if l == 2:
        c = math.sqrt(15 / (4 * math.pi))
        return jnp.stack(
            [
                c * x * y,
                c * y * z,
                c * x * z,
                0.5 * c * (x**2 - y**2),
                0.5 * math.sqrt(5 / (4 * math.pi)) * (2 * z**2 - x**2 - y**2),
            ]
        )
    raise NotImplementedError(f"solid harmonics for l = {l}")


def _channels(pseudo) -> tuple:
    return tuple(getattr(pseudo, "channels", ()) or ())


class _PreparedNonlocal:
    """V_nl at one k-point: psi -> B (D (B^dagger psi))."""

    def __init__(self, B, D):
        self.B = B
        self.D = D

    def apply(self, psi):
        return self.B @ (self.D @ (jnp.conj(self.B).T @ psi))


jax.tree_util.register_pytree_node(
    _PreparedNonlocal,
    lambda p: ((p.B, p.D), None),
    lambda _aux, children: _PreparedNonlocal(*children),
)


@jax.tree_util.register_pytree_node_class
class AtomicNonlocal:
    """Nonlocal pseudopotential operator term. ``pseudos`` maps species labels to
    pseudopotentials; those without ``channels`` contribute nothing."""

    def __init__(self, pseudos: dict):
        self.pseudos = dict(pseudos)

    def tree_flatten(self):
        return (), tuple(sorted(self.pseudos.items(), key=lambda kv: kv[0]))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(dict(aux))

    def __repr__(self):
        return f"AtomicNonlocal({self.pseudos})"

    def bind(self, structure: Structure) -> AtomicNonlocal:
        check_species(self.pseudos, structure)
        return self

    def n_projectors(self, structure: Structure) -> int:
        """Number of projector functions (a, l, m, i) in the cell."""
        return sum(
            (2 * ch.l + 1) * ch.n_projectors for s in structure.species for ch in _channels(self.pseudos[s])
        )

    def prepare(self, structure: Structure, basis_k: PlaneWaveBasis) -> _PreparedNonlocal:
        check_species(self.pseudos, structure)
        q = (basis_k.kpoints + basis_k.miller) @ structure.lattice.reciprocal  # (n_pad, 3)
        q2 = jnp.sum(q**2, axis=-1)
        scale = 1.0 / jnp.sqrt(structure.lattice.volume)
        columns, blocks = [], []
        for a, species in enumerate(structure.species):
            channels = _channels(self.pseudos[species])
            if not channels:
                continue
            phase = jnp.exp(-2j * jnp.pi * (basis_k.miller @ structure.positions[a]))
            weight = structure.weights[a]
            for ch in channels:
                radial = ch.projectors(q2)  # (n_i, n_pad)
                angular = solid_harmonics(ch.l, q)  # (2l+1, n_pad)
                h = jnp.asarray(ch.h, dtype=jnp.float64)
                for m in range(2 * ch.l + 1):
                    columns.append((phase * angular[m])[None, :] * radial)  # (n_i, n_pad)
                    blocks.append(weight * h)
        n_pad = basis_k.miller.shape[0]
        if not columns:
            return _PreparedNonlocal(jnp.zeros((n_pad, 0), dtype=jnp.complex128), jnp.zeros((0, 0)))
        B = scale * jnp.concatenate(columns, axis=0).T.astype(jnp.complex128)  # (n_pad, n_proj)
        B = jnp.where(basis_k.mask[:, None], B, 0)
        D = jax.scipy.linalg.block_diag(*blocks).astype(jnp.complex128)
        return _PreparedNonlocal(B, D)
