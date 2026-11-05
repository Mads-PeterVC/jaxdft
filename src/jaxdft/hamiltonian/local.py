"""Local pseudopotential term, applied matrix-free (theory post 1b, 'The Hamiltonian
matrix').

    V(G) = (1 / Omega) sum_j w_j exp(-2 pi i m.f_j) v_hat_{s(j)}(|G|),   G = m B

<k+G|V|k+G'> = V(G - G') couples a basis function only to those offset by one of the
K coupling vectors d with |d B| <= local_support (K = 44 for Cohen-Bergstresser Si,
independent of the cutoff). ``bind`` enumerates them on the host; ``prepare`` finds,
for every basis entry, the index of m - d by a jittable search on sorted integer keys,
and ``apply`` is a gather and a sum: O(K N) memory and work per band.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.structure.structure import Structure

# Miller indices are encoded as integer keys (m + R) . (1, 2R+1, (2R+1)^2); |m_i| < R.
_KEY_RADIUS = 2**19
_KEY_ABSENT = jnp.iinfo(jnp.int64).max
# A structure factor counts as non-zero above this, relative to the summed weights.
_STRUCTURE_FACTOR_RTOL = 1e-10


def _keys(miller):
    m = jnp.asarray(miller, dtype=jnp.int64) + _KEY_RADIUS
    width = 2 * _KEY_RADIUS + 1
    return (m[..., 0] * width + m[..., 1]) * width + m[..., 2]


def _safe_norm(G):
    """|G| with a finite gradient at G = 0."""
    g2 = jnp.sum(G**2, axis=-1)
    positive = g2 > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, g2, 1.0)), 0.0)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _PreparedLocal:
    neighbours: jax.Array  # (n_pad, K) index of m_i - d_k in the basis
    present: jax.Array  # (n_pad, K) whether m_i - d_k is in the basis
    values: jax.Array  # (K,) V(d_k), complex128

    def apply(self, psi):
        gathered = jnp.where(self.present[..., None], psi[self.neighbours], 0)  # (n_pad, K, n)
        return jnp.einsum("k,ikn->in", self.values, gathered)


@jax.tree_util.register_pytree_node_class
class LocalPotential:
    """Local pseudopotential term. ``pseudos`` maps species labels to pseudopotentials."""

    def __init__(self, pseudos: dict, couplings=None):
        self.pseudos = dict(pseudos)
        self.couplings = couplings  # (K, 3) int Miller vectors, set by bind()

    def tree_flatten(self):
        children = () if self.couplings is None else (self.couplings,)
        return children, (tuple(sorted(self.pseudos.items(), key=lambda kv: kv[0])), self.couplings is None)

    @classmethod
    def tree_unflatten(cls, aux, children):
        items, unbound = aux
        return cls(dict(items), None if unbound else children[0])

    # --- the potential --------------------------------------------------------------

    def _check_species(self, structure: Structure):
        missing = sorted(set(structure.species) - set(self.pseudos))
        if missing:
            raise ValueError(f"no pseudopotential given for species {missing}")

    def coefficients(self, structure: Structure, miller) -> jax.Array:
        """V(G) for G = m B, in Ha, complex128. Traceable in positions, weights and cell."""
        self._check_species(structure)
        miller = jnp.asarray(miller)
        q = _safe_norm(miller @ structure.lattice.reciprocal)
        total = jnp.zeros(miller.shape[:-1], dtype=jnp.complex128)
        for species, pseudo in self.pseudos.items():
            atoms = [j for j, s in enumerate(structure.species) if s == species]
            if not atoms:
                continue
            f = structure.positions[jnp.asarray(atoms)]
            w = structure.weights[jnp.asarray(atoms)]
            phases = jnp.exp(-2j * jnp.pi * (miller @ f.T))  # (..., n_atoms_of_species)
            total = total + (phases @ w.astype(jnp.complex128)) * pseudo.local(q)
        return total / structure.lattice.volume

    # --- host-side hook ---------------------------------------------------------------

    def _candidates(self, structure: Structure):
        """All Miller vectors with |m B| <= max support (provably complete box)."""
        supports = [self.pseudos[s].local_support for s in set(structure.species)]
        if any(s is None for s in supports):
            raise NotImplementedError(
                "a pseudopotential with unbounded local support needs the FFT-based local "
                "term (SCF milestone); the matrix-free gather handles finite support only"
            )
        q_max = max(supports, default=0.0)
        A = structure.lattice.vectors
        bound = [int(math.ceil(q_max * float(jnp.linalg.norm(a)) / (2 * math.pi))) for a in A]
        axes = [jnp.arange(-n, n + 1) for n in bound]
        box = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        q = _safe_norm(box @ structure.lattice.reciprocal)
        return box[q <= q_max], q[q <= q_max]

    def validate(self, structure: Structure) -> None:
        """Host-side checks: every species has a pseudopotential with finite support, and
        each pseudopotential is defined wherever its species' structure factor is non-zero."""
        self._check_species(structure)
        candidates, q = self._candidates(structure)
        for species, pseudo in self.pseudos.items():
            atoms = [j for j, s in enumerate(structure.species) if s == species]
            if not atoms or not hasattr(pseudo, "validate"):
                continue
            f = structure.positions[jnp.asarray(atoms)]
            w = structure.weights[jnp.asarray(atoms)]
            S = jnp.exp(-2j * jnp.pi * (candidates @ f.T)) @ w.astype(jnp.complex128)
            used = jnp.abs(S) > _STRUCTURE_FACTOR_RTOL * max(1.0, float(jnp.sum(jnp.abs(w))))
            pseudo.validate(q[used])

    def bind(self, structure: Structure) -> LocalPotential:
        """Validate, then fix the coupling vectors: candidates within the support where
        some species' form factor is non-zero. Positions and weights stay free (they
        change V(d), not which d can couple), so the bound term remains differentiable."""
        self.validate(structure)
        candidates, q = self._candidates(structure)
        nonzero = jnp.zeros(q.shape, dtype=bool)
        for species in set(structure.species):
            nonzero = nonzero | (self.pseudos[species].local(q) != 0)
        return LocalPotential(self.pseudos, couplings=candidates[nonzero])

    # --- term protocol ------------------------------------------------------------------

    def prepare(self, structure: Structure, basis_k: PlaneWaveBasis) -> _PreparedLocal:
        term = self
        if term.couplings is None:
            if isinstance(structure.lattice.vectors, jax.core.Tracer):
                raise RuntimeError("LocalPotential must be bound (bind(structure)) before tracing")
            term = self.bind(structure)
        d = term.couplings
        keys = jnp.where(basis_k.mask, _keys(basis_k.miller), _KEY_ABSENT)
        order = jnp.argsort(keys)
        sorted_keys = keys[order]
        target = _keys(basis_k.miller[:, None, :] - d[None, :, :])  # (n_pad, K)
        pos = jnp.clip(jnp.searchsorted(sorted_keys, target), 0, keys.shape[0] - 1)
        present = sorted_keys[pos] == target
        return _PreparedLocal(order[pos], present, term.coefficients(structure, d))
