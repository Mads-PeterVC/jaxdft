"""Local pseudopotential as an energy term, applied in real space through the FFT grid
(theory post 2a, 'The local term moves to real space', 'Nyquist planes',
'Electrostatics and the G = 0 terms').

    V~(G) = (1/Omega) sum_j lambda_j exp(-2 pi i m.f_j) v_hat_j(|G|)      (G != 0)
    V~(0) = (1/Omega) sum_j lambda_j alpha_j
    E_loc = Omega sum_G conj(V~(G)) rho~(G)

Unlike the 1b gather term it accepts form factors of unbounded support. Analytic
coefficients are zeroed on the Nyquist planes of the grid, where they have no
conjugate partner.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxdft.basis.planewave import PlaneWaveBasis, grid_miller, nyquist_mask
from jaxdft.hamiltonian.energy_terms import density_coefficients
from jaxdft.structure.structure import Structure

# A structure factor counts as non-zero above this, relative to the summed weights.
_STRUCTURE_FACTOR_RTOL = 1e-10


def _species_atoms(structure: Structure, species: str):
    return [j for j, s in enumerate(structure.species) if s == species]


def check_species(pseudos: dict, structure: Structure) -> None:
    missing = sorted(set(structure.species) - set(pseudos))
    if missing:
        raise ValueError(f"no pseudopotential given for species {missing}")


def ionic_charges(pseudos: dict, structure: Structure) -> jax.Array:
    """lambda_j Z_j per atom. Raises if a species has no point-ion charge."""
    check_species(pseudos, structure)
    z = []
    for s in structure.species:
        charge = getattr(pseudos[s], "ionic_charge", None)
        if charge is None:
            raise ValueError(f"pseudopotential for {s!r} has no ionic charge (an empirical model?)")
        z.append(float(charge))
    return structure.weights * jnp.asarray(z, dtype=jnp.float64)


@jax.tree_util.register_pytree_node_class
class AtomicLocal:
    """Local pseudopotential energy term. ``pseudos`` maps species labels to
    pseudopotentials (``local``, ``alpha``, optionally ``validate``)."""

    def __init__(self, pseudos: dict):
        self.pseudos = dict(pseudos)

    def tree_flatten(self):
        return (), tuple(sorted(self.pseudos.items(), key=lambda kv: kv[0]))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(dict(aux))

    def __repr__(self):
        return f"AtomicLocal({self.pseudos})"

    @property
    def is_ionic(self) -> bool:
        return all(getattr(p, "ionic_charge", None) is not None for p in self.pseudos.values())

    def ionic_charges(self, structure: Structure) -> jax.Array:
        return ionic_charges(self.pseudos, structure)

    def coefficients(self, structure: Structure, miller) -> jax.Array:
        """V~(G) for G = m B at any Miller indices (complex128, Ha), alpha at G = 0."""
        check_species(self.pseudos, structure)
        miller = jnp.asarray(miller)
        G = miller @ structure.lattice.reciprocal
        g2 = jnp.sum(G**2, axis=-1)
        nonzero = g2 > 0
        q = jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, g2, 1.0)), 0.0)
        total = jnp.zeros(miller.shape[:-1], dtype=jnp.complex128)
        for species, pseudo in self.pseudos.items():
            atoms = _species_atoms(structure, species)
            if not atoms:
                continue
            f = structure.positions[jnp.asarray(atoms)]
            w = structure.weights[jnp.asarray(atoms)]
            phases = jnp.exp(-2j * jnp.pi * (miller @ f.T))
            form = jnp.where(nonzero, pseudo.local(q), getattr(pseudo, "alpha", 0.0))
            total = total + (phases @ w.astype(jnp.complex128)) * form
        return total / structure.lattice.volume

    def grid_coefficients(self, structure: Structure, basis: PlaneWaveBasis) -> jax.Array:
        """V~(G) on the FFT grid (FFT order), zero on the Nyquist planes."""
        v = self.coefficients(structure, grid_miller(basis.fft_shape))
        return jnp.where(nyquist_mask(basis.fft_shape), v, 0)

    def energy(self, structure, basis, density):
        v = self.grid_coefficients(structure, basis)
        return structure.lattice.volume * jnp.real(jnp.sum(jnp.conj(v) * density_coefficients(density)))

    # --- host-side hook -------------------------------------------------------------

    def bind(self, structure: Structure) -> AtomicLocal:
        """Validate (host side): every species has a pseudopotential, and models with a
        ``validate`` hook and finite support are defined wherever their structure factor
        is non-zero within that support."""
        check_species(self.pseudos, structure)
        if isinstance(structure.lattice.vectors, jax.core.Tracer):
            return self
        A = np.asarray(structure.lattice.vectors)
        B = np.asarray(structure.lattice.reciprocal)
        for species, pseudo in self.pseudos.items():
            atoms = _species_atoms(structure, species)
            support = getattr(pseudo, "local_support", None)
            if not atoms or not hasattr(pseudo, "validate") or support is None:
                continue
            bound = [math.ceil(support * float(np.linalg.norm(a)) / (2 * math.pi)) for a in A]
            axes = [np.arange(-n, n + 1) for n in bound]
            box = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            q = np.linalg.norm(box @ B, axis=-1)
            box, q = box[q <= support], q[q <= support]
            f = np.asarray(structure.positions)[atoms]
            w = np.asarray(structure.weights)[atoms]
            S = np.exp(-2j * np.pi * (box @ f.T)) @ w
            used = np.abs(S) > _STRUCTURE_FACTOR_RTOL * max(1.0, float(np.sum(np.abs(w))))
            pseudo.validate(jnp.asarray(q[used]))
        return self
