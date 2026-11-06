"""Ewald ion-ion energy of point charges in a uniform neutralising background
(theory post 2a, 'Electrostatics and the G = 0 terms').

    E = 1/2 sum_{ij} sum'_R q_i q_j erfc(eta |d_ij + R|) / |d_ij + R|
        + (2 pi / Omega) sum_{G != 0} exp(-G^2 / 4 eta^2) / G^2 |sum_j q_j e^{iG.tau_j}|^2
        - eta / sqrt(pi) sum_j q_j^2 - pi / (2 eta^2 Omega) (sum_j q_j)^2

Truncation. The real-space sum runs over the images |n_i| <= n_real_i, after wrapping
fractional differences into [-1/2, 1/2); it contains every vector shorter than
r_R = min_i (n_real_i - 1/2) d_i, with d_i = 2 pi / |b_i| the interplanar spacing. The
reciprocal sum runs over |m_i| <= n_recip_i; it contains the sphere of radius
r_G = min_i 2 pi n_recip_i / |a_i| (Cauchy-Schwarz both ways). Both tails are below
~1e-18 relative when eta r_R >= X and r_G / (2 eta) >= X with X = 6.5.

Array shapes must be static under tracing (jax.grad does not expose concrete values),
so the integer ranges are either computed on the host from a concrete cell, or passed
in (the ``Ewald`` term derives them from the static FFT grid). In the latter case the
coverage conditions are checked under tracing, and the result is NaN if they fail:
the 2a failure convention.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erfc

from jaxdft.hamiltonian.atomic_local import check_species, ionic_charges
from jaxdft.structure.structure import Structure

COVERAGE = 6.5  # erfc(6.5) ~ 4e-20; exp(-6.5^2) ~ 5e-19


def _box(n):
    axes = [np.arange(-int(k), int(k) + 1) for k in n]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def _host_ranges(A: np.ndarray, eta: float):
    B = 2 * math.pi * np.linalg.inv(A).T
    d = 2 * math.pi / np.linalg.norm(B, axis=1)
    a = np.linalg.norm(A, axis=1)
    r_cut = COVERAGE / eta
    g_cut = 2 * COVERAGE * eta
    n_real = tuple(int(math.ceil(r_cut / di + 0.5)) for di in d)
    n_recip = tuple(int(math.ceil(g_cut * ai / (2 * math.pi))) for ai in a)
    return n_real, n_recip


def _coverage(lattice_vectors, n_real, n_recip):
    """(r_R, r_G) for the given static ranges, traced in the cell."""
    A = lattice_vectors
    B = 2 * jnp.pi * jnp.linalg.inv(A).T
    d = 2 * jnp.pi / jnp.linalg.norm(B, axis=1)
    a = jnp.linalg.norm(A, axis=1)
    r_real = jnp.min((jnp.asarray(n_real) - 0.5) * d)
    r_recip = jnp.min(2 * jnp.pi * jnp.asarray(n_recip) / a)
    return r_real, r_recip


def ewald_energy(structure: Structure, charges, eta=None, n_real=None, n_recip=None) -> jax.Array:
    """Ewald energy (Ha) of point charges ``charges`` (n_atoms,) at the structure's
    positions, with a neutralising background. Differentiable in positions, cell and
    charges.

    Without ``n_real``/``n_recip`` the cell must be concrete: ranges are chosen on the
    host for ``eta`` (default sqrt(pi) / Omega^(1/3)). With static ranges, ``eta``
    defaults to the value that balances the two truncations, and an insufficient range
    gives NaN.
    """
    q = jnp.asarray(charges, dtype=jnp.float64)
    A = structure.lattice.vectors
    if n_real is None or n_recip is None:
        if isinstance(A, jax.core.Tracer):
            raise ValueError("ewald_energy needs static n_real/n_recip when the cell is traced")
        A_host = np.asarray(A)
        if eta is None:
            eta = math.sqrt(math.pi) / abs(np.linalg.det(A_host)) ** (1 / 3)
        n_real, n_recip = _host_ranges(A_host, float(eta))
    r_real, r_recip = _coverage(A, n_real, n_recip)
    if eta is None:
        eta = jnp.sqrt(r_recip / (2 * r_real))
    eta = jnp.asarray(eta, dtype=jnp.float64)
    covered = (eta * r_real >= COVERAGE * (1 - 1e-9)) & (r_recip / (2 * eta) >= COVERAGE * (1 - 1e-9))

    volume = structure.lattice.volume
    f = structure.positions
    n_atoms = f.shape[0]
    if n_atoms == 0:
        return jnp.asarray(0.0)

    # Real space.
    box = _box(n_real)
    images = jnp.asarray(box, dtype=jnp.float64)  # (M, 3)
    origin = jnp.asarray(np.all(box == 0, axis=1))
    df = f[:, None, :] - f[None, :, :]
    shift = jnp.round(df)
    df = df - shift
    disp = (df[:, :, None, :] + images[None, None, :, :]) @ A  # (n, n, M, 3)
    r2 = jnp.sum(disp**2, axis=-1)
    # Only an atom and itself in the home cell is excluded; distinct coincident atoms
    # have infinite energy, as they should.
    self_image = jnp.eye(n_atoms, dtype=bool)[:, :, None] & origin[None, None, :]
    r = jnp.sqrt(jnp.where(self_image, 1.0, r2))
    pair = jnp.where(self_image, 0.0, erfc(eta * r) / r)
    e_real = 0.5 * jnp.einsum("i,j,ijm->", q, q, pair)

    # Reciprocal space.
    m = _box(n_recip)
    m = m[np.any(m != 0, axis=1)]
    G = jnp.asarray(m, dtype=jnp.float64) @ structure.lattice.reciprocal
    g2 = jnp.sum(G**2, axis=-1)
    S = jnp.exp(2j * jnp.pi * (jnp.asarray(m, dtype=jnp.float64) @ f.T)) @ q.astype(jnp.complex128)
    e_recip = 2 * jnp.pi / volume * jnp.sum(jnp.exp(-g2 / (4 * eta**2)) / g2 * jnp.abs(S) ** 2)

    e_self = -eta / jnp.sqrt(jnp.pi) * jnp.sum(q**2)
    e_background = -jnp.pi / (2 * eta**2 * volume) * jnp.sum(q) ** 2
    total = e_real + e_recip + e_self + e_background
    return jnp.where(covered, total, jnp.nan)


def grid_ranges(fft_shape, safety: float = 3.0):
    """Static Ewald ranges from an FFT grid: the reciprocal box of the grid, and enough
    real-space images that the balanced eta covers both sums for any cell up to
    ``safety`` times more skewed than a cube (checked under tracing; NaN otherwise)."""
    n_recip = tuple((n - 1) // 2 for n in fft_shape)
    # Cube: r_G r_R = 2 pi n_recip (n_real - 1/2) >= 2 X^2.
    n_real = tuple(int(math.ceil(safety * COVERAGE**2 / (math.pi * max(k, 1)) + 0.5)) for k in n_recip)
    return n_real, n_recip


@jax.tree_util.register_pytree_node_class
class Ewald:
    """Ewald energy term for the point ions of ``pseudos`` (charge lambda_j Z_j).
    Independent of the density."""

    density_dependent = False

    def __init__(self, pseudos: dict):
        self.pseudos = dict(pseudos)

    def tree_flatten(self):
        return (), tuple(sorted(self.pseudos.items(), key=lambda kv: kv[0]))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(dict(aux))

    def __repr__(self):
        return f"Ewald({self.pseudos})"

    def energy(self, structure, basis, density):
        check_species(self.pseudos, structure)
        n_real, n_recip = grid_ranges(basis.fft_shape)
        return ewald_energy(structure, ionic_charges(self.pseudos, structure), n_real=n_real, n_recip=n_recip)
