"""The Kohn-Sham SCF and the differentiable total energy (theory post 2a, 'The total
energy', 'Self-consistency as a fixed point', 'Forces and other derivatives';
engineering log, 2a API).

- ``scf(structure, basis, terms, options)``: host-facing. Validates the setup, runs
  the SCF and returns an ``SCFResult``; raises ``SCFError`` on failure.
- ``energy(structure, basis, terms, options, side)``: a differentiable function of the
  structure. The SCF runs inside on a stop-gradient copy of the structure; the energy
  is then the direct Kohn-Sham functional of the stopped converged orbitals, with the
  density rebuilt from them and the lambda-dependent occupations. By stationarity this
  is the Hellmann-Feynman derivative. Failures give NaN (value and gradient).
- ``total_energy(...)``: the direct functional of given orbitals and occupations.

The SCF never imports an XC functional, pseudopotential or term by name: terms are
either operator terms (``prepare``) or energy terms (``energy``), and the ionic and
ion-ion terms are found by the roles they declare (``jaxdft.hamiltonian.energy_terms``).
The mixer, eigensolver, preconditioner and initial density are ``SCFOptions``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp

from jaxdft.basis.planewave import PlaneWaveBasis
from jaxdft.calculations.bands import bind_terms
from jaxdft.hamiltonian.energy_terms import (
    is_energy_term,
    is_ion_ion_term,
    is_ionic_term,
    total_potential,
)
from jaxdft.hamiltonian.operator import apply_H, hamiltonian
from jaxdft.hamiltonian.preconditioners import diagonal_preconditioner
from jaxdft.scf.density import AtomicGaussians, clip_and_renormalise, electron_density
from jaxdft.scf.mixing import Anderson, Mixer
from jaxdft.scf.occupations import aufbau_occupations
from jaxdft.scf.solve import fixed_point
from jaxdft.solvers.lobpcg import lobpcg
from jaxdft.structure.structure import Structure

# The eigensolver is asked for residuals of EIGEN_TOL_FRACTION * tol and must reach
# EIGEN_CHECK_FRACTION * tol; both are floored near rounding (||H|| eps_mach ~ 1e-14).
EIGEN_TOL_FRACTION = 0.05
EIGEN_CHECK_FRACTION = 0.1
EIGEN_FLOOR = 2e-14


class SCFError(RuntimeError):
    """The SCF did not produce a valid ground state (not converged, eigensolver not
    converged, or HOMO-LUMO gap below ``min_gap``)."""


@dataclass(frozen=True)
class SCFOptions:
    """SCF settings. ``tol`` is on ||F(rho) - rho|| = (int (F(rho) - rho)^2)^(1/2).
    ``n_bands`` defaults to occupied + 1. ``min_gap`` (Ha) guards aufbau filling against
    a degenerate frontier level; it is a numerical criterion, not a physical gap.
    ``mixer`` is any ``Mixer`` (``jaxdft.scf.mixing``). ``preconditioner`` is a
    factory ``(Hamiltonian) -> precondition(r, eps)`` (``jaxdft.hamiltonian
    .preconditioners``), or None for none. ``initial_density(structure, basis,
    charges) -> rho`` gives the starting density when ``density0`` is not passed;
    the default is atomic Gaussians of width 0.8 Bohr."""

    tol: float = 1e-9
    max_iter: int = 100
    n_bands: int | None = None
    mixer: Mixer = field(default_factory=Anderson)
    eigensolver: Callable = lobpcg
    preconditioner: Callable | None = diagonal_preconditioner
    min_gap: float = 1e-3
    initial_density: Callable = field(default_factory=AtomicGaussians)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SCFResult:
    """Converged SCF. ``density`` is built from ``orbitals`` and ``occupations``; the
    orbitals are eigenvectors of H at the last input density. ``energies`` maps term
    names to energies (Ha); ``total_energy`` is their sum. ``eigen_residuals`` are
    ||H psi - eps psi|| per band."""

    density: jax.Array
    orbitals: jax.Array
    eigenvalues: jax.Array
    occupations: jax.Array
    energies: dict
    total_energy: jax.Array
    residual: jax.Array
    n_iter: jax.Array
    converged: jax.Array
    eigen_residuals: jax.Array


# --- the direct energy functional ------------------------------------------------------


def total_energy(structure: Structure, basis: PlaneWaveBasis, terms, orbitals, occupations, kweights=None):
    """Direct Kohn-Sham energy of ``orbitals`` (n_k, n_pad, n_bands) with
    ``occupations`` (n_k, n_bands): (E, {term name: energy}). Operator terms contribute
    sum_k w_k sum_n f_nk <psi_nk|T|psi_nk>; energy terms are evaluated at the density
    of the orbitals."""
    n_k = orbitals.shape[0]
    w = jnp.full(n_k, 1.0 / n_k) if kweights is None else jnp.asarray(kweights, dtype=jnp.float64)
    density = electron_density(structure.lattice, basis, orbitals, occupations, w)
    parts = {}
    for term in terms:
        name = type(term).__name__
        if is_energy_term(term):
            e = term.energy(structure, basis, density)
        else:

            def per_k(basis_k, c, f, term=term):
                c = jnp.where(basis_k.mask[:, None], c, 0)
                tc = jnp.where(basis_k.mask[:, None], term.prepare(structure, basis_k).apply(c), 0)
                return jnp.sum(f * jnp.real(jnp.sum(jnp.conj(c) * tc, axis=0)))

            e = jnp.dot(w, jax.vmap(per_k)(basis, orbitals, occupations))
        parts[name] = parts.get(name, 0.0) + e
    return sum(parts.values()), parts


# --- setup checks and static quantities ----------------------------------------------------


def _check_terms(terms):
    """The ionic term that fixes the electron count; raises if there is none, or if
    no term supplies the ion-ion energy (roles: ``jaxdft.hamiltonian.energy_terms``)."""
    ionic = [t for t in terms if is_ionic_term(t)]
    if not ionic:
        raise ValueError("no term defines ionic charges, so the electron count is unknown")
    if not any(is_ion_ion_term(t) for t in terms):
        raise ValueError(
            "the terms contain ionic pseudopotentials but no ion-ion term (e.g. Ewald): "
            "without the ion-ion energy the total is not the energy of a neutral cell"
        )
    return ionic[0]


# --- the traced SCF ----------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("n_bands", "mixer", "eigensolver", "preconditioner"))
def _scf_core(
    structure, basis, terms, density0, n_el, kweights, tol, max_iter, min_gap, *, n_bands, mixer, eigensolver, preconditioner
):
    volume = structure.lattice.volume
    n_tot = math.prod(basis.fft_shape)
    n_k = basis.miller.shape[0]
    occ = jnp.broadcast_to(aufbau_occupations(n_el, n_bands), (n_k, n_bands))
    eig_tol = jnp.maximum(EIGEN_TOL_FRACTION * tol, EIGEN_FLOOR)

    def diagonalise(potential, X_prev):
        def per_k(basis_k, x0):
            ham = hamiltonian(structure, basis_k, terms, potential=potential)
            precondition = None if preconditioner is None else preconditioner(ham)

            def apply(psi):
                return apply_H(ham, psi)

            eps, vecs = eigensolver(apply, basis_k.n_pad, n_bands, x0=x0, preconditioner=precondition, tol=eig_tol)
            vecs = jnp.where(basis_k.mask[:, None], vecs, 0)
            res = jnp.linalg.norm(apply(vecs) - vecs * eps[None, :], axis=0)
            return vecs, eps, res

        return jax.vmap(per_k)(basis, X_prev)

    def F(rho, aux):
        X_prev = aux[0]
        potential = total_potential(terms, structure, basis, rho)
        X, eps, res = diagonalise(potential, X_prev)
        rho_out = electron_density(structure.lattice, basis, X, occ, kweights)
        return rho_out, (X, eps, res)

    key = jax.random.PRNGKey(0)
    kr, ki = jax.random.split(key)
    shape = (n_k, basis.n_pad, n_bands)
    X0 = jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
    X0 = jnp.where(basis.mask[..., None], X0, 0)
    aux0 = (X0, jnp.zeros((n_k, n_bands)), jnp.zeros((n_k, n_bands)))

    def norm(r):
        return jnp.sqrt(volume / n_tot * jnp.sum(r**2))

    def project(rho):
        return clip_and_renormalise(rho, n_el, volume)

    res = fixed_point(F, density0, mixer, tol, max_iter, project=project, norm=norm, aux0=aux0)
    X, eps, eig_res = res.aux

    top = jnp.sum(occ[0] > 0) - 1  # highest (partially) occupied band
    has_next = top + 1 < n_bands
    nxt = jnp.minimum(top + 1, n_bands - 1)
    gap = jnp.take(eps, nxt, axis=1) - jnp.take(eps, jnp.maximum(top, 0), axis=1)
    gap_ok = has_next & jnp.all(gap > min_gap)
    eig_ok = jnp.all(eig_res < jnp.maximum(EIGEN_CHECK_FRACTION * tol, 5 * EIGEN_FLOOR))
    return {
        "orbitals": X,
        "eigenvalues": eps,
        "occupations": occ,
        "density": res.fx,
        "residual": res.residual,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "eigen_residuals": eig_res,
        "gap_ok": gap_ok,
        "eig_ok": eig_ok,
        "n_el_ok": n_el <= 2.0 * n_bands,
    }


_total_energy_jit = jax.jit(total_energy)


def _run(structure, basis, terms, options, density0, n_el, n_bands, charges):
    if density0 is None:
        density0 = options.initial_density(structure, basis, charges)
    n_k = basis.miller.shape[0]
    kweights = jnp.full(n_k, 1.0 / n_k)
    out = _scf_core(
        structure,
        basis,
        terms,
        jnp.asarray(density0, dtype=jnp.float64),
        n_el,
        kweights,
        jnp.asarray(options.tol, dtype=jnp.float64),
        jnp.asarray(options.max_iter),
        jnp.asarray(options.min_gap, dtype=jnp.float64),
        n_bands=n_bands,
        mixer=options.mixer,
        eigensolver=options.eigensolver,
        preconditioner=options.preconditioner,
    )
    return out, kweights


# --- public entry points -------------------------------------------------------------------------


def scf(
    structure: Structure, basis: PlaneWaveBasis, terms, options: SCFOptions | None = None, density0=None
) -> SCFResult:
    """Run the SCF on the host and return an ``SCFResult``; raise ``SCFError`` with a
    description if it fails."""
    options = SCFOptions() if options is None else options
    terms = bind_terms(structure, tuple(terms))
    ionic = _check_terms(terms)
    charges = ionic.ionic_charges(structure)
    n_el = float(jnp.sum(charges))
    n_bands = options.n_bands or int(math.ceil(n_el / 2 - 1e-12)) + 1
    if n_el > 2 * n_bands:
        raise ValueError(f"{n_el} electrons do not fit in n_bands={n_bands}")
    out, kweights = _run(structure, basis, terms, options, density0, jnp.asarray(n_el), n_bands, charges)

    problems = []
    if not bool(out["converged"]):
        problems.append(
            f"SCF not converged after {int(out['n_iter'])} iterations "
            f"(residual {float(out['residual']):.3e}, tol {options.tol:.1e})"
        )
    if not bool(out["eig_ok"]):
        problems.append(f"eigensolver not converged (max residual {float(jnp.max(out['eigen_residuals'])):.3e})")
    if not bool(out["gap_ok"]):
        problems.append(f"frontier gap below min_gap={options.min_gap} Ha (or no band above the occupied ones)")
    if problems:
        raise SCFError("; ".join(problems))

    e, parts = _total_energy_jit(structure, basis, terms, out["orbitals"], out["occupations"], kweights)
    return SCFResult(
        density=out["density"],
        orbitals=out["orbitals"],
        eigenvalues=out["eigenvalues"],
        occupations=out["occupations"],
        energies=parts,
        total_energy=e,
        residual=out["residual"],
        n_iter=out["n_iter"],
        converged=out["converged"],
        eigen_residuals=out["eigen_residuals"],
    )


def energy(
    structure: Structure,
    basis: PlaneWaveBasis,
    terms,
    options: SCFOptions | None = None,
    side: int = -1,
    density0=None,
) -> jax.Array:
    """Total energy (Ha) as a differentiable function of positions, cell and per-atom
    weights; ``jax.grad`` gives forces, cell gradients and weight derivatives.

    The SCF runs on a stop-gradient copy of ``structure``; the converged orbitals are
    stop-gradiented and the energy is the direct functional of them, with occupations
    depending on the weights through N_el = sum_j lambda_j Z_j. ``side`` picks the
    one-sided derivative at integer N_el: -1 the HOMO side, +1 the LUMO side.

    Array sizes must be static: ``n_bands`` defaults to ceil(sum_j Z_j / 2) + 1 from the
    unweighted charges (weights at most 1). Failure gives NaN, also in the gradient.
    """
    options = SCFOptions() if options is None else options
    if side not in (-1, 1):
        raise ValueError(f"side must be -1 or +1, got {side}")
    terms = bind_terms(structure, tuple(terms))
    ionic = _check_terms(terms)
    n_bands = options.n_bands or int(math.ceil(sum(ionic.static_ionic_charges(structure)) / 2 - 1e-12)) + 1

    fixed = jax.lax.stop_gradient(structure)
    fixed_charges = ionic.ionic_charges(fixed)
    out, kweights = _run(fixed, basis, terms, options, density0, jnp.sum(fixed_charges), n_bands, fixed_charges)
    ok = out["converged"] & out["eig_ok"] & out["gap_ok"] & out["n_el_ok"]

    n_el = jnp.sum(ionic.ionic_charges(structure))
    n_k = basis.miller.shape[0]
    occ = jnp.broadcast_to(aufbau_occupations(n_el, n_bands, side=side), (n_k, n_bands))
    orbitals = jax.lax.stop_gradient(out["orbitals"])
    e, _ = total_energy(structure, basis, terms, orbitals, occ, kweights)
    return e * jnp.where(ok, 1.0, jnp.nan)
