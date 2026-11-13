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

Observability (engineering log, 'SCF observability'): ``scf`` records a per-iteration
history in its result (and in ``SCFError``), and when the ``jaxdft`` logger is enabled
at INFO it logs the machine, the setup, one line per iteration as it happens, and the
result. ``energy`` neither records nor logs.
"""

from __future__ import annotations

import logging
import math
import time
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
from jaxdft.output import report
from jaxdft.output.environment import machine_info, peak_memory_bytes
from jaxdft.scf.density import AtomicGaussians, clip_and_renormalise, electron_density
from jaxdft.scf.mixing import Anderson, Mixer
from jaxdft.scf.occupations import frontier_gap, grid_occupations
from jaxdft.scf.solve import fixed_point
from jaxdft.solvers.eigen import EigenInfo
from jaxdft.solvers.lobpcg import lobpcg
from jaxdft.structure.structure import Structure

# At convergence the eigensolver is asked for residuals of EIGEN_TOL_FRACTION * tol and
# must reach EIGEN_CHECK_FRACTION * tol. Both are floored per k-point at what LOBPCG
# can attain, tau_floor(k) = EIGEN_FLOOR_FACTOR eps_mach max diag H_k (about 10 eps ||H||
# attained, with margin), and the check allows EIGEN_CHECK_FLOOR_MULTIPLE times that
# (theory post 2c, 'The floor scales with the operator'; engineering log, 2c API).
EIGEN_TOL_FRACTION = 0.05
EIGEN_CHECK_FRACTION = 0.1
EIGEN_FLOOR_FACTOR = 50
EIGEN_CHECK_FLOOR_MULTIPLE = 2
_EPS = float(jnp.finfo(jnp.float64).eps)

logger = logging.getLogger(__name__)


class SCFError(RuntimeError):
    """The SCF did not produce a valid ground state (not converged, eigensolver not
    converged, or the gap over all k-points below ``min_gap``). ``history`` and ``n_iter`` are
    those of the failed run (see ``SCFResult``), for diagnosis."""

    def __init__(self, message: str, history: dict | None = None, n_iter: int | None = None):
        super().__init__(message)
        self.history = history
        self.n_iter = n_iter


@dataclass(frozen=True)
class SCFOptions:
    """SCF settings. ``tol`` is on ||F(rho) - rho|| = (int (F(rho) - rho)^2)^(1/2).
    ``n_bands`` (the bands that must converge, and are returned) defaults to occupied
    + 1; the eigensolver carries ``buffer_bands`` more, which are neither checked nor
    returned (theory post 2b, decision 2). ``min_gap`` (Ha) guards the band filling:
    the gap over all k-points, CBM - VBM, must exceed it at convergence; it is a
    numerical criterion, not a physical gap. ``degeneracy_tol`` (Ha) groups states
    into levels for fractional occupations (``grid_occupations``).
    ``mixer`` is any ``Mixer`` (``jaxdft.scf.mixing``). ``preconditioner`` is a
    factory ``(Hamiltonian) -> precondition(r, eps)`` (``jaxdft.hamiltonian
    .preconditioners``), or None for none. ``initial_density(structure, basis,
    charges) -> rho`` gives the starting density when ``density0`` is not passed;
    the default is atomic Gaussians of width 0.8 Bohr.

    The eigensolver tolerance adapts to the SCF residual (theory post 2c): at step i
    it is min(tau_{i-1}, ``eigensolver_tol_ratio`` r_{i-1} / N_el), clipped to
    [EIGEN_TOL_FRACTION tol, ``eigensolver_tol_max``] (Ha), starting at the maximum; a
    loose step whose density residual is below ``tol`` is re-diagonalised tightly,
    and the tolerance stays tight from then on. ``eigensolver_tol_ratio`` = 0 gives
    a fixed tolerance. ``k_batch_size`` diagonalises the k-points in chunks of that
    size (``lax.map``), bounding memory; None is one batch. The defaults of
    ``buffer_bands`` (3) and ``k_batch_size`` (64) were measured on a V100 and on
    CPUs (engineering log, 2c cluster runs)."""

    tol: float = 1e-9
    max_iter: int = 100
    n_bands: int | None = None
    mixer: Mixer = field(default_factory=Anderson)
    eigensolver: Callable = lobpcg
    preconditioner: Callable | None = diagonal_preconditioner
    min_gap: float = 1e-3
    initial_density: Callable = field(default_factory=AtomicGaussians)
    buffer_bands: int = 3
    degeneracy_tol: float = 1e-6
    eigensolver_tol_ratio: float = 0.4
    eigensolver_tol_max: float = 5e-3
    k_batch_size: int | None = 64


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SCFResult:
    """Converged SCF. ``density`` is built from ``orbitals`` and ``occupations``; the
    orbitals are eigenvectors of H at the last input density. ``energies`` maps term
    names to energies (Ha); ``total_energy`` is their sum. ``eigen_residuals`` are
    ||H psi - eps psi|| per band.

    ``history`` maps names to arrays of length ``n_iter + 1``, one entry per evaluated
    input density rho_i (i = 0 is the initial guess): ``energy`` (Kohn-Sham energy of
    the orbitals of H[rho_i], Ha), ``delta_energy`` (change from i - 1; NaN at 0),
    ``residual`` (||F(rho_i) - rho_i||, the converged quantity), ``projection`` (norm
    of the change clipping and renormalising made to the mixed density), ``homo``,
    ``lumo``, ``gap`` (Ha, over all k), ``eig_residual_max`` and ``eig_iters`` (worst
    eigensolver residual and iteration count over k), ``eig_iters_mean`` (mean count
    over k), ``eig_tol`` (the step's adaptive tolerance before the per-k floor),
    ``refined`` (1 if the step was re-diagonalised tightly; the counts include both
    diagonalisations), ``eig_drops`` (largest late-drop count over k) and
    ``eig_stagnated`` (1 if any k-point ended by the stagnation stop)."""

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
    history: dict


# --- the direct energy functional ------------------------------------------------------


def total_energy(structure: Structure, basis: PlaneWaveBasis, terms, orbitals, occupations):
    """Direct Kohn-Sham energy of ``orbitals`` (n_k, n_pad, n_bands) with
    ``occupations`` (n_k, n_bands): (E, {term name: energy}). Operator terms contribute
    sum_k w_k sum_n f_nk <psi_nk|T|psi_nk> with the weights of the basis; energy terms
    are evaluated at the density of the orbitals."""
    w = basis.kweights
    density = electron_density(structure.lattice, basis, orbitals, occupations)
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


def _log_iteration(it, values, t0):
    """Host callback (``jax.debug.callback``): one live iteration line."""
    logger.info(report.iteration_line(int(it), values, elapsed=time.perf_counter() - float(t0)))


_STATIC = (
    "n_bands",
    "n_solve",
    "degeneracy_tol",
    "max_iter",
    "mixer",
    "eigensolver",
    "preconditioner",
    "history",
    "live",
    "k_batch_size",
)


def _map_k(f, xs, k_batch_size):
    """f mapped over the leading (k-point) axis of the pytree ``xs``: one vmap, or
    sequential chunks of ``k_batch_size`` vmapped k-points (lax.map)."""
    if k_batch_size is None:
        return jax.vmap(f)(xs)
    return jax.lax.map(f, xs, batch_size=int(k_batch_size))


@partial(jax.jit, static_argnames=_STATIC)
def _scf_core(
    structure,
    basis,
    terms,
    density0,
    n_el,
    tol,
    min_gap,
    tol_ratio,
    tol_max,
    t0,
    *,
    n_bands,
    n_solve,
    degeneracy_tol,
    max_iter,
    mixer,
    eigensolver,
    preconditioner,
    history,
    live,
    k_batch_size,
):
    """The traced SCF. The eigensolver iterates ``n_solve`` = ``n_bands`` + buffer
    bands and converges the first ``n_bands``; occupations, residual checks and
    results use those."""
    volume = structure.lattice.volume
    n_tot = math.prod(basis.fft_shape)
    n_k = basis.miller.shape[0]
    tight = EIGEN_TOL_FRACTION * tol
    loosest = jnp.maximum(tol_max, tight)
    # The adaptive tolerance needs occupations that eigenvalue errors cannot change:
    # complete levels at the frontier, N_el / 2 an integer. With a fractional count the
    # remainder goes level by level (grid_occupations, grouped within degeneracy_tol),
    # and a loose solve that splits a degenerate frontier level puts it into one state,
    # breaking the symmetry of the density for good (engineering log, 2c
    # implementation). Such runs use the fixed tolerance.
    half = n_el / 2
    ratio = jnp.where(jnp.abs(half - jnp.round(half)) > 1e-9, 0.0, tol_ratio)

    def norm(r):
        return jnp.sqrt(volume / n_tot * jnp.sum(r**2))

    def occupations(eps):
        return grid_occupations(eps[:, :n_bands], basis.kweights, n_el, tol=degeneracy_tol)

    def diagonalise(potential, X_prev, tau):
        """Per k-point: vectors, eigenvalues, residuals, EigenInfo and the residual
        the eig_ok check allows, at tolerance max(tau, floor of k)."""

        def per_k(args):
            basis_k, x0 = args
            ham = hamiltonian(structure, basis_k, terms, potential=potential)
            precondition = None if preconditioner is None else preconditioner(ham)
            floor = EIGEN_FLOOR_FACTOR * _EPS * jnp.max(jnp.where(basis_k.mask, ham.diagonal(), 0.0))

            def apply(psi):
                return apply_H(ham, psi)

            eps, vecs, info = eigensolver(
                apply,
                basis_k.n_pad,
                n_solve,
                x0=x0,
                preconditioner=precondition,
                tol=jnp.maximum(tau, floor),
                n_converge=n_bands,
                return_info=True,
            )
            vecs = jnp.where(basis_k.mask[:, None], vecs, 0)
            res = jnp.linalg.norm(apply(vecs) - vecs * eps[None, :], axis=0)
            check = jnp.maximum(EIGEN_CHECK_FRACTION * tol, EIGEN_CHECK_FLOOR_MULTIPLE * floor)
            return vecs, eps, res, info, check

        return _map_k(per_k, (basis, X_prev), k_batch_size)

    def density_of(X, eps):
        occ = occupations(eps)
        return electron_density(structure.lattice, basis, X[:, :, :n_bands], occ)

    def F(rho, aux):
        X_prev, e_prev, tau_prev, r_prev, latched = aux[0], aux[4], aux[6], aux[7], aux[8]
        potential = total_potential(terms, structure, basis, rho)
        proposed = jnp.where(ratio > 0, ratio * r_prev / n_el, 0.0)
        tau = jnp.where(latched, tight, jnp.clip(jnp.minimum(tau_prev, proposed), tight, loosest))
        X, eps, res, info, check = diagonalise(potential, X_prev, tau)
        rho_out = density_of(X, eps)
        r = norm(rho_out - rho)
        refine = (r < tol) & (tau > tight)

        def tighten(_):
            X2, eps2, res2, info2, check2 = diagonalise(potential, X, tight)
            info2 = EigenInfo(
                info.iterations + info2.iterations, info.stagnated | info2.stagnated, info.late_drops + info2.late_drops
            )
            rho2 = density_of(X2, eps2)
            return X2, eps2, res2, info2, check2, rho2, norm(rho2 - rho)

        X, eps, res, info, check, rho_out, r = jax.lax.cond(
            refine, tighten, lambda _: (X, eps, res, info, check, rho_out, r), None
        )
        Xn = X[:, :, :n_bands]
        e = total_energy(structure, basis, terms, Xn, occupations(eps))[0] if history else e_prev
        return rho_out, (X, eps, res, info, e, e - e_prev, tau, r, latched | refine, refine, check)

    def record(rho, rho_out, aux):
        _, eps, res, info, e, de, tau, _, _, refined, _ = aux
        vbm, cbm = frontier_gap(eps[:, :n_bands], n_el)
        return {
            "energy": e,
            "delta_energy": de,
            "homo": vbm,
            "lumo": cbm,
            "gap": cbm - vbm,
            "eig_residual_max": jnp.max(res[:, :n_bands]),
            "eig_iters": jnp.max(info.iterations),
            "eig_iters_mean": jnp.mean(info.iterations),
            "eig_tol": tau,
            "refined": refined,
            "eig_drops": jnp.max(info.late_drops),
            "eig_stagnated": jnp.any(info.stagnated),
        }

    key = jax.random.PRNGKey(0)
    kr, ki = jax.random.split(key)
    shape = (n_k, basis.n_pad, n_solve)
    X0 = jax.random.normal(kr, shape) + 1j * jax.random.normal(ki, shape)
    X0 = jnp.where(basis.mask[..., None], X0, 0)
    nan = jnp.asarray(jnp.nan)
    info0 = EigenInfo(jnp.zeros(n_k, dtype=int), jnp.zeros(n_k, dtype=bool), jnp.zeros(n_k, dtype=int))
    aux0 = (
        X0,
        jnp.zeros((n_k, n_solve)),
        jnp.zeros((n_k, n_solve)),
        info0,
        nan,
        nan,
        loosest,  # tau_{-1}: the first step uses the loosest tolerance
        jnp.asarray(jnp.inf),  # r_{-1}
        jnp.asarray(False),  # latched
        jnp.asarray(False),  # refined
        jnp.zeros(n_k),  # residual allowed by the eig_ok check, per k
    )

    def project(rho):
        return clip_and_renormalise(rho, n_el, volume)

    res = fixed_point(
        F,
        density0,
        mixer,
        tol,
        max_iter,
        project=project,
        norm=norm,
        aux0=aux0,
        record=record if history else None,
        on_step=(lambda it, values: jax.debug.callback(_log_iteration, it, values, t0, ordered=True)) if live else None,
    )
    X, eps, eig_res = res.aux[:3]
    check = res.aux[10]
    X, eps, eig_res = X[:, :, :n_bands], eps[:, :n_bands], eig_res[:, :n_bands]

    vbm, cbm = frontier_gap(eps, n_el)
    gap_ok = jnp.isfinite(cbm) & (cbm - vbm > min_gap)
    eig_ok = jnp.all(eig_res < check[:, None])
    return {
        "orbitals": X,
        "eigenvalues": eps,
        "occupations": occupations(eps),
        "density": res.fx,
        "residual": res.residual,
        "n_iter": res.n_iter,
        "converged": res.converged,
        "eigen_residuals": eig_res,
        "gap_ok": gap_ok,
        "eig_ok": eig_ok,
        "n_el_ok": n_el <= 2.0 * n_bands,
        "history": res.history,
    }


_total_energy_jit = jax.jit(total_energy)


def _n_solve(basis, n_bands, options) -> int:
    """Bands the eigensolver iterates: the needed ones plus buffer bands, capped by
    the smallest basis."""
    if options.buffer_bands < 0:
        raise ValueError(f"buffer_bands must be >= 0, got {options.buffer_bands}")
    return max(n_bands, min(n_bands + int(options.buffer_bands), int(jnp.min(basis.n_pw))))


def _core_arguments(structure, basis, terms, options, density0, n_el, n_bands, *, history, live):
    args = (
        structure,
        basis,
        terms,
        jnp.asarray(density0, dtype=jnp.float64),
        n_el,
        jnp.asarray(options.tol, dtype=jnp.float64),
        jnp.asarray(options.min_gap, dtype=jnp.float64),
        jnp.asarray(options.eigensolver_tol_ratio, dtype=jnp.float64),
        jnp.asarray(options.eigensolver_tol_max, dtype=jnp.float64),
        jnp.asarray(0.0),  # t0, replaced at call time when logging live
    )
    static = dict(
        n_bands=n_bands,
        n_solve=_n_solve(basis, n_bands, options),
        degeneracy_tol=float(options.degeneracy_tol),
        max_iter=int(options.max_iter),
        mixer=options.mixer,
        eigensolver=options.eigensolver,
        preconditioner=options.preconditioner,
        history=history,
        live=live,
        k_batch_size=None if options.k_batch_size is None else int(options.k_batch_size),
    )
    return args, static


def _run(structure, basis, terms, options, density0, n_el, n_bands, charges):
    """The traced path used by ``energy``: no history, no logging."""
    if density0 is None:
        density0 = options.initial_density(structure, basis, charges)
    args, static = _core_arguments(structure, basis, terms, options, density0, n_el, n_bands, history=False, live=False)
    return _scf_core(*args, **static)


def _integral(rho, structure):
    return float(structure.lattice.volume) / rho.size * float(jnp.sum(rho))


# --- public entry points -------------------------------------------------------------------------


def scf(
    structure: Structure, basis: PlaneWaveBasis, terms, options: SCFOptions | None = None, density0=None
) -> SCFResult:
    """Run the SCF on the host and return an ``SCFResult``; raise ``SCFError`` with a
    description if it fails.

    If the ``jaxdft`` logger is enabled at INFO (e.g. ``jaxdft.log_to_file``), the run
    is logged: machine, structure, terms, basis and settings, then one line per SCF
    iteration as it completes, then the result and timings."""
    t_start = time.perf_counter()
    options = SCFOptions() if options is None else options
    live = logger.isEnabledFor(logging.INFO)
    terms = bind_terms(structure, tuple(terms))
    ionic = _check_terms(terms)
    charges = ionic.ionic_charges(structure)
    n_el = float(jnp.sum(charges))
    n_bands = options.n_bands or int(math.ceil(n_el / 2 - 1e-12)) + 1
    if n_el > 2 * n_bands:
        raise ValueError(f"{n_el} electrons do not fit in n_bands={n_bands}")
    given = density0 is not None
    if density0 is None:
        density0 = options.initial_density(structure, basis, charges)
    args, static = _core_arguments(
        structure, basis, terms, options, density0, jnp.asarray(n_el), n_bands, history=True, live=live
    )

    if live:
        eig_tol = EIGEN_TOL_FRACTION * options.tol
        initial = "density0 argument" if given else report.describe(options.initial_density)
        lines = report.machine_lines(machine_info())
        lines += report.structure_lines(structure)
        lines += report.model_lines(terms)
        lines += report.basis_lines(structure, basis)
        lines += report.settings_lines(options, n_el, n_bands, eig_tol, initial, _integral(args[3], structure))
        _log_lines(lines)

    # Compile ahead of time, so compilation and the SCF itself are timed separately.
    t_setup = time.perf_counter()
    compiled = _scf_core.lower(*args, **static).compile()
    t_compiled = time.perf_counter()
    if live:
        _log_lines(report.iteration_lines_header(options.tol))
    out = compiled(*args[:-1], jnp.asarray(time.perf_counter()))
    out = jax.block_until_ready(out)
    t_solved = time.perf_counter()

    n_iter = int(out["n_iter"])
    history = {k: v[: n_iter + 1] for k, v in out["history"].items()}
    timings = {"setup": t_setup - t_start, "compile": t_compiled - t_setup, "SCF": t_solved - t_compiled}

    problems = []
    if not bool(out["converged"]):
        problems.append(
            f"SCF not converged after {n_iter} iterations "
            f"(residual {float(out['residual']):.3e}, tol {options.tol:.1e})"
        )
    if not bool(out["eig_ok"]):
        problems.append(f"eigensolver not converged (max residual {float(jnp.max(out['eigen_residuals'])):.3e})")
    if not bool(out["gap_ok"]):
        gap = float(history["gap"][-1]) if "gap" in history else float("nan")
        problems.append(
            f"gap over all k-points {gap:.3e} Ha below min_gap={options.min_gap} Ha "
            "(or no band above the occupied ones): the band filling is not the ground state"
        )
    if problems:
        message = "; ".join(problems)
        if live:
            timings["total"] = time.perf_counter() - t_start
            status = f"FAILED: {message}"
            _log_lines(
                report.result_lines(
                    status, None, None, out["eigenvalues"], out["occupations"], None, timings, peak_memory_bytes()
                )
            )
        raise SCFError(message, history=history, n_iter=n_iter)

    e, parts = _total_energy_jit(structure, basis, terms, out["orbitals"], out["occupations"])
    e = jax.block_until_ready(e)
    if live:
        timings["final energy (incl. compile)"] = time.perf_counter() - t_solved
        timings["total"] = time.perf_counter() - t_start
        status = f"converged in {n_iter} iterations: |F(rho)-rho| = {float(out['residual']):.3e} < {options.tol:.1e}"
        order = list(dict.fromkeys(type(t).__name__ for t in terms))
        _log_lines(
            report.result_lines(
                status,
                {k: parts[k] for k in order},
                e,
                out["eigenvalues"],
                out["occupations"],
                _integral(out["density"], structure),
                timings,
                peak_memory_bytes(),
            )
        )
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
        history=history,
    )


def _log_lines(lines):
    for line in lines:
        logger.info(line)


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
    one-sided derivative at integer N_el: -1 the VBM (HOMO) side, +1 the CBM (LUMO)
    side, with the frontier levels taken over all k-points (``grid_occupations``).

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
    out = _run(fixed, basis, terms, options, density0, jnp.sum(fixed_charges), n_bands, fixed_charges)
    ok = out["converged"] & out["eig_ok"] & out["gap_ok"] & out["n_el_ok"]

    n_el = jnp.sum(ionic.ionic_charges(structure))
    eps = jax.lax.stop_gradient(out["eigenvalues"])
    occ = grid_occupations(eps, basis.kweights, n_el, side=side, tol=options.degeneracy_tol)
    orbitals = jax.lax.stop_gradient(out["orbitals"])
    e, _ = total_energy(structure, basis, terms, orbitals, occ)
    return e * jnp.where(ok, 1.0, jnp.nan)
