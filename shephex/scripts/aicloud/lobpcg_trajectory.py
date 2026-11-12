"""Per-iteration trajectory of LOBPCG on CPU and V100: what limits the residual?

Three explanations of the V100 failures at tight tolerance are on the table
(theory post 2c, 'Robust stopping' and 'The Rayleigh-Ritz penalty'):

1. the penalty on dropped directions, 1e3 max|T|, inflates ||T|| and with it
   the error of the Rayleigh-Ritz eigh, which sets a floor on the GPU;
2. the same, also on the CPU at tolerances near 1e-13;
3. directions are dropped only at the first iteration, so the penalty is
   inactive afterwards; the solver drifts once past its attainable residual
   and returns its last iterate.

This job runs an instrumented copy of jaxdft.solvers.lobpcg and records per
iteration: the largest residual, the directions dropped in each of the two
orthonormalisation passes, the smallest Gram eigenvalue relative to the
largest (the drop threshold is 1e-12), max|T| including the penalty, and the
backward error of the Rayleigh-Ritz eigh, max_j ||T c_j - theta_j c_j||, for
the selected pairs. (``gram_min_rel`` pairs Gram eigenvalues with column
norms by index, which does not identify the zero columns; the column is
meaningless and reference/gpu_lobpcg_floor/extract.py drops it.) Variants: today's penalty (``jaxdft``), the interlacing
bound of the 2c post (``bound``) and today's penalty with the Rayleigh-Ritz
eigh on the host (``rr_host``). Each from three random starts, stopping at the
SCF's tight eigensolver tolerance (``stop``) and forced to MAX_ITER
iterations (``forced``). Problems as in gpu_lobpcg_dense_steps.py: H2 at
Gamma (10 Ha) and Si at k-point 1 of the 2^3 grid (12 Ha), at densities
converged on the CPU.

Locally (CPU only, writes JSON to the given path):

    uv run python shephex/scripts/aicloud/lobpcg_trajectory.py --local out.json

On aicloud, from the repository root:

    uv run hex run --dry shephex/scripts/aicloud/lobpcg_trajectory.py
    uv run hex run shephex/scripts/aicloud/lobpcg_trajectory.py
"""

import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import shephex
from shephex.experiment import ExperimentContext

from jaxdft import (
    PlaneWaveBasis,
    SCFOptions,
    apply_H,
    diamond,
    dimer_in_box,
    hamiltonian,
    lda_terms,
    monkhorst_pack,
    scf,
)
from jaxdft.hamiltonian.preconditioners import diagonal_preconditioner
from jaxdft.pseudo.gth import hgh_lda

DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"
EIG_TOL = 5e-14  # what the SCF asks for at tol = 1e-12 today
MAX_ITER = 300
DROP = 1e-12  # as jaxdft.solvers.lobpcg._DROP
SEEDS = (0, 1, 2)
MODES = {"stop": EIG_TOL, "forced": 0.0}


def host_eigh(a):
    shapes = (
        jax.ShapeDtypeStruct(a.shape[:1], jnp.float64),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
    )
    return jax.pure_callback(lambda m: np.linalg.eigh(np.asarray(m)), shapes, a)


def lobpcg_traced(apply, dim, k, preconditioner, seed, tol, penalty, rr_host):
    """Copy of jaxdft.solvers.lobpcg (3383ca3) with a per-iteration record."""
    eigh_rr = host_eigh if rr_host else jnp.linalg.eigh

    def gram_orthonormalise(V):
        norms = jnp.linalg.norm(V, axis=0)
        V = V / jnp.where(norms > 0, norms, 1.0)
        G = V.conj().T @ V
        lam, U = jnp.linalg.eigh(G)
        top = jnp.maximum(jnp.max(lam), 1e-300)
        keep = lam > DROP * top
        scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)
        # smallest relative Gram eigenvalue among non-zero columns
        rel = jnp.where(norms > 0, lam / top, jnp.inf)
        return V @ (U * scale), keep, jnp.min(jnp.abs(rel))

    def orth_against(X, V):
        for _ in range(2):
            V = V - X @ (X.conj().T @ V)
        return V

    key = jax.random.PRNGKey(seed)
    kr, ki = jax.random.split(key)
    x0 = jax.random.normal(kr, (dim, k)) + 1j * jax.random.normal(ki, (dim, k))
    X, _ = jnp.linalg.qr(jnp.asarray(x0, dtype=jnp.complex128))
    AX = apply(X)
    T = X.conj().T @ AX
    eps, C = eigh_rr(0.5 * (T + T.conj().T))
    X, AX = X @ C, AX @ C
    P = jnp.zeros_like(X)
    rnorm = jnp.linalg.norm(AX - X * eps[None, :], axis=0)
    nan = jnp.full(MAX_ITER, jnp.nan)
    rec = {
        "residual": nan,
        "drops1": nan,
        "drops2": nan,
        "gram_min_rel": nan,
        "T_max": nan,
        "rr_backward": nan,
        "sigma": nan,
    }

    def cond(state):
        return (jnp.max(state[4]) >= tol) & (state[5] < MAX_ITER)

    def body(state):
        X, AX, P, eps, _, it, rec = state
        R = AX - X * eps[None, :]
        W = preconditioner(R, eps)
        Z = orth_against(X, jnp.concatenate([W, P], axis=1))
        Q, keep1, g1 = gram_orthonormalise(Z)
        Q = orth_against(X, Q)
        Q, keep2, g2 = gram_orthonormalise(Q)
        keep = keep1 & keep2
        Q = jnp.where(keep[None, :], Q, 0)
        AQ = apply(Q)
        S = jnp.concatenate([X, Q], axis=1)
        AS = jnp.concatenate([AX, AQ], axis=1)
        T = S.conj().T @ AS
        T = 0.5 * (T + T.conj().T)
        if penalty == "jaxdft":
            sigma = 1e3 * (jnp.max(jnp.abs(T)) + 1.0)
        else:  # interlacing bound of the 2c post
            sigma = eps[-1] + (eps[-1] - eps[0]) + 1.0
        dropped = jnp.concatenate([jnp.zeros(k, dtype=bool), ~keep])
        T = T + jnp.diag(jnp.where(dropped, sigma, 0.0))
        theta, C = eigh_rr(T)
        bw = jnp.linalg.norm(T @ C[:, :k] - C[:, :k] * theta[None, :k], axis=0)
        C = C[:, :k]
        X_new, AX_new, P_new = S @ C, AS @ C, Q @ C[k:]
        Xq, Rq = jnp.linalg.qr(X_new)
        X_new, AX_new = Xq, AX_new @ jnp.linalg.inv(Rq)
        T2 = X_new.conj().T @ AX_new
        eps_new, C2 = eigh_rr(0.5 * (T2 + T2.conj().T))
        X_new, AX_new = X_new @ C2, AX_new @ C2
        rn = jnp.linalg.norm(AX_new - X_new * eps_new[None, :], axis=0)
        values = {
            "residual": jnp.max(rn),
            "drops1": jnp.sum(~keep1),
            "drops2": jnp.sum(~keep2 & keep1),
            "gram_min_rel": jnp.minimum(g1, g2),
            "T_max": jnp.max(jnp.abs(T)),
            "rr_backward": jnp.max(bw),
            "sigma": jnp.where(jnp.any(dropped), sigma, jnp.nan),
        }
        rec = {n: rec[n].at[it].set(values[n]) for n in rec}
        return X_new, AX_new, P_new, eps_new, rn, it + 1, rec

    state = (X, AX, P, eps, rnorm, jnp.asarray(0), rec)
    X, AX, P, eps, rnorm, it, rec = jax.lax.while_loop(cond, body, state)
    return eps, X, it, rec


def systems() -> dict:
    cpu = jax.devices("cpu")[0]
    out = {}
    with jax.default_device(cpu):
        h2 = dimer_in_box("H", bond_length=1.45, box=10.0)
        b = PlaneWaveBasis.build(h2.lattice, kpoints=np.zeros((1, 3)), ecut=10.0)
        t = lda_terms({"H": hgh_lda("H")})
        out["H2"] = (h2, b, 0, t, scf(h2, b, t, SCFOptions()))
        si = diamond(10.263, "Si", unit="Bohr")
        k, w = monkhorst_pack((2, 2, 2))
        b = PlaneWaveBasis.build(si.lattice, k, 12.0, kweights=w)
        t = lda_terms({"Si": hgh_lda("Si")})
        out["Si"] = (si, b, 1, t, scf(si, b, t, SCFOptions()))
    return out


def run(device, system, penalty, rr_host, seed, tol) -> dict:
    structure, basis, ik, terms, result = system
    k = result.eigenvalues.shape[-1] + 1
    with jax.default_device(device):
        basis_k = jax.device_put(basis[ik], device)
        ham = hamiltonian(
            jax.device_put(structure, device),
            basis_k,
            terms,
            density=jax.device_put(result.density, device),
        )

        @jax.jit
        def solve(ham):
            def apply(p):
                return apply_H(ham, p)

            precondition = diagonal_preconditioner(ham)
            eps, X, it, rec = lobpcg_traced(
                apply, basis_k.n_pad, k, precondition, seed, tol, penalty, rr_host
            )
            Xm = jnp.where(ham.mask[:, None], X, 0)
            true = jnp.linalg.norm(apply_H(ham, Xm) - Xm * eps[None, :], axis=0)
            return it, jnp.max(true), rec

        it, true, rec = solve(ham)
        n = int(it)
        return {
            "iterations": n,
            "true_residual_max": float(true),
            "trace": {
                name: [float(v) for v in np.asarray(arr)[:n]]
                for name, arr in rec.items()
            },
        }


def variants(dev_name):
    out = {"jaxdft": ("jaxdft", False), "bound": ("bound", False)}
    if dev_name == "gpu":
        out["rr_host"] = ("jaxdft", True)
    return out


def diagnose(devices: dict, progress=None) -> dict:
    report = {}
    for name, system in systems().items():
        for dev_name, dev in devices.items():
            for var, (penalty, rr_host) in variants(dev_name).items():
                for mode, tol in MODES.items():
                    for seed in SEEDS:
                        label = f"{name}/{dev_name}/{var}/{mode}/{seed}"
                        if progress is not None:
                            progress(label)
                        report[label] = run(
                            dev, system, penalty, rr_host, seed, tol
                        )
    return report


@shephex.hexperiment(context=True)
def lobpcg_trajectory(context: ExperimentContext) -> dict:
    devices = {"cpu": jax.devices("cpu")[0]}
    if jax.default_backend() == "gpu":
        devices["gpu"] = jax.devices("gpu")[0]
    context.add("devices", {k: d.device_kind for k, d in devices.items()})
    context.add("eig_tol", EIG_TOL)
    report = diagnose(devices, progress=context.update_progress)
    out = Path(context.directory) / "trajectories.json"
    out.write_text(json.dumps(report))
    summary = {
        label: [r["iterations"], r["true_residual_max"]]
        for label, r in report.items()
    }
    context.add("summary", summary)
    context.add("stage", "done")
    return summary


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--local":
        report = diagnose({"cpu": jax.devices("cpu")[0]}, progress=print)
        Path(sys.argv[2]).write_text(json.dumps(report))
    else:
        os.chdir(DATA)
        experiment = lobpcg_trajectory(directory="lobpcg-trajectory")
        shephex.slurm_execute(
            experiment, profile="aicloud-jaxdft-v100", time="00:45:00"
        )
