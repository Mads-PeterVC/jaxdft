"""Which dense step of LOBPCG sets the residual floor on a V100?

gpu_residual_floor.py showed that FFTs and apply_H agree between the V100 and
the CPU to rounding, while LOBPCG on the GPU stalls near 1e-12 (H2) and 4e-12
(Si) instead of reaching 5e-14. This job runs a diagnostic copy of
jaxdft.solvers.lobpcg on the GPU with one change at a time:

- ``rr``: the Rayleigh-Ritz eigh on the CPU (host callback);
- ``gram``: the Gram-matrix eigh of the orthonormalisation on the CPU;
- ``qr``: the QR and inverse of the re-orthonormalisation on the CPU;
- ``all``: all three on the CPU;
- ``penalty``: dropped directions penalised by 2 max|T| instead of 1e3 max|T|.

The unmodified copy (``none``) must reproduce jaxdft's lobpcg exactly, on CPU
and GPU; the job records both. The copy lives here only for this diagnosis.
From the repository root:

    uv run hex run --dry shephex/scripts/aicloud/gpu_lobpcg_dense_steps.py
    uv run hex run shephex/scripts/aicloud/gpu_lobpcg_dense_steps.py
"""

import os
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
    lobpcg,
    monkhorst_pack,
    scf,
)
from jaxdft.hamiltonian.preconditioners import diagonal_preconditioner
from jaxdft.pseudo.gth import hgh_lda
from jaxdft.scf.scf import EIGEN_FLOOR, EIGEN_TOL_FRACTION

DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"
EIG_TOL = max(EIGEN_TOL_FRACTION * 1e-12, EIGEN_FLOOR)
MAX_ITER = 300
DROP = 1e-12  # as jaxdft.solvers.lobpcg._DROP
VARIANTS = {
    "none": {},
    "rr": {"rr": True},
    "gram": {"gram": True},
    "qr": {"qr": True},
    "all": {"rr": True, "gram": True, "qr": True},
    "penalty": {"penalty": 2.0},
}


def host_eigh(a):
    shapes = (
        jax.ShapeDtypeStruct(a.shape[:1], jnp.float64),
        jax.ShapeDtypeStruct(a.shape, a.dtype),
    )
    return jax.pure_callback(lambda m: np.linalg.eigh(np.asarray(m)), shapes, a)


def host_qr_inv(a):
    k = a.shape[1]
    shapes = (
        jax.ShapeDtypeStruct(a.shape, a.dtype),
        jax.ShapeDtypeStruct((k, k), a.dtype),
    )

    def f(m):
        q, r = np.linalg.qr(np.asarray(m))
        return q, np.linalg.inv(r)

    return jax.pure_callback(f, shapes, a)


def lobpcg_variant(
    apply, dim, k, preconditioner, rr=False, gram=False, qr=False, penalty=1e3
):
    """Copy of jaxdft.solvers.lobpcg (default x0, fixed seed) with switches."""
    eigh_rr = host_eigh if rr else jnp.linalg.eigh
    eigh_gram = host_eigh if gram else jnp.linalg.eigh

    def gram_orthonormalise(V):
        norms = jnp.linalg.norm(V, axis=0)
        V = V / jnp.where(norms > 0, norms, 1.0)
        G = V.conj().T @ V
        lam, U = eigh_gram(G)
        keep = lam > DROP * jnp.maximum(jnp.max(lam), 1e-300)
        scale = jnp.where(keep, 1.0 / jnp.sqrt(jnp.where(keep, lam, 1.0)), 0.0)
        return V @ (U * scale), keep

    def orth_against(X, V):
        for _ in range(2):
            V = V - X @ (X.conj().T @ V)
        return V

    def qr_inv(A):
        if qr:
            return host_qr_inv(A)
        q, r = jnp.linalg.qr(A)
        return q, jnp.linalg.inv(r)

    key = jax.random.PRNGKey(0)
    kr, ki = jax.random.split(key)
    x0 = jax.random.normal(kr, (dim, k)) + 1j * jax.random.normal(ki, (dim, k))
    X, _ = jnp.linalg.qr(jnp.asarray(x0, dtype=jnp.complex128))
    AX = apply(X)
    T = X.conj().T @ AX
    eps, C = eigh_rr(0.5 * (T + T.conj().T))
    X, AX = X @ C, AX @ C
    P = jnp.zeros_like(X)
    rnorm = jnp.linalg.norm(AX - X * eps[None, :], axis=0)

    def cond(state):
        return (jnp.max(state[4]) >= EIG_TOL) & (state[5] < MAX_ITER)

    def body(state):
        X, AX, P, eps, _, it = state
        R = AX - X * eps[None, :]
        W = preconditioner(R, eps)
        Z = orth_against(X, jnp.concatenate([W, P], axis=1))
        Q, keep = gram_orthonormalise(Z)
        Q = orth_against(X, Q)
        Q, keep2 = gram_orthonormalise(Q)
        keep = keep & keep2
        Q = jnp.where(keep[None, :], Q, 0)
        AQ = apply(Q)
        S = jnp.concatenate([X, Q], axis=1)
        AS = jnp.concatenate([AX, AQ], axis=1)
        T = S.conj().T @ AS
        T = 0.5 * (T + T.conj().T)
        scale = jnp.max(jnp.abs(T)) + 1.0
        dropped = jnp.concatenate([jnp.zeros(k, dtype=bool), ~keep])
        T = T + jnp.diag(jnp.where(dropped, penalty * scale, 0.0))
        _, C = eigh_rr(T)
        C = C[:, :k]
        X_new, AX_new, P_new = S @ C, AS @ C, Q @ C[k:]
        X_new, Rinv = qr_inv(X_new)
        AX_new = AX_new @ Rinv
        T2 = X_new.conj().T @ AX_new
        eps_new, C2 = eigh_rr(0.5 * (T2 + T2.conj().T))
        X_new, AX_new = X_new @ C2, AX_new @ C2
        rnorm = jnp.linalg.norm(AX_new - X_new * eps_new[None, :], axis=0)
        return X_new, AX_new, P_new, eps_new, rnorm, it + 1

    state = (X, AX, P, eps, rnorm, jnp.asarray(0))
    X, AX, P, eps, rnorm, it = jax.lax.while_loop(cond, body, state)
    return eps, X, it, jnp.max(rnorm)


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


def run(device, system, variant) -> dict:
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
        n_pad = basis_k.n_pad

        @jax.jit
        def solve(ham):
            def apply(p):
                return apply_H(ham, p)

            precondition = diagonal_preconditioner(ham)
            if variant is None:
                eps, X, it = lobpcg(
                    apply,
                    n_pad,
                    k,
                    preconditioner=precondition,
                    tol=EIG_TOL,
                    max_iter=MAX_ITER,
                    return_iterations=True,
                )
                tracked = jnp.nan
            else:
                eps, X, it, tracked = lobpcg_variant(
                    apply, n_pad, k, precondition, **VARIANTS[variant]
                )
            Xm = jnp.where(ham.mask[:, None], X, 0)
            true = jnp.linalg.norm(apply_H(ham, Xm) - Xm * eps[None, :], axis=0)
            return eps, it, tracked, jnp.max(true)

        eps, it, tracked, true = solve(ham)
        return {
            "iterations": int(it),
            "tracked_residual_max": float(tracked),
            "true_residual_max": float(true),
            "eigenvalues": [float(e) for e in eps],
            "sentinel": float(ham.sentinel),
        }


@shephex.hexperiment(context=True)
def gpu_lobpcg_dense_steps(context: ExperimentContext) -> dict:
    devices = {"cpu": jax.devices("cpu")[0]}
    if jax.default_backend() == "gpu":
        devices["gpu"] = jax.devices("gpu")[0]
    context.add("devices", {k: d.device_kind for k, d in devices.items()})
    context.add("eig_tol", EIG_TOL)
    report = {}
    for name, system in systems().items():
        entry = {}
        for dev_name, dev in devices.items():
            context.add("stage", f"{name} {dev_name}")
            entry[f"{dev_name}/jaxdft"] = run(dev, system, None)
            names = VARIANTS if dev_name == "gpu" else ["none"]
            for v in names:
                entry[f"{dev_name}/{v}"] = run(dev, system, v)
        report[name] = entry
        context.add(name, entry)
    context.add("stage", "done")
    return report


if __name__ == "__main__":
    os.chdir(DATA)
    experiment = gpu_lobpcg_dense_steps(directory="gpu-lobpcg-dense-steps")
    shephex.slurm_execute(experiment, profile="aicloud-jaxdft-v100", time="00:30:00")
