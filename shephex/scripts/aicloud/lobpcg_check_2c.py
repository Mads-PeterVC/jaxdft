"""The Milestone 2c lobpcg on the problems of the GPU residual diagnosis.

lobpcg_trajectory.py showed that on a V100 LOBPCG drops directions at every
iteration near convergence and that the old penalty on dropped directions
then held the residual at 1e-12 to 6e-12. The theory post (02c, 'On the
clusters') asks for the implemented solver to converge from every start on
the same problems, with late drops present. This job runs jaxdft.lobpcg as
the SCF calls it (diagonal preconditioner, one buffer band, n_converge = the
needed bands, tolerance = the per-k floor) on the GPU and the CPU of the same
node, from three random starts, and once more with an unreachable tolerance
(the stagnation stop). It records iterations, late drops, the stagnation flag
and the true residual of the needed bands against the floor and against the
residual the SCF's eig_ok check allows (2 x floor). From the repository root:

    uv run hex run --dry shephex/scripts/aicloud/lobpcg_check_2c.py
    uv run hex run shephex/scripts/aicloud/lobpcg_check_2c.py
"""

import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import shephex
from shephex.experiment import ExperimentContext

import jaxdft
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
from jaxdft.scf.scf import EIGEN_CHECK_FLOOR_MULTIPLE, EIGEN_FLOOR_FACTOR

DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"
SEEDS = (0, 1, 2)
MODES = ("floor", "unreachable")


def git_commit() -> str:
    source = Path(jaxdft.__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def systems() -> dict:
    """H2 at Gamma (10 Ha) and Si at k-point 1 of the 2^3 grid (12 Ha), at
    densities converged on the CPU, as in lobpcg_trajectory.py."""
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


def run(device, system, seed, mode) -> dict:
    structure, basis, ik, terms, result = system
    n_bands = result.eigenvalues.shape[-1]
    with jax.default_device(device):
        basis_k = jax.device_put(basis[ik], device)
        ham = hamiltonian(
            jax.device_put(structure, device),
            basis_k,
            terms,
            density=jax.device_put(result.density, device),
        )
        dmax = jnp.max(jnp.where(ham.mask, ham.diagonal(), 0.0))
        floor = float(EIGEN_FLOOR_FACTOR * np.finfo(np.float64).eps * dmax)
        tol = floor if mode == "floor" else 0.0

        @jax.jit
        def solve(ham):
            def apply(p):
                return apply_H(ham, p)

            eps, X, info = lobpcg(
                apply,
                basis_k.n_pad,
                n_bands + 1,
                preconditioner=diagonal_preconditioner(ham),
                tol=tol,
                n_converge=n_bands,
                seed=seed,
                return_info=True,
            )
            X = jnp.where(ham.mask[:, None], X, 0)
            r = jnp.linalg.norm(apply(X) - X * eps[None, :], axis=0)
            return jnp.max(r[:n_bands]), info

        residual, info = solve(ham)
        residual = float(residual)
        return {
            "iterations": int(info.iterations),
            "stagnated": bool(info.stagnated),
            "late_drops": int(info.late_drops),
            "residual": residual,
            "floor": floor,
            "within_check": residual < EIGEN_CHECK_FLOOR_MULTIPLE * floor,
        }


@shephex.hexperiment(context=True)
def lobpcg_check_2c(context: ExperimentContext) -> dict:
    devices = {"cpu": jax.devices("cpu")[0]}
    if jax.default_backend() == "gpu":
        devices["gpu"] = jax.devices("gpu")[0]
    context.add("commit", git_commit())
    context.add("devices", {k: d.device_kind for k, d in devices.items()})
    report = {}
    for name, system in systems().items():
        for dev_name, dev in devices.items():
            for mode in MODES:
                for seed in SEEDS:
                    label = f"{name}/{dev_name}/{mode}/{seed}"
                    context.update_progress(label)
                    report[label] = run(dev, system, seed, mode)
    context.add("runs", report)
    context.add("stage", "done")
    return report


if __name__ == "__main__":
    os.chdir(DATA)
    experiment = lobpcg_check_2c(directory="lobpcg-check-2c")
    shephex.slurm_execute(experiment, profile="aicloud-jaxdft-v100", time="00:30:00")
