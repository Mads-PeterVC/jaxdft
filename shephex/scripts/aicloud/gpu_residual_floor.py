"""Why LOBPCG residuals stop near 1e-12 on a V100 but reach ~3e-14 on CPU.

The test suite on the V100 fails every test that runs the SCF at tol = 1e-12,
because the eigensolver must reach residuals of 1e-13 and stops near 1.5e-12
(on CPU the same H2 run reaches 2.7e-14). This job compares, on the GPU and
on the CPU of the same node (JAX_PLATFORMS=cuda,cpu gives both):

- the FFT round trip on the basis grid;
- apply_H for the same block on both devices;
- eigh of random Hermitian matrices of Rayleigh-Ritz size;
- LOBPCG at the SCF's tight eigensolver tolerance: the true residual
  ||H x - eps x|| after a growing iteration cap, and the GPU orbitals
  re-checked on the CPU.

for H2 (the failing test size) and silicon at one k-point of the 2³ grid, each
at a density converged on the CPU. From the repository root:

    uv run hex run --dry shephex/scripts/aicloud/gpu_residual_floor.py
    uv run hex run shephex/scripts/aicloud/gpu_residual_floor.py
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

# shephex maps the working directory at submission to the profile's remote
# directory, so submitting from here puts the runs directly under it.
DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"

TIGHT_SCF_TOL = 1e-12  # tests/support/tolerances.py: SCF_TOL_TIGHT
EIG_TOL = max(EIGEN_TOL_FRACTION * TIGHT_SCF_TOL, EIGEN_FLOOR)
ITERATION_CAPS = (20, 40, 80, 160, 300)


def systems() -> dict:
    """Structure, single-k basis, terms and CPU-converged density per system."""
    cpu = jax.devices("cpu")[0]
    out = {}
    with jax.default_device(cpu):
        h2 = dimer_in_box("H", bond_length=1.45, box=10.0)
        b = PlaneWaveBasis.build(h2.lattice, kpoints=np.zeros((1, 3)), ecut=10.0)
        t = lda_terms({"H": hgh_lda("H")})
        r = scf(h2, b, t, SCFOptions())
        out["H2"] = (h2, b, 0, t, r)

        si = diamond(10.263, "Si", unit="Bohr")
        k, w = monkhorst_pack((2, 2, 2))
        b = PlaneWaveBasis.build(si.lattice, k, 12.0, kweights=w)
        t = lda_terms({"Si": hgh_lda("Si")})
        r = scf(si, b, t, SCFOptions())
        out["Si"] = (si, b, 1, t, r)  # k-point 1 is not Gamma
    return out


def true_residual(ham, X, eps):
    mask = ham.mask[:, None]
    X = jnp.where(mask, X, 0)
    return jnp.linalg.norm(apply_H(ham, X) - X * eps[None, :], axis=0)


def on_device(device, system) -> dict:
    structure, basis, ik, terms, result = system
    n_solve = result.eigenvalues.shape[-1] + 1  # one buffer band, as the SCF
    rng = np.random.default_rng(0)
    out = {}
    with jax.default_device(device):
        basis_k = jax.device_put(basis[ik], device)
        density = jax.device_put(result.density, device)
        structure = jax.device_put(structure, device)
        ham = hamiltonian(structure, basis_k, terms, density=density)
        n_pad = basis_k.n_pad

        grid = rng.normal(size=basis.fft_shape) + 1j * rng.normal(size=basis.fft_shape)
        x = jnp.asarray(grid)
        back = jnp.fft.ifftn(jnp.fft.fftn(x))
        out["fft_roundtrip_max_rel"] = float(
            jnp.max(jnp.abs(back - x)) / jnp.max(jnp.abs(x))
        )

        psi = rng.normal(size=(n_pad, n_solve)) + 1j * rng.normal(size=(n_pad, n_solve))
        out["Hpsi"] = np.asarray(jax.jit(apply_H)(ham, jnp.asarray(psi)))

        eigh = {}
        for n in (n_solve, 2 * n_solve, 3 * n_solve):
            a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
            a = jnp.asarray(0.5 * (a + a.conj().T))
            lam, v = jnp.linalg.eigh(a)
            eigh[n] = {
                "residual_rel": float(
                    jnp.linalg.norm(a @ v - v * lam) / jnp.linalg.norm(a)
                ),
                "orthogonality": float(jnp.linalg.norm(v.conj().T @ v - jnp.eye(n))),
            }
        out["eigh"] = eigh

        @jax.jit
        def solve(ham, cap):
            def apply(p):
                return apply_H(ham, p)

            return lobpcg(
                apply,
                n_pad,
                n_solve,
                preconditioner=diagonal_preconditioner(ham),
                tol=EIG_TOL,
                max_iter=cap,
                return_iterations=True,
            )

        caps = {}
        for cap in ITERATION_CAPS:
            eps, X, its = solve(ham, jnp.asarray(cap))
            res = true_residual(ham, X, eps)
            caps[cap] = {
                "iterations": int(its),
                "true_residual_max": float(jnp.max(res)),
            }
        out["lobpcg"] = caps
        out["final"] = (np.asarray(eps), np.asarray(X))
    out["n_pad"], out["n_solve"] = int(n_pad), int(n_solve)
    return out


def recheck_on_cpu(system, eps, X) -> float:
    """True residual of orbitals from any device, evaluated on the CPU."""
    structure, basis, ik, terms, result = system
    with jax.default_device(jax.devices("cpu")[0]):
        ham = hamiltonian(structure, basis[ik], terms, density=result.density)
        return float(jnp.max(true_residual(ham, jnp.asarray(X), jnp.asarray(eps))))


@shephex.hexperiment(context=True)
def gpu_residual_floor(context: ExperimentContext) -> dict:
    context.add("stage", "setup")
    devices = {"cpu": jax.devices("cpu")[0]}
    if jax.default_backend() == "gpu":
        devices["gpu"] = jax.devices("gpu")[0]
    context.add("devices", {k: d.device_kind for k, d in devices.items()})
    context.add("eig_tol", EIG_TOL)
    report = {}
    for name, system in systems().items():
        context.add("stage", name)
        per = {k: on_device(d, system) for k, d in devices.items()}
        ref = per["cpu"]["Hpsi"]
        entry = {}
        for k, r in per.items():
            entry[k] = {
                "fft_roundtrip_max_rel": r["fft_roundtrip_max_rel"],
                "Hpsi_rel_diff_vs_cpu": float(
                    np.linalg.norm(r["Hpsi"] - ref) / np.linalg.norm(ref)
                ),
                "eigh": {str(n): v for n, v in r["eigh"].items()},
                "lobpcg": {str(c): v for c, v in r["lobpcg"].items()},
                "final_residual_rechecked_on_cpu": recheck_on_cpu(system, *r["final"]),
                "n_pad": r["n_pad"],
                "n_solve": r["n_solve"],
            }
        report[name] = entry
        context.add(name, entry)
    context.add("stage", "done")
    return report


if __name__ == "__main__":
    os.chdir(DATA)
    experiment = gpu_residual_floor(directory="gpu-residual-floor")
    shephex.slurm_execute(experiment, profile="aicloud-jaxdft-v100", time="00:30:00")
