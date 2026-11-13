"""Buffer bands and k-point chunk size on a V100 (Milestone 2c).

The 2b post left the number of buffer bands to a wall-clock measurement on
the GPU, where a wider block is cheaper per column than on the CPU; the 2c
post adds the k-point chunk size, which bounds memory and limits lockstep to
a chunk, at the cost of parallel width. On the CPU one buffer band was
fastest, and two removed a refinement spike at Gamma (engineering log, 2c
implementation). One experiment per setting: silicon at 12 Ha on the 8^3
grid (512 k-points), default SCFOptions otherwise. Each runs the SCF twice
in one process: first_s includes compilation, steady_s is the repeat. From
the repository root:

    uv run hex run --dry shephex/scripts/aicloud/si_eigensolver_settings_scan.py
    uv run hex run shephex/scripts/aicloud/si_eigensolver_settings_scan.py
    uv run hex report -p shephex/data/aicloud/si-eigensolver-settings
"""

import os
import subprocess
import time
from pathlib import Path

import jax
import numpy as np
import shephex
from shephex.experiment import ExperimentContext

import jaxdft
from jaxdft import PlaneWaveBasis, SCFOptions, diamond, lda_terms, monkhorst_pack, scf
from jaxdft.output.environment import peak_memory_bytes
from jaxdft.pseudo.gth import hgh_lda

DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"
GRID = 8
# (buffer_bands, k_batch_size): the buffer scan at one batch, then chunk sizes
# dividing 512 at the default buffer. shephex options cannot be None, so
# k_batch_size = 0 stands for one batch (SCFOptions(k_batch_size=None)).
SETTINGS = [(b, 0) for b in (0, 1, 2, 3)] + [(1, c) for c in (64, 128, 256)]


def git_commit() -> str:
    source = Path(jaxdft.__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


@shephex.hexperiment(context=True)
def si_settings(
    buffer_bands: int, k_batch_size: int, context: ExperimentContext
) -> dict:
    context.add("stage", "setup")
    context.add("commit", git_commit())
    context.add("devices", [d.device_kind for d in jax.devices()])
    context.add("backend", jax.default_backend())
    si = diamond(10.263, "Si", unit="Bohr")
    kpoints, weights = monkhorst_pack((GRID, GRID, GRID))
    basis = PlaneWaveBasis.build(si.lattice, kpoints, 12.0, kweights=weights)
    terms = lda_terms({"Si": hgh_lda("Si")})
    options = SCFOptions(buffer_bands=buffer_bands, k_batch_size=k_batch_size or None)

    context.add("stage", "first scf")
    start = time.perf_counter()
    result = scf(si, basis, terms, options)
    energy = float(result.total_energy)
    first = time.perf_counter() - start

    context.add("stage", "steady scf")
    start = time.perf_counter()
    float(scf(si, basis, terms, options).total_energy)
    steady = time.perf_counter() - start

    h = result.history
    summary = {
        "n_kpoints": int(kpoints.shape[0]),
        "energy": energy,
        "n_iter": int(result.n_iter),
        "first_s": first,
        "steady_s": steady,
        "sum_eig_iters": float(np.sum(h["eig_iters"])),
        "sum_eig_iters_mean": float(np.sum(h["eig_iters_mean"])),
        "refined_steps": int(np.sum(h["refined"])),
        "eig_drops_max": int(np.max(h["eig_drops"])),
        "peak_device_memory_gb": (peak_memory_bytes() or 0) / 1e9,
    }
    for key, value in summary.items():
        context.add(key, value)
    context.add("stage", "done")
    return summary


if __name__ == "__main__":
    os.chdir(DATA)
    experiments = [
        si_settings(
            buffer_bands=b, k_batch_size=c, directory="si-eigensolver-settings"
        )
        for b, c in SETTINGS
    ]
    shephex.slurm_execute(
        experiments, profile="aicloud-jaxdft-v100", time="00:45:00", array_limit=4
    )
