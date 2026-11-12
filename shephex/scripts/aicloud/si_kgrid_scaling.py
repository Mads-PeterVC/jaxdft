"""Cost of SCF LDA silicon against the size of the k-grid, on an aicloud V100.

The Milestone 2b performance table (docs/engineering/performance.qmd) shows
the time per k-point and SCF step growing with the number of k-points,
although the k-points are one vmap batch. One experiment per Γ-centred n³
grid records the time of a first SCF (with compilation), a steady repeat, the
peak memory (host and GPU) and the machine, at the 12 Ha of that table, as
the biocloud CPU runs do. From the repository
root:

    uv run hex run --dry shephex/scripts/aicloud/si_kgrid_scaling.py
    uv run hex run shephex/scripts/aicloud/si_kgrid_scaling.py
    uv run hex report -p shephex/data/aicloud/si-kgrid-scaling
"""

import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import jax
import shephex
from shephex.experiment import ExperimentContext

import jaxdft
from jaxdft import PlaneWaveBasis, SCFOptions, diamond, lda_terms, monkhorst_pack, scf
from jaxdft.output.environment import peak_memory_bytes
from jaxdft.pseudo.gth import hgh_lda

# shephex maps the working directory at submission to the profile's remote
# directory, so submitting from here puts the runs directly under it.
DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"
GRIDS = (1, 2, 3, 4, 5, 6, 7, 8)


def git_commit() -> str:
    """Commit of the jaxdft checkout that is imported."""
    source = Path(jaxdft.__file__).resolve().parent
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()


def machine() -> dict:
    """Node, CPU model and the number of CPUs this process may use."""
    model = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    if hasattr(os, "sched_getaffinity"):
        visible = len(os.sched_getaffinity(0))
    else:
        visible = os.cpu_count()
    return {
        "node": socket.gethostname(),
        "cpu_model": model,
        "cpus_visible": visible,
        "cpus_node": os.cpu_count(),
        "devices": [d.device_kind for d in jax.devices()],
        "backend": jax.default_backend(),
    }


def peak_memory_gb() -> float:
    """Peak resident memory of this process (ru_maxrss: KiB on Linux, bytes
    on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1 if sys.platform == "darwin" else 1024
    return peak * scale / 1e9


@shephex.hexperiment(context=True)
def si_scf_timing(n: int, ecut: float, context: ExperimentContext) -> dict:
    context.add("stage", "setup")
    context.add("commit", git_commit())
    for key, value in machine().items():
        context.add(key, value)
    jaxdft.log_to_file(str(context.directory / "si_scf.log"))

    si = diamond(10.263, "Si", unit="Bohr")
    kpoints, weights = monkhorst_pack((n, n, n))
    basis = PlaneWaveBasis.build(si.lattice, kpoints, ecut, kweights=weights)
    terms = lda_terms({"Si": hgh_lda("Si")})
    context.add("n_kpoints", int(kpoints.shape[0]))
    context.add("n_pw_max", int(basis.n_pw.max()))
    context.add("fft_shape", list(map(int, basis.fft_shape)))

    context.add("stage", "first scf")
    start = time.perf_counter()
    result = scf(si, basis, terms, SCFOptions())
    energy = float(result.total_energy)
    first = time.perf_counter() - start
    context.add("first_s", first)

    context.add("stage", "steady scf")
    start = time.perf_counter()
    float(scf(si, basis, terms, SCFOptions()).total_energy)
    steady = time.perf_counter() - start

    n_iter = int(result.n_iter)
    summary = {
        "energy": energy,
        "n_iter": n_iter,
        "first_s": first,
        "steady_s": steady,
        "ms_per_k_step": 1e3 * steady / kpoints.shape[0] / n_iter,
        "peak_memory_gb": peak_memory_gb(),
        "peak_device_memory_gb": (peak_memory_bytes() or 0) / 1e9,
    }
    for key, value in summary.items():
        context.add(key, value)
    context.add("stage", "done")
    return summary


if __name__ == "__main__":
    os.chdir(DATA)
    experiments = [
        si_scf_timing(n=n, ecut=12.0, directory="si-kgrid-scaling") for n in GRIDS
    ]
    shephex.slurm_execute(experiments, profile="aicloud-jaxdft-v100", time="01:00:00")
