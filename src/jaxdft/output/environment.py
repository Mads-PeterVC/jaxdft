"""Machine and software details for the head of a log (host side, no physics)."""

from __future__ import annotations

import datetime
import functools
import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import jax

# Environment variables that change how or where a run executes.
ENVIRONMENT_VARIABLES = (
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_NTASKS",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "JAX_PLATFORMS",
    "XLA_FLAGS",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "OMP_NUM_THREADS",
)


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@functools.cache
def git_revision(path=None) -> str | None:
    """Short commit of the git work tree holding ``path`` (default: this package),
    with ``+dirty`` if it has uncommitted changes; None outside a work tree. Cached
    per process: the code a process runs is the code it imported."""
    where = Path(__file__).resolve().parent if path is None else Path(path)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(where), *args], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()

    try:
        commit = git("rev-parse", "--short=12", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.SubprocessError):
        return None
    return commit + ("+dirty" if dirty else "")


def machine_info() -> dict:
    """Software versions, host, JAX backend and devices, and the run-relevant
    environment variables that are set. All values are plain Python objects."""
    devices = jax.devices()
    kinds: dict[str, int] = {}
    for d in devices:
        name = d.platform if d.device_kind == d.platform else f"{d.platform} ({d.device_kind})"
        kinds[name] = kinds.get(name, 0) + 1
    return {
        "time": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "jaxdft": _version("jaxdft"),
        "git": git_revision(),
        "host": platform.node(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": _version("jaxlib"),
        "backend": jax.default_backend(),
        "devices": kinds,
        "processes": f"{jax.process_index()} of {jax.process_count()}",
        "cpu_count": os.cpu_count(),
        "x64": bool(jax.config.jax_enable_x64),
        "environment": {k: os.environ[k] for k in ENVIRONMENT_VARIABLES if k in os.environ},
    }


def peak_memory_bytes() -> int | None:
    """Largest ``peak_bytes_in_use`` over local devices, or None if the backend does
    not report it (CPU)."""
    peaks = []
    for d in jax.local_devices():
        stats = d.memory_stats() if hasattr(d, "memory_stats") else None
        if stats and "peak_bytes_in_use" in stats:
            peaks.append(int(stats["peak_bytes_in_use"]))
    return max(peaks) if peaks else None
