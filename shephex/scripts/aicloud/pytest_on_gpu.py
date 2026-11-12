"""Run the jaxdft test suite on an aicloud V100.

Nothing in the suite is GPU-specific; running it on a GPU node checks that
everything that passes on CPU also passes with JAX's default device on the
GPU. The tests run in the jaxdft clone the profile's venv belongs to, which
must be synced with the dev and docs groups (`uv sync --extra cuda`). From the
repository root:

    uv run hex run --dry shephex/scripts/aicloud/pytest_on_gpu.py
    uv run hex run shephex/scripts/aicloud/pytest_on_gpu.py
    uv run hex report -p shephex/data/aicloud/pytest-gpu
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import jax
import shephex
from shephex.experiment import ExperimentContext

import jaxdft

# shephex maps the working directory at submission to the profile's remote
# directory, so submitting from here puts the runs directly under it.
DATA = Path(__file__).resolve().parents[2] / "data" / "aicloud"


@shephex.hexperiment(context=True)
def pytest_on_gpu(context: ExperimentContext) -> dict:
    # This process and the pytest subprocess share the GPU; JAX would
    # otherwise reserve 75 % of its memory in each.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    clone = Path(jaxdft.__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=clone,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    context.add("commit", commit)
    context.add("devices", [d.device_kind for d in jax.devices()])
    context.add("backend", jax.default_backend())
    context.add("stage", "pytest")

    output = context.directory / "pytest.txt"
    start = time.perf_counter()
    with output.open("w") as stream:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-ra", "-p", "no:cacheprovider"],
            cwd=clone,
            check=False,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
    elapsed = time.perf_counter() - start

    lines = output.read_text().splitlines()
    summary = lines[-1] if lines else ""
    counts = {word: int(number) for number, word in re.findall(r"(\d+) (\w+)", summary)}
    result = {
        "returncode": completed.returncode,
        "summary": summary,
        "counts": counts,
        "elapsed_s": elapsed,
    }
    for key, value in result.items():
        context.add(key, value)
    context.add("stage", "done")
    return result


if __name__ == "__main__":
    os.chdir(DATA)
    experiment = pytest_on_gpu(directory="pytest-gpu")
    shephex.slurm_execute(experiment, profile="aicloud-jaxdft-v100", time="02:00:00")
