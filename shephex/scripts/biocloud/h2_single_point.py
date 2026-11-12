"""Cluster pilot: the H2 single point of examples/h2_single_point.py on Slurm.

Checks that the remote environment, the profile and the result transfer work
before anything larger is submitted. From the repository root:

    uv run hex run --dry shephex/scripts/biocloud/h2_single_point.py
    uv run hex run shephex/scripts/biocloud/h2_single_point.py
    uv run hex report -p shephex/data/biocloud/h2-pilot
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import shephex
from shephex.experiment import ExperimentContext

import jaxdft
from jaxdft import PlaneWaveBasis, SCFOptions, dimer_in_box, lda_terms, scf
from jaxdft.pseudo.gth import hgh_lda


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


# shephex maps the working directory at submission to the profile's remote
# directory, so submitting from here puts h2-pilot/ directly under it.
DATA = Path(__file__).resolve().parents[2] / "data" / "biocloud"


@shephex.hexperiment(context=True)
def h2_single_point(
    bond_length: float, box: float, ecut: float, context: ExperimentContext
) -> dict:
    context.add("stage", "setup")
    context.add("commit", git_commit())
    context.add("cpus_visible", len(os.sched_getaffinity(0)))
    context.add("cpus_node", os.cpu_count())
    jaxdft.log_to_file(str(context.directory / "h2_single_point.log"))

    structure = dimer_in_box("H", bond_length=bond_length, box=box)
    gamma = np.zeros((1, 3))
    basis = PlaneWaveBasis.build(structure.lattice, kpoints=gamma, ecut=ecut)
    terms = lda_terms({"H": hgh_lda("H")})

    context.add("stage", "scf")
    result = scf(structure, basis, terms, SCFOptions(tol=1e-9))

    energy = float(result.total_energy)
    n_iter = int(result.n_iter)
    context.add("stage", "done")
    context.add("energy", energy)
    context.add("n_iter", n_iter)
    return {"total_energy": energy, "n_iter": n_iter}


if __name__ == "__main__":
    os.chdir(DATA)
    experiment = h2_single_point(
        bond_length=1.45, box=10.0, ecut=10.0, directory="h2-pilot"
    )
    shephex.slurm_execute(experiment, profile="biocloud-jaxdft-cpu", time="00:10:00")
