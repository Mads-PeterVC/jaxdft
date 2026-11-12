"""Collect the biocloud silicon k-grid scaling runs into one tracked JSON file.

The runs are submitted by shephex/scripts/biocloud/si_kgrid_scaling.py and
pulled into shephex/data/biocloud/si-kgrid-scaling/, which is not tracked.
This script reads each experiment's shephex context and jaxdft log and writes
si_kgrid_scaling.json next to it. From the repository root:

    uv run python reference/biocloud_si_kgrid/extract.py

Every experiment must have completed; the script refuses partial data.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "shephex" / "data" / "biocloud" / "si-kgrid-scaling"
OUTPUT = Path(__file__).resolve().parent / "si_kgrid_scaling.json"

CONTEXT_KEYS = (
    "commit",
    "node",
    "cpu_model",
    "cpus_visible",
    "cpus_node",
    "n_kpoints",
    "n_pw_max",
    "fft_shape",
    "n_iter",
    "energy",
    "first_s",
    "steady_s",
    "peak_memory_gb",
)


def eigensolver_iterations(log: str) -> list[list[int]]:
    """Worst eigensolver iterations over k-points, per SCF step, for each SCF
    run in the log (the first run and the steady repeat)."""
    runs = []
    for section in log.split("== SCF iterations")[1:]:
        table = section.split("== Result")[0].splitlines()
        header = next(i for i, line in enumerate(table) if line.startswith("iter"))
        names = table[header].split("  ")
        names = [name.strip() for name in names if name.strip()]
        column = names.index("eig it")
        steps = []
        for line in table[header + 1 :]:
            fields = line.split()
            if not fields:
                continue
            # "dE" is "-" on step 0, so every row has the same field count.
            assert len(fields) == len(names), line
            steps.append(int(fields[column]))
        runs.append(steps)
    return runs


def experiment(directory: Path) -> dict:
    meta = json.loads((directory / "shephex" / "meta.json").read_text())
    if meta["status"] != "completed":
        sys.exit(f"{directory.name} has status {meta['status']!r}")
    context = json.loads((directory / "shephex" / "context.json").read_text())
    options = json.loads((directory / "shephex" / "options.json").read_text())
    runs = eigensolver_iterations((directory / "si_scf.log").read_text())
    if len(runs) != 2:
        sys.exit(f"{directory.name}: expected 2 SCF runs in the log")
    row = {"id": directory.name, "job": context["job-id"]}
    row.update(options["kwargs"])
    row.update({key: context[key] for key in CONTEXT_KEYS})
    row["eig_iterations_first"] = runs[0]
    row["eig_iterations_steady"] = runs[1]
    return row


def main() -> None:
    directories = sorted(SOURCE.glob("*-exp"))
    if not directories:
        sys.exit(f"No experiments in {SOURCE}; pull them with hex remote pull")
    rows = sorted(map(experiment, directories), key=lambda r: r["n_kpoints"])
    data = {
        "description": (
            "SCF LDA silicon on Γ-centred n³ k-grids, timed on biocloud"
        ),
        "generator": "reference/biocloud_si_kgrid/extract.py",
        "submit_script": "shephex/scripts/biocloud/si_kgrid_scaling.py",
        "experiments": rows,
    }
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} experiments to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
