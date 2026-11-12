"""Collect the two V100 diagnosis runs of the LOBPCG residual floor into one
tracked JSON file.

The runs are submitted by shephex/scripts/aicloud/gpu_residual_floor.py and
shephex/scripts/aicloud/gpu_lobpcg_dense_steps.py and pulled into
shephex/data/aicloud/, which is not tracked. Their results are the shephex
contexts, which this script copies into gpu_lobpcg_floor.json.

The per-iteration trajectories of shephex/scripts/aicloud/lobpcg_trajectory.py
go to lobpcg_trajectory.json, rounded to four significant digits. Its
``gram_min_rel`` column is left out: the script pairs Gram eigenvalues with
column norms by index, which does not identify the zero columns, so the
column is meaningless. The same script run on the laptop with ``--local``
(CPU only) is added when its output exists at
shephex/data/laptop/lobpcg-trajectory/trajectories.json:

    uv run python shephex/scripts/aicloud/lobpcg_trajectory.py --local \
        shephex/data/laptop/lobpcg-trajectory/trajectories.json

From the repository root:

    uv run python reference/gpu_lobpcg_floor/extract.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "shephex" / "data" / "aicloud"
OUTPUT = Path(__file__).resolve().parent / "gpu_lobpcg_floor.json"
RUNS = {
    "building_blocks": ("gpu-residual-floor", "gpu_residual_floor.py"),
    "dense_steps": ("gpu-lobpcg-dense-steps", "gpu_lobpcg_dense_steps.py"),
}


def run(directory: str, script: str) -> dict:
    experiments = sorted((DATA / directory).glob("*-exp"))
    if len(experiments) != 1:
        sys.exit(f"Expected one experiment in {DATA / directory}")
    shephex_dir = experiments[0] / "shephex"
    meta = json.loads((shephex_dir / "meta.json").read_text())
    if meta["status"] != "completed":
        sys.exit(f"{experiments[0].name} has status {meta['status']!r}")
    context = json.loads((shephex_dir / "context.json").read_text())
    return {
        "id": experiments[0].name,
        "submit_script": f"shephex/scripts/aicloud/{script}",
        "result": context,
    }


TRAJECTORY = Path(__file__).resolve().parent / "lobpcg_trajectory.json"
SKIP = {"gram_min_rel"}


def rounded(x: float):
    return None if math.isnan(x) else float(f"{x:.4g}")


def trajectory() -> dict:
    experiments = sorted((DATA / "lobpcg-trajectory").glob("*-exp"))
    if len(experiments) != 1:
        sys.exit(f"Expected one experiment in {DATA / 'lobpcg-trajectory'}")
    exp = experiments[0]
    meta = json.loads((exp / "shephex" / "meta.json").read_text())
    if meta["status"] != "completed":
        sys.exit(f"{exp.name} has status {meta['status']!r}")
    context = json.loads((exp / "shephex" / "context.json").read_text())
    runs = compact(json.loads((exp / "trajectories.json").read_text()))
    out = {
        "description": (
            "Per-iteration LOBPCG record on a V100 and the CPU of the same"
            " node: residual, dropped directions per orthonormalisation pass,"
            " max|T| with the penalty, Rayleigh-Ritz backward error, penalty"
            " value. Labels are system/device/variant/mode/seed. ``runs``: aicloud"
            " V100 node; ``laptop_runs``: Apple-silicon laptop, CPU only."
        ),
        "generator": "reference/gpu_lobpcg_floor/extract.py",
        "submit_script": "shephex/scripts/aicloud/lobpcg_trajectory.py",
        "id": exp.name,
        "job": context.get("job-id"),
        "devices": context.get("devices"),
        "eig_tol": context.get("eig_tol"),
        "runs": runs,
    }
    laptop = ROOT / "shephex" / "data" / "laptop" / "lobpcg-trajectory"
    if (laptop / "trajectories.json").exists():
        out["laptop_runs"] = compact(
            json.loads((laptop / "trajectories.json").read_text())
        )
    return out


def compact(runs: dict) -> dict:
    for run in runs.values():
        run["true_residual_max"] = rounded(run["true_residual_max"])
        run["trace"] = {
            name: [rounded(v) for v in values]
            for name, values in run["trace"].items()
            if name not in SKIP
        }
    return runs


def main() -> None:
    data = {
        "description": (
            "LOBPCG residual floor on a V100: FFT, apply_H, eigh and LOBPCG"
            " variants against the CPU of the same node"
        ),
        "generator": "reference/gpu_lobpcg_floor/extract.py",
        "runs": {name: run(*where) for name, where in RUNS.items()},
    }
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    TRAJECTORY.write_text(json.dumps(trajectory(), ensure_ascii=False) + "\n")
    print(f"Wrote {TRAJECTORY.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
