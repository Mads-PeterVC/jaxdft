"""Collect the two V100 diagnosis runs of the LOBPCG residual floor into one
tracked JSON file.

The runs are submitted by shephex/scripts/aicloud/gpu_residual_floor.py and
shephex/scripts/aicloud/gpu_lobpcg_dense_steps.py and pulled into
shephex/data/aicloud/, which is not tracked. Their results are the shephex
contexts, which this script copies into gpu_lobpcg_floor.json. From the
repository root:

    uv run python reference/gpu_lobpcg_floor/extract.py
"""

import json
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


if __name__ == "__main__":
    main()
