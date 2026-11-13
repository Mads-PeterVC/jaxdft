"""Collect the Milestone 2c eigensolver runs on the clusters into tracked JSON.

The runs are submitted by the scripts named below and pulled into
shephex/data/<cluster>/<directory>/, which is not tracked. From the repository
root:

    uv run python reference/eigensolver_2c/extract.py

writes, next to this script:

- lobpcg_check.json: the implemented lobpcg on the problems of the GPU
  residual diagnosis (aicloud, GPU and CPU of one node);
- settings_v100.json: buffer bands and k-chunk sizes on the 8^3 grid (V100);
- k_chunks.json: k-chunk sizes at three buffer bands, on the V100 and on
  biocloud CPUs.

Every experiment must have completed; the script refuses partial data.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "shephex" / "data"


def experiments(cluster: str, directory: str) -> list[dict]:
    rows = []
    paths = sorted((DATA / cluster / directory).glob("*-exp"))
    if not paths:
        sys.exit(f"No experiments in {DATA / cluster / directory}; pull them first")
    for path in paths:
        meta = json.loads((path / "shephex" / "meta.json").read_text())
        if meta["status"] != "completed":
            sys.exit(f"{path.name} has status {meta['status']!r}")
        context = json.loads((path / "shephex" / "context.json").read_text())
        options = json.loads((path / "shephex" / "options.json").read_text())
        row = {"cluster": cluster, "id": path.name, "job": context.get("job-id")}
        row.update(options["kwargs"])
        row.update({k: v for k, v in context.items() if k not in ("job-id", "stage", "progress")})
        rows.append(row)
    return rows


def write(name: str, description: str, scripts: list[str], rows: list[dict]) -> None:
    data = {
        "description": description,
        "generator": "reference/eigensolver_2c/extract.py",
        "submit_scripts": scripts,
        "experiments": rows,
    }
    (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} experiment(s) to reference/eigensolver_2c/{name}")


def main() -> None:
    write(
        "lobpcg_check.json",
        "Milestone 2c lobpcg on the GPU residual-diagnosis problems, V100 and CPU",
        ["shephex/scripts/aicloud/lobpcg_check_2c.py"],
        experiments("aicloud", "lobpcg-check-2c"),
    )
    write(
        "settings_v100.json",
        "Silicon 8^3 at 12 Ha on a V100: buffer bands, and k-chunks at one buffer band",
        ["shephex/scripts/aicloud/si_eigensolver_settings_scan.py"],
        sorted(
            experiments("aicloud", "si-eigensolver-settings"),
            key=lambda r: (r["k_batch_size"], r["buffer_bands"]),
        ),
    )
    write(
        "k_chunks.json",
        "Silicon 8^3 at 12 Ha, three buffer bands: k-chunk sizes on a V100 and on CPUs",
        [
            "shephex/scripts/aicloud/si_k_chunk_scan.py",
            "shephex/scripts/biocloud/si_k_chunk_scan.py",
        ],
        sorted(
            experiments("aicloud", "si-k-chunks") + experiments("biocloud", "si-k-chunks"),
            key=lambda r: (r["cluster"], r["k_batch_size"]),
        ),
    )


if __name__ == "__main__":
    main()
