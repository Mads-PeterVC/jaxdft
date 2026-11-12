"""Collect the jaxdft test suite run on an aicloud V100 into a tracked JSON file.

The run is submitted by shephex/scripts/aicloud/pytest_on_gpu.py and pulled
into shephex/data/aicloud/pytest-gpu/, which is not tracked. This script reads
each experiment's shephex context and pytest output and writes
pytest_gpu.json next to it. From the repository root:

    uv run python reference/aicloud_pytest/extract.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "shephex" / "data" / "aicloud" / "pytest-gpu"
OUTPUT = Path(__file__).resolve().parent / "pytest_gpu.json"

SECTION = re.compile(r"^_+ (?:ERROR at setup of )?(\S+) _+$")
OUTCOME = re.compile(r"^(FAILED|ERROR) (\S+)")


def error_lines(text: str) -> dict[str, str]:
    """Last exception line ('E   SomeError: ...') of each failure section, by
    the test name the section header shows."""
    errors, name = {}, None
    for line in text.splitlines():
        header = SECTION.match(line)
        if header:
            name = header.group(1)
        elif name and re.match(r"^E\s+\S*(Error|Exception)\b", line):
            errors[name] = line[1:].strip()
    return errors


def experiment(directory: Path) -> dict:
    meta = json.loads((directory / "shephex" / "meta.json").read_text())
    if meta["status"] != "completed":
        sys.exit(f"{directory.name} has status {meta['status']!r}")
    context = json.loads((directory / "shephex" / "context.json").read_text())
    text = (directory / "pytest.txt").read_text()
    errors = error_lines(text)
    problems = []
    for line in text.splitlines():
        match = OUTCOME.match(line)
        if match:
            outcome, nodeid = match.groups()
            name = nodeid.split("::")[-1]
            problems.append(
                {"outcome": outcome, "test": nodeid, "error": errors.get(name)}
            )
    return {
        "id": directory.name,
        "job": context["job-id"],
        "commit": context["commit"],
        "devices": context["devices"],
        "backend": context["backend"],
        "returncode": context["returncode"],
        "summary": context["summary"],
        "counts": context["counts"],
        "elapsed_s": context["elapsed_s"],
        "problems": problems,
    }


def main() -> None:
    directories = sorted(SOURCE.glob("*-exp"))
    if not directories:
        sys.exit(f"No experiments in {SOURCE}; pull them with hex remote pull")
    data = {
        "description": "jaxdft test suite on an aicloud V100 (JAX default device)",
        "generator": "reference/aicloud_pytest/extract.py",
        "submit_script": "shephex/scripts/aicloud/pytest_on_gpu.py",
        "runs": [experiment(d) for d in directories],
    }
    OUTPUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(data['runs'])} run(s) to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
