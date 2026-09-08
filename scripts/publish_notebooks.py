"""Publish verified execution copies to canonical paths without changing source cells.

This command does not execute kernels or call Git. A results-branch workflow runs
all quality gates and the executor before invoking it, then stages only notebooks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

WARNING = re.compile(r"\b(?:\w*Warning):|\bWARNING\s*[|:]")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def runtime_versions() -> dict[str, str]:
    import sys

    versions = {package.metadata["Name"]: package.version for package in distributions()}
    versions["python"] = sys.version
    return versions


def dependencies(root: Path) -> dict[str, str]:
    result = {}
    for directory in ("reports", "configs"):
        if not (root / directory).is_dir():
            raise FileNotFoundError(root / directory)
        for path in sorted((root / directory).rglob("*")):
            if path.is_file():
                result[str(path.relative_to(root))] = sha256(path)
    result["notebooks/requirements.txt"] = sha256(root / "notebooks/requirements.txt")
    result["executor"] = sha256(root / "scripts/execute_notebooks.py")
    return result


def cell_sources(notebook: dict[str, Any]) -> list[tuple[str, str]]:
    return [(cell["cell_type"], "".join(cell["source"])) for cell in notebook["cells"]]


def identity(
    notebook: dict[str, Any], inputs: dict[str, str], runtime: dict[str, str], timeout: int,
) -> str:
    value = {"runtime": runtime, "dependencies": inputs, "timeout": timeout,
             "cells": cell_sources(notebook)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_outputs(notebook: dict[str, Any]) -> int:
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        if type(cell.get("execution_count")) is not int:
            raise ValueError("Cannot publish unexecuted code cells")
        count += 1
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                raise ValueError("Cannot publish a notebook error output")
            if output["output_type"] == "stream" and WARNING.search("".join(output["text"])):
                raise ValueError("Cannot publish a notebook warning")
    if count == 0:
        raise ValueError("Cannot publish an empty notebook")
    return count


def atomic_bytes(path: Path, data: bytes) -> None:
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def publish(root: Path, executed: Path, *, timeout: int = 300) -> dict[str, Any]:
    """Validate the entire batch before copying any bytes; receipts are committed last."""
    root, executed = root.resolve(), executed.resolve()
    if timeout <= 0 or any(executed.is_relative_to(root / path)
                           for path in ("notebooks", "reports", "configs")):
        raise ValueError("Execution outputs must be separate, with a positive cell timeout")
    sources = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    manifest = load(executed / "manifest.json")
    if (not sources or manifest["status"] != "passed"
            or manifest["notebooks"] != len(sources)
            or manifest["reused_notebooks"] != len(sources)):
        raise ValueError("Require successful execution and verified reuse of every notebook")
    receipts = manifest["receipts"]
    names = [path.name for path in sources]
    if len(receipts) != len(names) or sorted(receipt["notebook"] for receipt in receipts) != names:
        raise ValueError("Execution manifest has unexpected, missing, or duplicate notebooks")
    by_name = {receipt["notebook"]: receipt for receipt in receipts}
    inputs, runtime = dependencies(root), runtime_versions()
    prepared: list[tuple[Path, bytes, dict[str, Any]]] = []
    for source in sources:
        result_path = executed / source.name
        if source.is_symlink() or result_path.is_symlink():
            raise ValueError("Canonical notebooks must be ordinary files, not symlinks")
        receipt = load(result_path.with_suffix(".json"))
        original = load(source)
        raw = result_path.read_bytes()
        result = json.loads(raw)
        if (receipt != by_name[source.name]
                or receipt["notebook"] != source.name
                or receipt["sha256"] != hashlib.sha256(raw).hexdigest()
                or receipt["input_id"] != identity(original, inputs, runtime, timeout)
                or receipt["runtime"] != runtime
                or cell_sources(result) != cell_sources(original)
                or validate_outputs(result) != receipt["code_cells"]):
            raise ValueError(f"Execution/source/evidence mismatch: {source.name}")
        prepared.append((source, raw, receipt))
    total_cells = sum(receipt["code_cells"] for _, _, receipt in prepared)
    if manifest["code_cells"] != total_cells:
        raise ValueError("Execution manifest code-cell total is inconsistent")
    record = {
        "status": "passed", "published_at_utc": datetime.now(UTC).isoformat(),
        "notebooks": len(prepared), "code_cells": total_cells,
        "retained_compute_seconds": manifest["retained_compute_seconds"],
        "verification": "Source cells, input/runtime fingerprints, output bytes and reuse verified",
        "receipts": [receipt for _, _, receipt in prepared],
    }
    for source, raw, _ in prepared:
        atomic_bytes(source, raw)
    # Kept outside reports/configs, so publishing the receipt does not invalidate input identity.
    atomic_bytes(root / "notebooks/execution.json",
                 (json.dumps(record, indent=2, sort_keys=True) + "\n").encode())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/notebooks"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    result = publish(Path(__file__).resolve().parents[1], args.output_dir, timeout=args.timeout)
    print(f"OTTO_NOTEBOOKS_PUBLISHED notebooks={result['notebooks']} cells={result['code_cells']}")


if __name__ == "__main__":
    main()
