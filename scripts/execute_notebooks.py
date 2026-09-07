"""Execute the committed analysis notebooks in isolated kernels and resumable copies."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

WARNING = re.compile(r"\b(?:\w*Warning):|\bWARNING\s*[|:]")
INPUT_DIRECTORIES = ("reports", "configs")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=1, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def validate_outputs(notebook: dict[str, Any]) -> int:
    """Reject partial runs, error outputs and Python warnings; retain real outputs."""
    count = 0
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code" or not "".join(cell["source"]).strip():
            continue
        if not isinstance(cell.get("execution_count"), int):
            raise ValueError("notebook contains an unexecuted code cell")
        count += 1
        for output in cell.get("outputs", []):
            if output["output_type"] == "error":
                raise ValueError("notebook contains an error output")
            if output["output_type"] == "stream" and WARNING.search("".join(output["text"])):
                raise ValueError("notebook contains a Python warning")
    if not count:
        raise ValueError("notebook has no executable code cells")
    return count


def valid_receipt(output: Path, input_id: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads(output.with_suffix(".json").read_text())
        if receipt["input_id"] != input_id or receipt["sha256"] != sha256(output):
            return None
        if validate_outputs(json.loads(output.read_text())) != receipt["code_cells"]:
            return None
        return dict(receipt)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def input_checksums(root: Path) -> dict[str, str]:
    """Hash the same evidence/configuration trees copied into execution workspaces."""
    checksums = {}
    for name in INPUT_DIRECTORIES:
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                checksums[str(path.relative_to(root))] = sha256(path)
    return checksums


def prepare_workspace(root: Path, workspace: Path) -> None:
    """Copy compact inputs, not training data; notebook writes stay isolated."""
    for name in INPUT_DIRECTORIES:
        shutil.copytree(root / name, workspace / name)
    (workspace / "notebooks").mkdir()


def validate_kernel_log(text: str) -> None:
    """Kernel startup warnings live outside notebook output and must also fail."""
    if WARNING.search(text):
        raise ValueError(f"kernel emitted a warning: {text.strip()}")


def _execute(root: Path, output: Path, *, timeout: int) -> dict[str, Any]:
    """Keep source/evidence untouched; commit each successful notebook independently."""
    if timeout <= 0:
        raise ValueError("cell timeout must be positive")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = threading.Lock()

    def emit(event: str, **fields: Any) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), "event": event,
                  "total_elapsed_seconds": round(time.perf_counter() - started, 3), **fields}
        text = json.dumps(record, sort_keys=True)
        with lock:
            print(text, flush=True)
            with (output / "execution.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    nbformat = importlib.import_module("nbformat")
    nbclient = importlib.import_module("nbclient")
    manager = importlib.import_module("jupyter_client")
    runtime = {distribution.metadata["Name"]: distribution.version
               for distribution in distributions()}
    runtime["python"] = sys.version
    dependencies = input_checksums(root)
    dependencies["notebooks/requirements.txt"] = sha256(root / "notebooks/requirements.txt")
    dependencies["executor"] = sha256(Path(__file__))
    sources = sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb"))
    if not sources:
        raise ValueError("no canonical analysis notebooks found")
    receipts = []
    reused = 0
    for source in sources:
        stage_started = time.perf_counter()
        notebook = nbformat.read(source, as_version=4)
        identity = {"runtime": runtime, "dependencies": dependencies, "timeout": timeout,
                    "cells": [(c.cell_type, c.source) for c in notebook.cells]}
        input_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        destination = output / source.name
        receipt = valid_receipt(destination, input_id)
        if receipt is not None:
            reused += 1
            receipts.append(receipt)
            emit("notebook_reused", notebook=source.name, code_cells=receipt["code_cells"])
            continue
        emit("notebook_start", notebook=source.name)
        stop = threading.Event()

        def heartbeat(
            stop_event: threading.Event = stop,
            notebook_name: str = source.name,
            beginning: float = stage_started,
        ) -> None:
            while not stop_event.wait(15):
                emit("heartbeat", notebook=notebook_name,
                     stage_elapsed_seconds=round(time.perf_counter() - beginning, 3))

        thread = threading.Thread(target=heartbeat, name="notebook-heartbeat", daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory(prefix="otto-notebooks-") as temporary:
                workspace = Path(temporary)
                prepare_workspace(root, workspace)
                # Bind the kernel to this exact analysis interpreter, not a user's default kernel.
                # Private filesystem sockets avoid exposing local outputs over plaintext TCP.
                kernel = manager.KernelManager(
                    kernel_name="python3", transport="ipc", ip=str(workspace / "kernel"),
                )
                kernel.kernel_spec.argv = [
                    sys.executable, "-Xfrozen_modules=off", "-m", "ipykernel_launcher",
                    "-f", "{connection_file}",
                ]
                client = nbclient.NotebookClient(
                    notebook, km=kernel, timeout=timeout, allow_errors=False,
                    force_raise_errors=True, ipython_hist_file=":memory:",
                    resources={"metadata": {"path": str(workspace / "notebooks")}},
                )
                kernel_log = workspace / "kernel.log"
                try:
                    with kernel_log.open("w", encoding="utf-8") as kernel_stderr:
                        result = client.execute(cleanup_kc=True, stderr=kernel_stderr)
                finally:
                    startup = kernel_log.read_text()
                    if startup.strip():
                        emit("kernel_stderr", notebook=source.name, text=startup)
                validate_kernel_log(startup)
                cells = validate_outputs(result)
                atomic_json(destination, result)
                receipt = {
                    "input_id": input_id, "sha256": sha256(destination), "code_cells": cells,
                    "notebook": source.name, "runtime": runtime,
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "compute_seconds": round(time.perf_counter() - stage_started, 3),
                }
                atomic_json(destination.with_suffix(".json"), receipt)
                receipts.append(receipt)
                emit("notebook_complete", notebook=source.name, code_cells=cells,
                     stage_elapsed_seconds=receipt["compute_seconds"])
        except BaseException as error:
            emit("notebook_failed", notebook=source.name, error=str(error),
                 stage_elapsed_seconds=round(time.perf_counter() - stage_started, 3))
            raise
        finally:
            stop.set()
            thread.join(timeout=16)
    summary = {"status": "passed", "notebooks": len(receipts), "reused_notebooks": reused,
               "code_cells": sum(r["code_cells"] for r in receipts),
               "attempt_elapsed_seconds": round(time.perf_counter() - started, 3),
               "retained_compute_seconds": sum(r["compute_seconds"] for r in receipts),
               "receipts": receipts}
    atomic_json(output / "manifest.json", summary)
    emit("notebooks_complete", notebooks=len(receipts), reused_notebooks=reused,
         code_cells=summary["code_cells"])
    return summary


def execute(root: Path, output: Path, *, timeout: int) -> dict[str, Any]:
    """Prevent concurrent writers and accidental modification of input evidence."""
    root, output = root.resolve(), output.resolve()
    if any(output.is_relative_to(root / name) for name in ("notebooks", *INPUT_DIRECTORIES)):
        raise ValueError("execution outputs must be separate from source notebooks and evidence")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("notebook output already has an active writer") from error
        try:
            return _execute(root, output, timeout=timeout)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/notebooks"))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    execute(Path(__file__).resolve().parents[1], args.output_dir.resolve(), timeout=args.timeout)


if __name__ == "__main__":
    main()
