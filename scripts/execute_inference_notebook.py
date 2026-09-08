"""Execute the canonical inference notebook against full staged competition inputs."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def execute(root: Path) -> dict[str, Any]:
    helpers = importlib.import_module("execute_notebooks")
    nbformat = importlib.import_module("nbformat")
    nbclient = importlib.import_module("nbclient")
    manager = importlib.import_module("jupyter_client.manager")
    root = root.resolve()
    source = root / "notebooks/10_competition_inference.ipynb"
    output = root / "artifacts/inference/notebooks"
    output.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    os.environ["OTTO_FULL_INFERENCE"] = "1"
    started = time.perf_counter()
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(15):
            print(
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "stage": "full_inference_notebook",
                        "elapsed_seconds": time.perf_counter() - started,
                        "prediction_part_receipts": len(
                            list((root / "artifacts/inference/prediction").glob("part-*.json"))
                        ),
                    }
                ),
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="otto-inference-kernel-") as directory:
            workspace = Path(directory)
            kernel = manager.KernelManager(
                kernel_name="python3", transport="ipc", ip=str(workspace / "kernel")
            )
            kernel.kernel_spec.argv = [
                sys.executable,
                "-Xfrozen_modules=off",
                "-m",
                "ipykernel_launcher",
                "-f",
                "{connection_file}",
            ]
            client = nbclient.NotebookClient(
                notebook,
                km=kernel,
                timeout=14000,
                allow_errors=False,
                force_raise_errors=True,
                ipython_hist_file=":memory:",
                resources={"metadata": {"path": str(root)}},
            )
            kernel_log = workspace / "kernel.log"
            with kernel_log.open("w") as stream:
                result = client.execute(cleanup_kc=True, stderr=stream)
            helpers.validate_kernel_log(kernel_log.read_text())
            code_cells = helpers.validate_outputs(result)
        destination = output / source.name
        helpers.atomic_json(destination, result)
        prediction = root / "artifacts/inference/prediction/manifest.json"
        report = {
            "status": "passed",
            "mode": "full competition inference",
            "source_sha256": helpers.sha256(source),
            "sha256": helpers.sha256(destination),
            "prediction_manifest_sha256": helpers.sha256(prediction),
            "code_cells": code_cells,
            "requirements_sha256": helpers.sha256(root / "notebooks/requirements.txt"),
            "python": sys.version,
            "completed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        helpers.atomic_json(output / "execution.json", report)
        return report
    finally:
        stop.set()
        thread.join(timeout=16)


if __name__ == "__main__":
    print(json.dumps(execute(Path(__file__).resolve().parents[1]), indent=2))
