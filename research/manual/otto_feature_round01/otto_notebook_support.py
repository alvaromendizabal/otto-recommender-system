"""Read-only Round 01 review: explicit Plotly MIME, no training, no auto-renderers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KERNEL_NAME = "otto-notebook"
KERNEL_DISPLAY = "OTTO - Notebook"
LABELS = {
    "control_shared": "134-feature saved control",
    "plus_support": "+24 support features",
    "plus_timing": "+24 timing features",
    "plus_both": "+48 combined features",
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlink output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def verify_review(root: Path) -> dict[str, Any]:
    contract = json.loads((root / "otto_notebook_review_contract.json").read_text())
    for relative, expected in contract["files"].items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError(f"Unsafe reviewed input: {relative}")
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Reviewed output missing or changed: {relative}. Preserve it for review.")
    result = json.loads((root / "outputs/result.json").read_text())
    if result["status"] != "ROUND01_SCREEN_COMPLETED":
        raise ValueError("This viewer requires a completed Round 01 result")
    if result["source_commit"] != contract["source_commit"]:
        raise ValueError("Result source differs from reviewed source")
    return result


def collect(root: Path) -> Path:
    """Small diagnostic bundle only. Never includes datasets, models, connection files or tokens."""
    directory = root / "outputs/notebook_support"
    directory.mkdir(parents=True, exist_ok=True)
    paths = sorted(directory.glob("*.json"))
    for path in paths:
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError("Unexpected diagnostic file size or symlink")
    destination = root / "otto_notebook_check.zip"
    if destination.is_symlink():
        raise ValueError("Diagnostic ZIP destination is a symlink")
    temp = destination.with_name(destination.name + f".{os.getpid()}.tmp")
    try:
        with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            entries = []
            for path in paths:
                name = path.relative_to(root).as_posix()
                archive.write(path, name)
                entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha(path)})
            archive.writestr("MANIFEST.json", json.dumps(entries, indent=2))
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)
    return destination


class SavedReview:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.directory = self.root / "outputs/notebook_support"
        self.result: dict[str, Any] = {}
        self.state: dict[str, Any] = {}
        self.started = time.monotonic()

    def save(self) -> None:
        self.state["updated_utc"] = utc()
        self.state["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        write_json(self.directory / "notebook_session.json", self.state)
        collect(self.root)

    def start(self) -> None:
        self.result = verify_review(self.root)
        versions = {}
        for package in ("ipykernel", "ipython", "jupyter_client", "plotly"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        setup = json.loads((self.directory / "setup.json").read_text())
        expected = setup["selected_notebook_python"]
        # Keep venv paths lexical: resolving their Python symlink loses environment identity.
        if os.path.abspath(sys.executable) != os.path.abspath(expected):
            raise RuntimeError(f"Select {KERNEL_DISPLAY}. Current: {sys.executable}; expected: {expected}")
        self.state = {
            "status": "REVIEW_STARTED", "started_utc": utc(), "notebook_python": sys.executable,
            "notebook_versions": versions, "ml_python": setup["ml_python"],
            "kernel_spec": KERNEL_NAME, "renderer": "application/vnd.plotly.v1+json (explicit raw MIME)",
            "new_model_fits": 0, "feature_rebuilds": 0, "package_installs": 0,
            "chart_payloads_emitted": [], "browser_render_verified": False,
            "result_sha256": sha(self.root / "outputs/result.json"),
        }
        self.save()
        print("KERNEL_READY", flush=True)
        print("Notebook Python:", sys.executable)
        print("ML Python (unchanged; not started by this notebook):", setup["ml_python"])
        print("Mode: saved results only. No run_stage calls, model fits, or data downloads.")

    def summary(self) -> None:
        if not self.result:
            raise RuntimeError("Run the first cell before the summary")
        for arm, label in LABELS.items():
            item = self.result["arms"][arm]
            print(f"{label:28s} {item['weighted_recall_at_20']:.6f}  hits={item['hits']}")
        timing = self.result["comparisons"]["plus_timing__minus__control_shared"]
        print("Timing gain:", f"{timing['gain']:+.6f}")
        print("Descriptive interval:", timing["descriptive_95_interval"])
        print("Both forward-fold gains:", timing["fold_gains"])
        print("Primary combined arm failed its gate. Timing-only remains a replication hypothesis.")
        print("Fitting-only result; not a Kaggle score. No automatic feature retention.")

    def show_chart(self, number: int) -> None:
        if type(number) is not int or not 1 <= number <= 7:
            raise ValueError("Chart number must be an integer from 1 through 7")
        if not self.state:
            raise RuntimeError("Run the first cell before displaying charts")
        relative = f"outputs/plots/{number:02d}.json"
        contract = json.loads((self.root / "otto_notebook_review_contract.json").read_text())
        path = self.root / relative
        if path.stat().st_size > 512 * 1024 or sha(path) != contract["files"][relative]:
            raise ValueError("Chart file changed or exceeds the display size limit")
        figure = json.loads(path.read_text())
        if not isinstance(figure.get("data"), list) or not isinstance(figure.get("layout"), dict):
            raise ValueError("Saved chart is not a Plotly figure specification")
        title = figure["layout"].get("title", {}).get("text", f"Round 01 chart {number}")
        from IPython.display import display
        # Do not call display(Figure), fig.show(), FigureWidget, browser, iframe or HTML renderers.
        # JupyterLab receives only the existing small JSON figure. No JavaScript bundle is embedded.
        display({"application/vnd.plotly.v1+json": figure,
                 "text/plain": f"{title}. Interactive fallback: download outputs/round01_report.html."},
                raw=True)
        emitted = self.state["chart_payloads_emitted"]
        if number not in emitted:
            emitted.append(number)
        self.state["last_chart"] = number
        self.save()
        print(f"CHART_{number}_PAYLOAD_SENT — Python display call returned.", flush=True)

    def finish(self) -> None:
        if sorted(self.state.get("chart_payloads_emitted", [])) != list(range(1, 8)):
            raise ValueError("Not all seven chart payloads have been emitted")
        self.state["status"] = "NOTEBOOK_REVIEW_COMPLETE"
        self.state["code_finished_utc"] = utc()
        self.save()
        print("NOTEBOOK_REVIEW_COMPLETE", flush=True)
        print("No model training, package installation, or feature construction was performed.")
        print("Browser rendering is separate; this marker proves that the final Python cell ran.")
        print("RETURN_FILE:", self.root / "otto_notebook_check.zip")
