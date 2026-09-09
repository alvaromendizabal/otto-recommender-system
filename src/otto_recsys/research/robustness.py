"""Frozen robustness plans and guarded reuse of the reference window's inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tomllib
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

FROZEN_MODULES = {
    "configuration.py",
    "protocol.py",
    "dataset.py",
    "retrievers.py",
    "features.py",
    "screening.py",
    "materialize.py",
    "training.py",
    "study.py",
    "evaluation.py",
    "metrics.py",
}
FROZEN_FILES = {"configs/research.toml", "pyproject.toml", "uv.lock"}


def fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_plan(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the precommitted method and original input inventory before reuse."""
    plan = tomllib.loads((root / "configs/robustness.toml").read_text())
    reference = json.loads((root / "reports/robustness/reference_launch.json").read_text())
    if plan["schema_version"] != 1 or fingerprint(reference) != plan["reference_launch_digest"]:
        raise ValueError("robustness reference input inventory differs from the frozen protocol")
    seeds = plan["model_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("robustness requires three distinct nonnegative integer model seeds")
    if plan["cohort_seed"] != reference["training"]["seed"]:
        raise ValueError("cohort seed must preserve the original query sampling")
    expected_training = {k: v for k, v in reference["training"].items() if k != "seed"}
    if plan["training"] != expected_training:
        raise ValueError("robustness changes the frozen ranker settings")
    if set(plan["frozen_code"]) != FROZEN_MODULES or set(plan["frozen_files"]) != FROZEN_FILES:
        raise ValueError("frozen research implementation inventory is incomplete")
    for name, expected in plan["frozen_files"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen research configuration changed: {name}")
    for name, expected in plan["frozen_code"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("invalid frozen research source path")
        source = root / "src/otto_recsys/research" / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen research implementation changed: {name}")
    windows = plan["windows"]
    if len(windows) != 3 or len({w["name"] for w in windows}) != 3:
        raise ValueError("robustness requires three distinct temporal windows")
    intervals = []
    reference_window = None
    for window in windows:
        if re.fullmatch(r"[a-z][a-z0-9_]*", window["name"]) is None:
            raise ValueError("invalid robustness window name")
        points = [
            datetime.fromisoformat(window[k])
            for k in ("history_end", "fit_end", "selection_end", "evaluation_end")
        ]
        if any(t.tzinfo is None or t.utcoffset() != UTC.utcoffset(t) for t in points):
            raise ValueError("robustness boundaries must use UTC")
        widths = [(b - a).total_seconds() for a, b in pairwise(points)]
        if widths != [3 * 86400, 86400, 2 * 86400]:
            raise ValueError(
                "robustness windows require 3-day fit, 1-day selection, 2-day evaluation"
            )
        intervals.append((points[2], points[3]))
        if window["name"] == plan["reference_window"]:
            reference_window = window
    ordered = sorted(intervals)
    if any(a[1] > b[0] for a, b in pairwise(ordered)):
        raise ValueError("robustness evaluation intervals must not overlap")
    original = tomllib.loads((root / "configs/research.toml").read_text())["protocol"]
    if reference_window is None or any(
        reference_window[key] != original[key]
        for key in ("history_end", "fit_end", "selection_end", "evaluation_end")
    ):
        raise ValueError("reference window differs from the completed experiment")
    for key in ("fit_sessions", "selection_sessions"):
        if plan[key] != original[key]:
            raise ValueError("robustness changes a frozen session budget")
    if plan["candidate_budget"] != 400 or plan["bootstrap_replicates"] != 1000:
        raise ValueError("robustness changes candidate coverage or bootstrap precision")
    if plan["bootstrap_seed"] != plan["cohort_seed"]:
        raise ValueError("bootstrap seed must remain fixed independently of model seeds")
    execution = plan["execution"]
    if execution["maximum_concurrent_jobs"] != 1 or execution["maximum_runtime_seconds"] != 7200:
        raise ValueError("reference replication exceeds the frozen resource bounds")
    return plan, reference


def cells(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "cell_id": f"{window['name']}_seed_{seed}",
            "window": window["name"],
            "model_seed": seed,
            "cohort_seed": plan["cohort_seed"],
        }
        for window in plan["windows"]
        for seed in plan["model_seeds"]
    ]


def seed_launch(
    root: Path,
    cell_id: str,
    *,
    source_commit: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Reuse reference caches while changing only the independently recorded model seed."""
    plan, reference = load_plan(root)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ValueError("replication needs exact source commit and archive identities")
    matches = [cell for cell in cells(plan) if cell["cell_id"] == cell_id]
    if len(matches) != 1:
        raise ValueError("replication cell is not in the frozen protocol")
    cell = matches[0]
    if cell_id not in plan["execution"]["initial_cells"]:
        raise ValueError("this cell requires its own window inputs or verified baseline reuse")
    if cell["window"] != plan["reference_window"]:
        raise ValueError("reference caches cannot be reused for an earlier temporal window")
    launch = copy.deepcopy(reference)
    launch["source_commit"] = source_commit
    launch["source_sha256"] = source_sha256
    launch["training"]["seed"] = cell["model_seed"]
    launch["evaluation_seed"] = plan["bootstrap_seed"]
    launch["bootstrap_replicates"] = plan["bootstrap_replicates"]
    launch["resources"]["maximum_runtime_seconds"] = plan["execution"]["maximum_runtime_seconds"]
    prefix, separator, _ = reference["checkpoint_uri"].partition("/runs/")
    if not separator or plan["reference_corpus_id"] not in prefix:
        raise ValueError("reference checkpoint namespace differs from the frozen corpus")
    launch["checkpoint_uri"] = f"{prefix}/robustness/{fingerprint(plan)}/{cell_id}/checkpoints"
    launch["robustness"] = {
        **cell,
        "protocol_id": fingerprint(plan),
        "reference_corpus_id": plan["reference_corpus_id"],
        "reference_source_commit": plan["reference_source_commit"],
        "comparison": "repeat the frozen selection procedure; report all planned cells",
    }
    return launch


def verify_seed_launch(root: Path, launch: dict[str, Any]) -> dict[str, Any]:
    """Reject altered data, settings and output locations before any cloud mutation."""
    expected = seed_launch(
        root,
        launch["robustness"]["cell_id"],
        source_commit=launch["source_commit"],
        source_sha256=launch["source_sha256"],
    )
    if launch != expected:
        raise ValueError("replication launch differs from its frozen inputs, settings or namespace")
    return dict(expected["robustness"])


def verification_launch(
    root: Path, training_launch: dict[str, Any], *, source_commit: str, source_sha256: str
) -> dict[str, Any]:
    """Use separate verifier code while retaining the exact completed training identity."""
    cell = verify_seed_launch(root, training_launch)
    launch = seed_launch(
        root, cell["cell_id"], source_commit=source_commit, source_sha256=source_sha256
    )
    launch.update(task="verification", training_launch=copy.deepcopy(training_launch))
    launch["resources"]["maximum_runtime_seconds"] = 1800
    return launch


def verify_verification_launch(root: Path, launch: dict[str, Any]) -> dict[str, Any]:
    expected = verification_launch(
        root,
        launch["training_launch"],
        source_commit=launch["source_commit"],
        source_sha256=launch["source_sha256"],
    )
    if launch != expected:
        raise ValueError("verification launch differs from its frozen training inputs")
    return dict(expected["robustness"])
