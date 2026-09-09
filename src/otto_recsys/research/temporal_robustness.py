"""Isolate earlier windows while preserving the precommitted research procedure.

Only boundaries and model seeds vary. Shared window inputs are prepared from
original events, never from the reference window's later fitted artifacts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any

from otto_recsys.research.robustness import cells, fingerprint, load_plan

BOUNDARIES = ("history_end", "fit_end", "selection_end", "evaluation_end")
INPUT_DIRECTORIES = (
    "corpus", "retrieval", "screening_cache", "screening", "fit_cache", "selection_cache"
)
INPUT_REPORTS = (
    "preparation.json",
    "corpus/contract.json", "corpus/manifest.json",
    "retrieval/contract.json", "retrieval/manifest.json",
    "screening_cache/contract.json", "screening_cache/manifest.json",
    "screening/contract.json", "screening/selection.json",
    "fit_cache/contract.json", "fit_cache/manifest.json",
    "selection_cache/contract.json", "selection_cache/manifest.json",
)


def original_source(repository: Path) -> dict[str, Any]:
    """Bind the new source inventory to the original audited event bytes."""
    path = repository / "reports/research/temporal_contract.json"
    manifest = json.loads((repository / "reports/research/manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][path.name]:
        raise ValueError("original source contract checksum differs")
    return dict(json.loads(path.read_text())["source"])


def window_launch(
    repository: Path, cell_id: str, *, source_commit: str, source_sha256: str
) -> dict[str, Any]:
    plan, reference = load_plan(repository)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not re.fullmatch(
        r"[0-9a-f]{64}", source_sha256
    ):
        raise ValueError("window execution requires exact source identities")
    matches = [c for c in cells(plan) if c["cell_id"] == cell_id]
    if len(matches) != 1 or matches[0]["window"] == plan["reference_window"]:
        raise ValueError("fresh window execution requires a predeclared earlier cell")
    cell = matches[0]
    window = next(w for w in plan["windows"] if w["name"] == cell["window"])
    config = tomllib.loads((repository / "configs/research.toml").read_text())
    source = original_source(repository)
    inventory = json.loads((repository / "reports/robustness/source_inventory.json").read_text())
    if (
        len(inventory) != len(source["parts"])
        or {p["path"]: p["sha256"] for p in inventory} != source["parts"]
        or any(type(p["bytes"]) is not int or p["bytes"] <= 0 for p in inventory)
    ):
        raise ValueError("earlier window source differs from original event inventory")
    prefix, separator, _ = reference["checkpoint_uri"].partition("/runs/")
    if not separator or plan["reference_corpus_id"] not in prefix:
        raise ValueError("invalid frozen project namespace")
    base = f"{prefix}/robustness/{fingerprint(plan)}"
    resources = {
        **reference["resources"], "maximum_runtime_seconds": 7200, "memory_gib": 64
    }
    return {
        "task": "window_study",
        "source_commit": source_commit,
        "source_sha256": source_sha256,
        "region": reference["region"],
        "owner_account": reference["owner_account"],
        "train_uri": f"{prefix.split('/ranking/')[0]}/processed/train/",
        "train_files": inventory,
        "conversion_manifest": source["conversion_manifest"],
        "input_checkpoint_uri": f"{base}/windows/{cell['window']}/inputs",
        "checkpoint_uri": f"{base}/{cell_id}/checkpoints",
        "protocol": {
            **{k: window[k] for k in BOUNDARIES}, "seed": plan["cohort_seed"],
            "fit_sessions": plan["fit_sessions"], "selection_sessions": plan["selection_sessions"],
        },
        "retrieval": config["retrieval"],
        "features": config["features"],
        "training": {**plan["training"], "seed": cell["model_seed"]},
        "evaluation_seed": plan["bootstrap_seed"],
        "bootstrap_replicates": plan["bootstrap_replicates"],
        "resources": resources,
        "robustness": {
            **cell, "protocol_id": fingerprint(plan),
            "reference_corpus_id": plan["reference_corpus_id"],
            "reference_source_commit": plan["reference_source_commit"],
            "comparison": "repeat the frozen selection procedure; report all planned cells",
        },
    }


def verify_window_launch(repository: Path, launch: dict[str, Any]) -> dict[str, Any]:
    expected = window_launch(
        repository, launch["robustness"]["cell_id"],
        source_commit=launch["source_commit"], source_sha256=launch["source_sha256"],
    )
    if launch != expected:
        raise ValueError("window launch differs from frozen inputs, settings or namespaces")
    return dict(expected["robustness"])


def window_verification_launch(
    repository: Path, training_launch: dict[str, Any], *, source_commit: str, source_sha256: str
) -> dict[str, Any]:
    cell = verify_window_launch(repository, training_launch)
    launch = window_launch(
        repository, cell["cell_id"], source_commit=source_commit, source_sha256=source_sha256
    )
    launch.update(task="window_verification", training_launch=copy.deepcopy(training_launch))
    launch["resources"]["maximum_runtime_seconds"] = 1800
    return launch


def verify_window_verification(repository: Path, launch: dict[str, Any]) -> dict[str, Any]:
    expected = window_verification_launch(
        repository, launch["training_launch"], source_commit=launch["source_commit"],
        source_sha256=launch["source_sha256"],
    )
    if launch != expected:
        raise ValueError("window verification launch differs from frozen training inputs")
    return dict(expected["robustness"])


def protocol_values(launch: dict[str, Any]) -> dict[str, Any]:
    return {
        k: int(datetime.fromisoformat(v).timestamp() * 1000) if k in BOUNDARIES else v
        for k, v in launch["protocol"].items()
    }
