"""Publish compact comparisons only from completed, linked verification evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from otto_recsys.research.robustness import (
    cells,
    fingerprint,
    load_plan,
    verify_seed_launch,
    verify_verification_launch,
)

PAYLOADS = {
    "job_status.json",
    "evaluation/report.json",
    "evaluation/contract.json",
    "evaluation_seal.json",
    "models/study_contract.json",
    "models/ablation_report.json",
}
WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_audit(audit: dict[str, Any], report: dict[str, Any]) -> None:
    require(audit["status"] == report["status"] == "passed", "incomplete result")
    require(
        audit["audit_id"] == fingerprint({k: v for k, v in audit.items() if k != "audit_id"}),
        "verification receipt checksum differs",
    )
    require(
        audit["evaluation_id"] == report["input_id"] and audit["seal_id"] == report["seal_id"],
        "mixed evaluation lineage",
    )
    statistics = audit["statistics"]
    require(statistics["sessions"] == report["sessions"], "evaluation cohort differs")
    require(audit["native_replay"]["mismatches"] == 0, "native prediction replay differs")
    require(audit["ablations"]["native_models"] == 24, "planned native models are missing")
    require(audit["ablations"]["variants"] == 8, "planned ablations are missing")
    for name in ("selected", "core", "fusion"):
        score = report["scores"][name]
        actual = 0.0
        for objective, weight in WEIGHTS.items():
            metric = score["objectives"][objective]
            checked = statistics["scores"][name]["objectives"][objective]
            require(
                all(metric[k] == checked[k] for k in ("hits", "denominator")),
                "verified metric counts differ",
            )
            recall = metric["hits"] / metric["denominator"]
            require(
                math.isclose(recall, metric["recall_at_20"], rel_tol=0, abs_tol=1e-12),
                "incorrect objective recall",
            )
            actual += weight * recall
        require(
            math.isclose(actual, score["weighted_recall_at_20"], rel_tol=0, abs_tol=1e-12),
            "incorrect weighted recall",
        )
    for name, interval in report["paired_intervals"].items():
        gain = (
            report["scores"]["selected"]["weighted_recall_at_20"]
            - report["scores"][name]["weighted_recall_at_20"]
        )
        require(
            math.isclose(gain, interval["absolute_gain"], rel_tol=0, abs_tol=1e-12),
            "incorrect feature gain",
        )


def comparison(root: Path) -> dict[str, Any]:
    """Read evidence without training, rescoring, or changing the frozen selection rule."""
    plan, _ = load_plan(root)
    protocol_id = fingerprint(plan)
    sources: dict[str, str] = {}

    def track(path: Path) -> dict[str, Any]:
        require(path.resolve().is_relative_to(root.resolve()), "invalid evidence path")
        sources[str(path.relative_to(root))] = digest(path)
        return read(path)

    reference = root / "reports/research"
    manifest = track(reference / "manifest.json")
    baseline: dict[str, Any] = {}
    for name in ("evaluation", "audit", "ablations", "evaluation_seal"):
        path = reference / f"{name}.json"
        require(digest(path) == manifest["files"][path.name], "reference checksum differs")
        baseline[name] = track(path)
    check_audit(baseline["audit"], baseline["evaluation"])
    rows = []
    for cell in cells(plan):
        original = (
            cell["window"] == plan["reference_window"] and cell["model_seed"] == plan["cohort_seed"]
        )
        directory = root / "reports/robustness/cells" / cell["cell_id"]
        if not original and not directory.exists():
            continue
        if original:
            report, audit = baseline["evaluation"], baseline["audit"]
            ablations, seal = baseline["ablations"], baseline["evaluation_seal"]
            evidence = "reports/research/audit.json"
        else:
            launch = track(root / "reports/robustness/runs" / f"{cell['cell_id']}.launch.json")
            verify_seed_launch(root, launch)
            verification = track(directory / "verification_launch.json")
            verify_verification_launch(root, verification)
            require(verification["training_launch"] == launch, "verification launch differs")
            audit = track(directory / "robustness_audit/report.json")
            status = track(directory / "robustness_audit/job_status.json")
            require(
                status["status"] == "passed" and status["stage"] == "complete",
                "managed verification is incomplete",
            )
            require(
                status["audit_id"] == audit["audit_id"]
                and status["source_commit"] == verification["source_commit"]
                and status["source_sha256"] == verification["source_sha256"],
                "managed verification identity differs",
            )
            inputs = audit["inputs"]
            require(audit["input_id"] == fingerprint(inputs), "audit input identity differs")
            require(
                audit["protocol_id"] == inputs["protocol_id"] == protocol_id
                and audit["cell_id"] == cell["cell_id"]
                and inputs["launch"] == fingerprint(launch),
                "replication belongs to different protocol or cell",
            )
            require(
                inputs["reference_audit_id"] == baseline["audit"]["audit_id"],
                "replication uses a different reference audit",
            )
            require(set(inputs["files"]) == PAYLOADS, "audit payload inventory differs")
            for name, expected in inputs["files"].items():
                require(digest(directory / name) == expected, "audited payload checksum differs")
                track(directory / name)
            report = read(directory / "evaluation/report.json")
            check_audit(audit, report)
            require(audit["paired_intervals"] == report["paired_intervals"], "intervals differ")
            ablations = read(directory / "models/ablation_report.json")
            seal = read(directory / "evaluation_seal.json")
            evidence = str((directory / "robustness_audit/report.json").relative_to(root))
        require(ablations["chosen"] == audit["ablations"]["chosen"], "selection differs")
        require(seal["seal_id"] == report["seal_id"], "selection seal differs")
        rows.append(
            {
                **cell,
                "status": "verified_reference" if original else "verified",
                "sessions": report["sessions"],
                "scores": report["scores"],
                "paired_intervals": report["paired_intervals"],
                "chosen": ablations["chosen"],
                "selected_features": {o: len(m["features"]) for o, m in seal["models"].items()},
                "audit_id": audit["audit_id"],
                "audit_evidence": evidence,
                "prediction_comparison": audit.get("prediction_comparison"),
            }
        )
    sources["configs/robustness.toml"] = digest(root / "configs/robustness.toml")
    return {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "planned_cells": len(cells(plan)),
        "verified_cells": len(rows),
        "verified_windows": len({row["window"] for row in rows}),
        "scope": (
            "All verified planned cells are retained. Per-cell intervals condition on fitted "
            "models and the temporal cohort. Repeated seeds are not independent datasets."
        ),
        "rows": rows,
        "source_files": sources,
    }
