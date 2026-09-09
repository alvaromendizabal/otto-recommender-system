"""Prevent incomplete or altered replication evidence from becoming portfolio claims."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from otto_recsys.research.robustness import FROZEN_FILES, FROZEN_MODULES, fingerprint
from otto_recsys.research.robustness_report import check_audit, comparison

ROOT = Path(__file__).resolve().parents[1]


def test_actual_reference_audit_and_metric_counts_agree() -> None:
    report = json.loads((ROOT / "reports/research/evaluation.json").read_text())
    audit = json.loads((ROOT / "reports/research/audit.json").read_text())
    check_audit(audit, report)


@pytest.mark.parametrize("mutation", ["incomplete", "checksum", "lineage", "count", "recall"])
def test_modified_evidence_cannot_be_reported(mutation: str) -> None:
    report = json.loads((ROOT / "reports/research/evaluation.json").read_text())
    audit = json.loads((ROOT / "reports/research/audit.json").read_text())
    if mutation == "incomplete":
        report["status"] = "running"
    elif mutation == "checksum":
        audit["elapsed_seconds"] += 1
    elif mutation == "lineage":
        report["seal_id"] = "a" * 64
    elif mutation == "count":
        report["scores"]["selected"]["objectives"]["orders"]["hits"] += 1
    else:
        report["scores"]["selected"]["weighted_recall_at_20"] += 0.01
    with pytest.raises(ValueError):
        check_audit(audit, report)


def test_failed_replay_cannot_be_hidden_by_rehashing_receipt() -> None:
    report = json.loads((ROOT / "reports/research/evaluation.json").read_text())
    audit = json.loads((ROOT / "reports/research/audit.json").read_text())
    audit["native_replay"]["mismatches"] = 1
    audit["audit_id"] = fingerprint({k: v for k, v in audit.items() if k != "audit_id"})
    with pytest.raises(ValueError, match="replay"):
        check_audit(audit, report)


def test_published_comparison_rebuilds_from_current_verified_inputs() -> None:
    published = json.loads((ROOT / "reports/robustness/comparison.json").read_text())
    assert comparison(ROOT) == published
    progress = json.loads((ROOT / "reports/robustness/progress.json").read_text())
    verified = {c["cell_id"]: c for c in progress["cells"] if c["status"].startswith("verified")}
    assert set(verified) == {c["cell_id"] for c in published["rows"]}
    assert progress["completed_new_replications"] == len(verified) - 1
    for row in published["rows"]:
        assert (
            verified[row["cell_id"]]["weighted_recall_at_20"]
            == (row["scores"]["selected"]["weighted_recall_at_20"])
        )


def test_different_paired_gain_is_rejected() -> None:
    report = json.loads((ROOT / "reports/research/evaluation.json").read_text())
    audit = json.loads((ROOT / "reports/research/audit.json").read_text())
    changed = copy.deepcopy(report)
    changed["paired_intervals"]["core"]["absolute_gain"] += 0.005
    with pytest.raises(ValueError, match="feature gain"):
        check_audit(audit, changed)


@pytest.mark.parametrize("mutation", ["payload", "running", "protocol"])
def test_replication_bundle_cannot_publish_changed_or_unfinished_evidence(
    tmp_path: Path, mutation: str
) -> None:
    published = json.loads((ROOT / "reports/robustness/comparison.json").read_text())
    paths = (
        set(published["source_files"])
        | FROZEN_FILES
        | {f"src/otto_recsys/research/{name}" for name in FROZEN_MODULES}
        | {"reports/robustness/reference_launch.json"}
    )
    for name in paths:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    assert comparison(tmp_path) == published
    base = tmp_path / "reports/robustness/cells/reference_seed_20260909"
    if mutation == "payload":
        path = base / "evaluation/report.json"
        path.write_text(path.read_text() + "\n")
    elif mutation == "running":
        path = base / "robustness_audit/job_status.json"
        value = json.loads(path.read_text())
        value["status"] = "running"
        path.write_text(json.dumps(value))
    else:
        path = base / "robustness_audit/report.json"
        value = json.loads(path.read_text())
        value["protocol_id"] = "a" * 64
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        comparison(tmp_path)
