"""A completed release or pilot must not silently close unresolved feature research."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from otto_recsys.research.feature_gate import feature_gate


def inventory_fixture(root: Path) -> Path:
    source = Path("configs/feature_research.json")
    inventory = json.loads(source.read_text())
    destination = root / source
    destination.parent.mkdir(parents=True)
    shutil.copyfile(source, destination)
    for entry in inventory["evidence"].values():
        target = root / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["path"], target)
    return destination


def test_real_inventory_keeps_unresolved_scientific_work_open() -> None:
    value = feature_gate(Path.cwd())
    assert value["feature_research_gate"] == "open"
    assert value["final_training_ready"] is False
    assert "task_neural" in value["unresolved_feature_families"]
    assert "asof_drift" in value["unresolved_feature_families"]
    assert value["feature_temporal_confirmation"] == "pending"


def test_absent_inventory_cannot_pass_gate(tmp_path: Path) -> None:
    assert feature_gate(tmp_path)["final_training_ready"] is False


@pytest.mark.parametrize("change", ["missing", "corrupt", "escape"])
def test_gate_rejects_unverifiable_evidence(tmp_path: Path, change: str) -> None:
    path = inventory_fixture(tmp_path)
    value = json.loads(path.read_text())
    entry = value["evidence"]["ablation"]
    if change == "missing":
        (tmp_path / entry["path"]).unlink()
    elif change == "corrupt":
        (tmp_path / entry["path"]).write_text('{"status":"passed"}')
    else:
        entry["path"] = "../outside.json"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="evidence"):
        feature_gate(tmp_path)


@pytest.mark.parametrize("change", ["no_result", "no_question", "no_constraint", "duplicate"])
def test_gate_requires_reviewable_dispositions(tmp_path: Path, change: str) -> None:
    path = inventory_fixture(tmp_path)
    value = json.loads(path.read_text())
    if change == "no_result":
        value["families"][0]["evidence"] = ["catalog"]
    elif change == "no_question":
        next(f for f in value["families"] if f["disposition"] == "open")[
            "remaining_question"
        ] = ""
    elif change == "no_constraint":
        next(f for f in value["families"] if f["disposition"] == "excluded")[
            "data_constraint"
        ] = ""
    else:
        value["families"].append(value["families"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        feature_gate(tmp_path)


def test_closed_family_inventory_still_requires_temporal_confirmation(tmp_path: Path) -> None:
    path = inventory_fixture(tmp_path)
    value = json.loads(path.read_text())
    value["families"] = [f for f in value["families"] if f["disposition"] != "open"]
    path.write_text(json.dumps(value))
    assert feature_gate(tmp_path)["final_training_ready"] is False
    value["temporal_confirmation"] = {"status": "passed", "evidence": []}
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="confirmation"):
        feature_gate(tmp_path)


def test_accepted_submission_does_not_override_open_feature_gate() -> None:
    from test_project_status import run_status

    result = run_status(Path.cwd())
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["kaggle_submission"] == "complete (after deadline)"
    assert status["feature_research_gate"] == "open"
    assert status["final_training_ready"] is False
    assert "feature engineering open" in status["next_task"]
