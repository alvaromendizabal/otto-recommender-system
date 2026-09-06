from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_resume_proof_public_artifact_is_sanitized_and_passed() -> None:
    path = Path("reports/metrics/two_tower_resume_proof.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = path.read_text(encoding="utf-8")
    assert payload["status"] == "passed"
    assert payload["resumed_from_step"] == 40
    assert payload["final_step"] == 80
    assert payload["advanced_steps"] == 40
    assert "arn:aws" not in text
    assert "otto-recsys-560403859723" not in text


def test_two_tower_notebook_is_committed_as_executed_evidence() -> None:
    payload = json.loads(Path("notebooks/05_two_tower_results.ipynb").read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert any(cell.get("execution_count") is not None for cell in code_cells)
    notebook_text = json.dumps(payload)
    assert "two_tower_resume_proof.json" in notebook_text
    assert "Fold 0 experimental contract" in notebook_text


def test_completed_fold_training_evidence_is_consistent_and_public() -> None:
    path = Path("reports/metrics/two_tower_fold0_training.json")
    payload = json.loads(path.read_text())
    assert payload["status"] == "passed"
    assert payload["global_step"] == 9600
    assert payload["completed_epochs"] == len(payload["history"]) == 4
    assert payload["best_valid_loss"] == min(row["valid"]["loss"] for row in payload["history"])
    assert payload["billable_seconds"] == 621
    assert "arn:aws" not in path.read_text()
    assert "560403859723" not in path.read_text()
    notebook = json.loads(Path("notebooks/05_two_tower_results.ipynb").read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is not None
            assert not any(output["output_type"] == "error" for output in cell["outputs"])
    assert Path("reports/figures/two_tower_learning_curves.png").stat().st_size > 1000


def test_export_evidence_covers_every_objective_without_claiming_quality() -> None:
    path = Path("reports/metrics/two_tower_fold0_export.json")
    payload = json.loads(path.read_text())
    assert payload["status"] == "passed"
    assert payload["sessions"] == 103468
    assert payload["catalogue_items"] == 1852162
    assert payload["billable_seconds"] == 627
    assert payload["prediction_parts"] == sum(
        row["parts"] for row in payload["objectives"].values()
    )
    assert payload["completed_retrieval_seconds"] == pytest.approx(
        sum(sum(row["bucket_seconds"]) for row in payload["objectives"].values())
    )
    assert all(row["sessions"] == payload["sessions"] for row in payload["objectives"].values())
    assert "pending" in payload["quality_evaluation"]
    assert "560403859723" not in path.read_text()
    assert Path("reports/figures/two_tower_export.png").stat().st_size > 1000
