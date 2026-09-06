from __future__ import annotations

import json
from pathlib import Path


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
    payload = json.loads(
        Path("notebooks/05_two_tower_results.ipynb").read_text(encoding="utf-8")
    )
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert any(cell.get("execution_count") is not None for cell in code_cells)
    notebook_text = json.dumps(payload)
    assert "two_tower_resume_proof.json" in notebook_text
    assert "Fold 0 experimental contract" in notebook_text
