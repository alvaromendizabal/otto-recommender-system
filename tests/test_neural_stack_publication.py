"""Public contracts for the OTTO supervised-neural frontier snapshot."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / "research" / "neural_stack"


def test_all_public_point_scores_and_state_are_explicit() -> None:
    status = json.loads((ROOT / "status.json").read_text())
    assert status["competition"]["private_score"] == 0.571
    assert status["competition"]["public_score"] == 0.57121
    assert status["competition"]["remaining_private_gap"] == 0.03403
    assert status["neural_stack"]["state"] == "IN_PROGRESS_NO_SELECTION_OR_EVALUATION_RESULT"


def test_reproduction_matrix_does_not_claim_complete_winner_reproduction() -> None:
    matrix = json.loads((ROOT / "reproduction_matrix.json").read_text())
    statuses = {row["component"]: row["status"] for row in matrix["components"]}
    ensemble = "Eight winning neural candidate models v15/v18/v21/v23/v27/v29/v31/v42"
    objective = "Winner v42 task-conditioned sequence MLP and hard-negative contrastive objective"
    assert "Not recreated as an integrated ensemble" in statuses[ensemble]
    assert "pending" in statuses[objective].lower()


def test_notebook_is_executed_and_bound_to_receipt() -> None:
    receipt = json.loads((ROOT / "notebook_receipt.json").read_text())
    path = ROOT / "03_neural_stack_status.ipynb"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == receipt["sha256"]
    notebook = json.loads(path.read_text())
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert [cell["execution_count"] for cell in code] == list(range(1, len(code) + 1))
    outputs = [output for cell in code for output in cell["outputs"]]
    plots = [
        output["data"]
        for output in outputs
        if "application/vnd.plotly.v1+json" in output.get("data", {})
    ]
    assert len(plots) == 3
    assert all("image/svg+xml" in plot for plot in plots)
    assert "NEURAL_STACK_STATUS_NOTEBOOK_COMPLETE" in str(code[-1]["outputs"])


def test_candidate_augmentation_preserves_incumbent_order() -> None:
    path = ROOT / "candidate_integration.py"
    spec = importlib.util.spec_from_file_location("candidate_integration_public", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original = np.array([10, 20, 30], dtype=np.int64)
    neural = np.array([30, 40, 50, 40, 999], dtype=np.int64)
    result = module.combine_candidates(original, neural, padding_id=100)
    np.testing.assert_array_equal(result, np.array([10, 20, 30, 40, 50]))


@pytest.mark.parametrize("bad", [[10, 10], [-1, 10]])
def test_candidate_augmentation_rejects_invalid_incumbent(bad: list[int]) -> None:
    path = ROOT / "candidate_integration.py"
    spec = importlib.util.spec_from_file_location("candidate_integration_public_bad", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with pytest.raises(ValueError):
        module.combine_candidates(np.array(bad), np.array([40]), padding_id=100)
