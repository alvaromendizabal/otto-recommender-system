from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/"research"/"neural_stack"

def test_status_keeps_competition_and_temporal_metrics_separate():
    s=json.loads((ROOT/"status.json").read_text())
    assert s["competition"]["private_score"] == 0.571
    assert s["competition"]["public_score"] == 0.57121
    assert s["competition"]["remaining_private_gap"] == 0.03403
    assert s["neural_stack"]["state"] == "IN_PROGRESS_NO_SELECTION_OR_EVALUATION_RESULT"

def test_reproduction_matrix_does_not_claim_complete_winner_reproduction():
    m=json.loads((ROOT/"reproduction_matrix.json").read_text())
    statuses={row["component"]:row["status"] for row in m["components"]}
    assert "Not recreated as an integrated ensemble" in statuses["Eight winning neural candidate models v15/v18/v21/v23/v27/v29/v31/v42"]
    assert "pending" in statuses["Winner v42 task-conditioned sequence MLP and hard-negative contrastive objective"].lower()

def test_notebook_is_executed_and_bound_to_receipt():
    r=json.loads((ROOT/"notebook_receipt.json").read_text())
    p=ROOT/"03_neural_stack_status.ipynb"
    assert hashlib.sha256(p.read_bytes()).hexdigest()==r["sha256"]
    nb=json.loads(p.read_text())
    code=[c for c in nb["cells"] if c["cell_type"]=="code"]
    assert [c["execution_count"] for c in code]==list(range(1,len(code)+1))
    outputs=[o for c in code for o in c.get("outputs",[])]
    assert sum("application/vnd.plotly.v1+json" in o.get("data",{}) for o in outputs)==3
    assert sum("image/svg+xml" in o.get("data",{}) for o in outputs)==3
    assert "NEURAL_STACK_STATUS_NOTEBOOK_COMPLETE" in str(code[-1]["outputs"])

def test_candidate_augmentation_preserves_incumbent_order():
    import importlib.util
    import numpy as np
    path=ROOT/"candidate_integration.py"
    spec=importlib.util.spec_from_file_location("candidate_integration_public",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    original=np.array([10,20,30],dtype=np.int64)
    neural=np.array([30,40,50,40,999],dtype=np.int64)
    result=module.combine_candidates(original,neural,padding_id=100)
    np.testing.assert_array_equal(result,np.array([10,20,30,40,50]))
