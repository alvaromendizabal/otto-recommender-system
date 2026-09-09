from __future__ import annotations

import hashlib
import json
from pathlib import Path

from otto_recsys.cloud.robustness_pipeline import REMAINING_CELLS, build_batch
from scripts.robustness_status import all_finished, rows

ROOT = Path(__file__).resolve().parents[1]


def test_batch_has_no_duplicate_training_or_shared_window_writers() -> None:
    batch = build_batch(ROOT)
    steps = {s["Name"]: s for s in batch["definition"]["Steps"]}
    assert len(steps) == 8
    assert batch["maximum_parallel_steps"] == 3
    jobs = [s["Arguments"]["ProcessingJobName"] for s in steps.values()]
    assert len(set(jobs)) == 8
    assert batch["existing_job"] not in jobs
    for cell in REMAINING_CELLS:
        suffix = cell.replace("_", "-")
        assert steps[f"Audit-{suffix}"]["DependsOn"] == [f"Train-{suffix}"]
    for seed in (20260909, 20260910):
        assert steps[f"Train-middle-seed-{seed}"]["DependsOn"] == [
            "Audit-middle-seed-20260908"
        ]
    assert "DependsOn" not in steps["Train-early-seed-20260910"]
    assert "DependsOn" not in steps["Train-middle-seed-20260908"]


def test_batch_preserves_the_frozen_protocol_and_bounds_every_job() -> None:
    protocol = ROOT / "configs/robustness.toml"
    before = protocol.read_bytes()
    batch = build_batch(ROOT)
    assert protocol.read_bytes() == before
    assert len({u["uri"] for u in batch["uploads"]}) == 8
    for item in batch["uploads"]:
        assert hashlib.sha256(item["content"].encode()).hexdigest() == item["sha256"]
        launch = json.loads(item["content"])
        assert launch["robustness"]["protocol_id"] == batch["protocol_id"]
        assert launch["training"]["seed"] == launch["robustness"]["model_seed"]
        assert launch["resources"]["maximum_runtime_seconds"] == (
            7200 if item["phase"] == "train" else 1800
        )
        assert launch["source_sha256"] == batch["source_sha256"]
    assert all("RetryPolicies" not in s for s in batch["definition"]["Steps"])


def test_monitor_uses_observed_job_names_and_keeps_waiting_steps() -> None:
    batch = build_batch(ROOT)
    observed = {"steps": {"PipelineExecutionSteps": [{
        "StepName": "Train-middle-seed-20260908", "StepStatus": "Executing",
        "Metadata": {"ProcessingJob": {
            "Arn": "arn:aws:sagemaker:region:account:processing-job/actual-job-suffix"
        }},
    }]}}
    status = rows(batch, observed)
    assert len(status) == 8
    running = [r for r in status if r["status"] == "Executing"]
    assert len(running) == 1
    assert running[0]["job"] == "actual-job-suffix"
    assert sum(r["status"] == "Waiting" for r in status) == 7


def test_monitor_keeps_watching_inference_after_validation_finishes() -> None:
    observed = {
        "execution": {"PipelineExecutionStatus": "Succeeded"},
        "existing_job": {"ProcessingJobStatus": "Completed"},
        "additional_jobs": [{"ProcessingJobStatus": "InProgress"}],
    }
    assert not all_finished(observed)
    observed["additional_jobs"][0]["ProcessingJobStatus"] = "Completed"
    assert all_finished(observed)
