"""Schedule the remaining frozen replications without racing shared window inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from otto_recsys.research.temporal_robustness import (
    verify_window_launch,
    verify_window_verification,
    window_launch,
    window_verification_launch,
)

REMAINING_CELLS = (
    "early_seed_20260910",
    "middle_seed_20260908",
    "middle_seed_20260909",
    "middle_seed_20260910",
)


def build_batch(repository: Path) -> dict[str, Any]:
    """Reuse the audited worker; change scheduling only, never the frozen protocol."""
    runs = repository / "reports/robustness/runs"
    previous = json.loads((runs / "early_seed_20260909.launch.json").read_text())
    verify_window_launch(repository, previous)
    template = json.loads((runs / "early_seed_20260909.request.json").read_text())
    source = previous["source_sha256"]
    commit = previous["source_commit"]
    protocol = previous["robustness"]["protocol_id"]
    pipeline_name = f"otto-robustness-{protocol[:12]}-remaining"
    steps: list[dict[str, Any]] = []
    uploads = []
    jobs = []
    for cell in REMAINING_CELLS:
        training = window_launch(
            repository, cell, source_commit=commit, source_sha256=source
        )
        audit = window_verification_launch(
            repository, training, source_commit=commit, source_sha256=source
        )
        verify_window_launch(repository, training)
        verify_window_verification(repository, audit)
        step_base = cell.replace("_", "-")
        for phase, launch in (("train", training), ("audit", audit)):
            request = copy.deepcopy(template)
            name = f"otto-batch-{phase}-{step_base}-{source[:8]}"
            request["ProcessingJobName"] = name
            request["StoppingCondition"]["MaxRuntimeInSeconds"] = (
                launch["resources"]["maximum_runtime_seconds"]
            )
            for tag in request["Tags"]:
                if tag["Key"] == "ResearchCell":
                    tag["Value"] = cell
            base = launch["checkpoint_uri"].removesuffix("/checkpoints")
            uri = f"{base}/batch/{source}/{phase}/launch.json"
            for item in request["ProcessingInputs"]:
                if item["InputName"] == "launch":
                    item["S3Input"]["S3Uri"] = uri
            payload = json.dumps(launch, sort_keys=True, indent=2) + "\n"
            uploads.append({
                "uri": uri, "content": payload,
                "sha256": hashlib.sha256(payload.encode()).hexdigest(),
                "cell_id": cell, "phase": phase,
            })
            dependencies = []
            if phase == "audit":
                dependencies = [f"Train-{step_base}"]
            elif cell in {"middle_seed_20260909", "middle_seed_20260910"}:
                dependencies = ["Audit-middle-seed-20260908"]
            step_name = f"{phase.title()}-{step_base}"
            # Pipeline-level tags are propagated; Processing step schema has no Tags.
            arguments = {k: v for k, v in request.items() if k != "Tags"}
            step: dict[str, Any] = {
                "Name": step_name, "Type": "Processing", "Arguments": arguments,
            }
            if dependencies:
                step["DependsOn"] = dependencies
            steps.append(step)
            jobs.append({
                "cell_id": cell, "phase": phase, "step": step_name,
                "job_name": name, "checkpoint_uri": launch["checkpoint_uri"],
                "input_checkpoint_uri": launch["input_checkpoint_uri"],
                "maximum_runtime_seconds": request["StoppingCondition"]["MaxRuntimeInSeconds"],
                "request": request,
            })
    return {
        "schema_version": 1, "pipeline_name": pipeline_name,
        "protocol_id": protocol, "source_commit": commit, "source_sha256": source,
        "region": previous["region"], "owner_account": previous["owner_account"],
        "role_arn": template["RoleArn"], "maximum_parallel_steps": 3,
        "scheduling_authorization": (
            "User requested concurrent jobs on 2026-09-09. This operational amendment "
            "supersedes the original single-job cap without modifying the frozen protocol."
        ),
        "existing_job": template["ProcessingJobName"],
        "early_preparation_evidence": (
            "reports/robustness/cells/early_seed_20260908/robustness_audit/report.json"
        ),
        "definition": {"Version": "2020-12-01", "Steps": steps},
        "jobs": jobs, "uploads": uploads,
    }
