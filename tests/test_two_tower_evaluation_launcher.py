from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    scripts = Path("scripts").resolve()
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "evaluation_launcher", scripts / "launch_two_tower_evaluation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "ensure_committed_remote_state", lambda: "a" * 40)
    monkeypatch.setattr(module, "run_exact_source_preflight", lambda _: None)
    monkeypatch.setattr(module, "create_deterministic_source_archive", lambda *a: "b" * 64)
    monkeypatch.setattr(module, "verify_source_archive", lambda *a: {})
    monkeypatch.setattr(module, "verify_uploaded_source_roundtrip", lambda **kw: {})
    monkeypatch.setattr(module, "put_json_s3", lambda *a, **kw: None)
    monkeypatch.setattr(
        module, "run_command", lambda *a, **kw: subprocess.CompletedProcess(a, 0, "", "")
    )
    monkeypatch.setattr(module, "pipeline_executions", lambda _: [])
    from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig, build_fold_pipeline_definition

    definition = build_fold_pipeline_definition(
        role_arn="role",
        image_uri="image",
        source_uri="source",
        commit="old",
        run_id="run",
        config=FoldTrainingConfig(bucket="bucket"),
    )

    def s3_json(bucket: str, key: str) -> dict:
        if key.endswith("latest.json"):
            return {"run_id": "run", "pipeline_execution_arn": "trained/execution"}
        if key.endswith("training_manifest.json"):
            return {"code_commit": "old", "validation_fold": 0, "input_id": "trained"}
        if key.endswith("run_manifest.json"):
            return {"code_commit": "old", "input_manifests": {"ranking": "rank", "items": "items"}}
        if key.endswith("pipeline_definition.json"):
            return definition
        raise AssertionError(key)

    monkeypatch.setattr(module, "s3_json", s3_json)
    return module


@pytest.mark.parametrize("existing", [None, "Executing", "Stopping", "Succeeded", "Failed"])
def test_explicit_start_is_idempotent_and_failed_attempt_can_resume(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    existing: str | None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["launch", "--bucket", "bucket", "--start"])
    if existing:
        monkeypatch.setattr(
            launcher,
            "pipeline_executions",
            lambda _: [
                {"PipelineExecutionStatus": existing, "PipelineExecutionArn": "eval/existing"}
            ],
        )
    calls = []

    def aws_json(arguments: list[str]) -> dict:
        calls.append(arguments)
        if arguments[1] == "describe-pipeline-execution":
            return {"PipelineExecutionStatus": "Succeeded"}
        return {"PipelineExecutionArn": "eval/new"}

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    assert launcher.main() == 0
    starts = [call for call in calls if call[1] == "start-pipeline-execution"]
    assert len(starts) == int(existing in {None, "Failed"})


def test_registration_does_not_start_compute(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["launch", "--bucket", "bucket"])
    calls = []

    def aws_json(arguments: list[str]) -> dict:
        calls.append(arguments)
        return {"PipelineExecutionStatus": "Succeeded"}

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    assert launcher.main() == 0
    assert not any(call[1] == "start-pipeline-execution" for call in calls)


@pytest.mark.parametrize("status,expected", [("Succeeded", 0), ("Failed", 1), ("Stopped", 1)])
def test_watch_terminal_states(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, status: str, expected: int
) -> None:
    monkeypatch.setattr(
        launcher,
        "aws_json",
        lambda args: (
            {"PipelineExecutionStatus": status}
            if args[1] == "describe-pipeline-execution"
            else {"PipelineExecutionSteps": []}
        ),
    )
    monkeypatch.setattr(launcher, "s3_json", lambda *a: {"status": "passed"})
    assert (
        launcher.watch(
            {"pipeline_execution_arn": "eval", "checkpoint_key": "key/"},
            bucket="bucket",
            download=False,
            max_wait=1,
        )
        == expected
    )
