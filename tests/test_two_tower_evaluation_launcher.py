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
    source = Path("gpu/two_tower").resolve()
    # Exercise the real packaged parser even though cloud calls are isolated.
    validate_launch = module.validate_evaluation_launch
    monkeypatch.setattr(
        module,
        "validate_evaluation_launch",
        lambda _, definition: validate_launch(source, definition),
    )
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


def test_invalid_worker_argument_blocks_every_cloud_write(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["launch", "--bucket", "bucket", "--start"])
    build_definition = launcher.evaluation_definition

    def invalid_definition(**kwargs: object) -> dict:
        definition = build_definition(**kwargs)
        definition["Steps"][0]["Arguments"]["HyperParameters"]["unsupported-option"] = "800"
        return definition

    monkeypatch.setattr(launcher, "evaluation_definition", invalid_definition)
    calls: list[list[str]] = []

    def aws_json(arguments: list[str]) -> dict:
        calls.append(arguments)
        assert arguments[1] == "describe-pipeline-execution"
        return {"PipelineExecutionStatus": "Succeeded"}

    def reject_write(*args: object, **kwargs: object) -> None:
        pytest.fail("cloud write occurred before validation")

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    monkeypatch.setattr(launcher, "run_command", reject_write)
    monkeypatch.setattr(launcher, "put_json_s3", reject_write)
    with pytest.raises(RuntimeError, match="unsupported-option"):
        launcher.main()
    assert len(calls) == 1


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


def test_watch_reports_worker_failure_and_billable_time(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def aws_json(arguments: list[str]) -> dict:
        if arguments[1] == "describe-pipeline-execution":
            return {"PipelineExecutionStatus": "Failed"}
        if arguments[1] == "list-pipeline-execution-steps":
            return {"PipelineExecutionSteps": [{"Metadata": {"TrainingJob": {"Arn": "jobs/eval"}}}]}
        assert arguments[1] == "describe-training-job"
        return {
            "TrainingJobStatus": "Failed",
            "SecondaryStatus": "Failed",
            "FailureReason": "unrecognized arguments: -k 800",
            "BillableTimeInSeconds": 160,
        }

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    monkeypatch.setattr(launcher, "training_logs", lambda *args, **kwargs: [])
    assert (
        launcher.watch(
            {"pipeline_execution_arn": "eval", "checkpoint_key": "key/"},
            bucket="bucket",
            download=False,
            max_wait=1,
        )
        == 1
    )
    output = capsys.readouterr().out
    assert "worker_status=Failed" in output
    assert '"billable_seconds": 160' in output
    assert "unrecognized arguments: -k 800" in output
    assert "if any were uploaded" in output
    assert "RESUME_AVAILABLE" not in output
