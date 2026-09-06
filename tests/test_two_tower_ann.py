from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from otto_recsys.cloud.sagemaker_pipeline import (
    create_deterministic_source_archive,
    verify_source_archive,
)
from otto_recsys.cloud.source_preflight import validate_ann_launch
from otto_recsys.cloud.two_tower_ann import ann_definition, load_ann_parameters, stage_ann_source
from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig, build_fold_pipeline_definition


def training_definition() -> dict:
    return build_fold_pipeline_definition(
        role_arn="role",
        image_uri="proven-image",
        source_uri="source",
        commit="previous",
        run_id="training",
        config=FoldTrainingConfig(bucket="bucket"),
    )


def test_managed_contract_preserves_proven_compute_and_validates_worker_arguments() -> None:
    original = training_definition()
    params = load_ann_parameters(Path("configs/two_tower_ann.toml"))
    definition = ann_definition(
        training_definition=original,
        bucket="bucket",
        training_run_id="trained",
        run_id="run",
        reference_run_id="export",
        reference_input_id="reference",
        reference_manifest_sha256="hash",
        source_uri="source",
        commit="commit",
        training_manifest={"input_id": "trained", "validation_fold": 0},
        input_manifests={"ranking": "ranking", "items": "items"},
        parameters=params,
    )
    args = definition["Steps"][0]["Arguments"]
    assert (
        args["AlgorithmSpecification"]
        == original["Steps"][0]["Arguments"]["AlgorithmSpecification"]
    )
    assert args["ResourceConfig"] == original["Steps"][0]["Arguments"]["ResourceConfig"]
    assert "CheckpointConfig" not in args
    assert args["HyperParameters"]["checkpoint-uri"].endswith("ann/fold-0/run/checkpoints/")
    assert {x["ChannelName"] for x in args["InputDataConfig"]} == {
        "ranking",
        "items",
        "trained",
        "reference",
    }
    assert "RetryPolicies" not in definition["Steps"][0]
    validate_ann_launch(Path("gpu/two_tower"), definition)
    args["HyperParameters"]["unexpected"] = "value"
    with pytest.raises(RuntimeError, match="unrecognized"):
        validate_ann_launch(Path("gpu/two_tower"), definition)


def test_ann_dependency_profile_is_separate_and_archive_is_exact(tmp_path: Path) -> None:
    source = Path("gpu/two_tower")
    before = (source / "requirements.txt").read_bytes()
    staged = tmp_path / "source"
    stage_ann_source(source, staged)
    assert (staged / "requirements.txt").read_bytes() == (
        source / "requirements-ann.txt"
    ).read_bytes()
    assert (source / "requirements.txt").read_bytes() == before
    archive = tmp_path / "source.tar.gz"
    create_deterministic_source_archive(staged, archive)
    verify_source_archive(staged, archive)
    assert b"faiss-cpu==1.15.0" in (staged / "requirements.txt").read_bytes()


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    scripts = Path("scripts").resolve()
    source = Path("gpu/two_tower").resolve()
    parameters = load_ann_parameters(Path("configs/two_tower_ann.toml"))
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "ann_launcher", scripts / "launch_two_tower_ann.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "load_ann_parameters", lambda _: parameters)
    monkeypatch.setattr(
        module, "stage_ann_source", lambda _, destination: stage_ann_source(source, destination)
    )
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

    def s3_json(bucket: str, key: str) -> dict:
        if key.endswith("latest.json"):
            is_export = "/evaluations/" in key
            return {
                "run_id": "export" if is_export else "training",
                "checkpoint_key": "reference/",
                "pipeline_execution_arn": "execution/export" if is_export else "execution/training",
            }
        if key.endswith("training_manifest.json"):
            return {"input_id": "trained", "validation_fold": 0}
        if key.endswith("pipeline_definition.json"):
            return training_definition()
        if key.endswith("run_manifest.json"):
            return {"input_manifests": {"ranking": "ranking", "items": "items"}}
        if key.endswith("prediction_manifest.json"):
            return {
                "input_id": "reference",
                "training_input_id": "trained",
                "status": "passed",
                "sessions": 103468,
            }
        raise AssertionError(key)

    monkeypatch.setattr(module, "s3_json", s3_json)
    return module


@pytest.mark.parametrize("existing", [None, "Executing", "Stopping", "Succeeded", "Failed"])
def test_explicit_start_retains_active_and_successful_runs(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    existing: str | None,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ann", "--bucket", "bucket", "--start"])
    if existing:
        monkeypatch.setattr(
            launcher,
            "pipeline_executions",
            lambda _: [
                {"PipelineExecutionStatus": existing, "PipelineExecutionArn": "existing/ann"}
            ],
        )
    calls = []

    def aws_json(arguments: list[str]) -> dict:
        calls.append(arguments)
        return {"PipelineExecutionStatus": "Succeeded", "PipelineExecutionArn": "new/ann"}

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    assert launcher.main() == 0
    assert sum(c[1] == "start-pipeline-execution" for c in calls) == int(
        existing in {None, "Failed"}
    )


def test_registration_never_starts_paid_compute(
    launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "argv", ["ann", "--bucket", "bucket"])
    calls = []
    monkeypatch.setattr(
        launcher, "aws_json", lambda a: calls.append(a) or {"PipelineExecutionStatus": "Succeeded"}
    )
    assert launcher.main() == 0
    assert not any(c[1] == "start-pipeline-execution" for c in calls)


def test_unknown_worker_argument_blocks_all_cloud_writes(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ann", "--bucket", "bucket", "--start"])
    original = launcher.ann_definition

    def malformed(**kwargs: object) -> dict:
        definition = original(**kwargs)
        definition["Steps"][0]["Arguments"]["HyperParameters"]["bad-option"] = "true"
        return definition

    monkeypatch.setattr(launcher, "ann_definition", malformed)
    monkeypatch.setattr(launcher, "aws_json", lambda _: {"PipelineExecutionStatus": "Succeeded"})

    def reject(*args: object, **kwargs: object) -> None:
        pytest.fail("cloud write occurred before launch validation")

    monkeypatch.setattr(launcher, "run_command", reject)
    monkeypatch.setattr(launcher, "put_json_s3", reject)
    with pytest.raises(RuntimeError, match="bad-option"):
        launcher.main()


@pytest.mark.parametrize("state,expected", [("Succeeded", 0), ("Failed", 1), ("Stopped", 1)])
def test_monitor_reports_terminal_status_and_billable_time(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    state: str,
    expected: int,
) -> None:
    def aws_json(arguments: list[str]) -> dict:
        if arguments[1] == "describe-pipeline-execution":
            return {"PipelineExecutionStatus": state}
        if arguments[1] == "list-pipeline-execution-steps":
            return {"PipelineExecutionSteps": [{"Metadata": {"TrainingJob": {"Arn": "job/name"}}}]}
        return {
            "SecondaryStatus": "Completed",
            "BillableTimeInSeconds": 10,
            "FailureReason": "sample reason" if state == "Failed" else None,
        }

    monkeypatch.setattr(launcher, "aws_json", aws_json)
    monkeypatch.setattr(launcher, "training_logs", lambda *a, **kw: ["heartbeat"])
    monkeypatch.setattr(
        launcher,
        "s3_json",
        lambda *a: {
            "status": "passed",
            "input_id": "run",
            "full_reference_ranking": {"weighted_recall_at_20": 0.2},
            "selected_nprobe": 64,
            "confirmation_fidelity_passed": True,
        },
    )
    assert (
        launcher.watch(
            {
                "run_id": "run",
                "pipeline_execution_arn": "pipeline/execution/test",
                "checkpoint_key": "ann/run/checkpoints/",
            },
            bucket="bucket",
            download=False,
            max_wait=1,
        )
        == expected
    )
    assert '"billable_seconds": 10' in capsys.readouterr().out
