from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig


@pytest.fixture
def launcher(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fold_launcher", Path("scripts/launch_two_tower_fold.py").resolve()
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: argparse.Namespace(
            bucket="otto-test-bucket",
            config=Path("config.toml"),
            profile="fold0",
            region=None,
            instance_type=None,
            role_arn=None,
            start=False,
            force=False,
        ),
    )
    monkeypatch.setattr(
        module,
        "load_fold_config",
        lambda *a, **kw: FoldTrainingConfig(bucket="otto-test-bucket", region="us-west-2"),
    )
    monkeypatch.setattr(module, "ensure_committed_remote_state", lambda: "a" * 40)
    monkeypatch.setattr(module, "resolve_role_arn", lambda _: "arn:aws:iam::123:role/test")
    monkeypatch.setattr(module, "run_exact_source_preflight", lambda _: None)
    monkeypatch.setattr(module, "create_deterministic_source_archive", lambda *a: "b" * 64)
    monkeypatch.setattr(
        module,
        "verify_source_archive",
        lambda *a: {
            "files": 26,
            "manifest_sha256": "c" * 64,
        },
    )
    monkeypatch.setattr(module, "verify_uploaded_source_roundtrip", lambda **kw: {"files": 26})
    monkeypatch.setattr(
        module, "run_command", lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "{}", "")
    )
    monkeypatch.setattr(module, "s3_json", lambda *a: {"global_step": 40})
    monkeypatch.setattr(module, "head_s3", lambda *a: {"ContentLength": 100})
    monkeypatch.setattr(module, "put_json_s3", lambda *a, **kw: None)
    monkeypatch.setattr(module, "register_pipeline", lambda **kw: {"PipelineArn": "pipeline"})
    monkeypatch.setattr(module, "pipeline_executions", lambda _: [])
    return module


def set_options(module: ModuleType, monkeypatch: pytest.MonkeyPatch, **options: Any) -> None:
    args = module.parse_args()
    for key, value in options.items():
        setattr(args, key, value)
    monkeypatch.setattr(module, "parse_args", lambda: args)


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("status", ["Executing", "Stopping"])
def test_repeated_start_preserves_active_reference_even_with_force(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    force: bool,
    status: str,
) -> None:
    set_options(launcher, monkeypatch, start=True, force=force)
    existing = {
        "PipelineExecutionArn": "pipeline/execution/existing",
        "PipelineExecutionStatus": status,
    }
    monkeypatch.setattr(launcher, "pipeline_executions", lambda _: [existing])
    calls: list[object] = []
    monkeypatch.setattr(launcher, "aws_json", lambda args: calls.append(args))
    assert launcher.main() == 0
    assert calls == []
    latest = json.loads(Path("artifacts/two_tower_fold/latest.json").read_text())
    assert latest["pipeline_execution_arn"] == existing["PipelineExecutionArn"]
    assert latest["status"] == status


def test_registration_does_not_start_gpu(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(launcher, "aws_json", lambda args: calls.append(args))
    assert launcher.main() == 0
    assert calls == []
    latest = json.loads(Path("artifacts/two_tower_fold/latest.json").read_text())
    assert "pipeline_execution_arn" not in latest
    assert latest["region"] == "us-west-2"


def test_partial_training_manifest_does_not_prevent_resume(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_options(launcher, monkeypatch, start=True)
    previous = {
        "PipelineExecutionArn": "pipeline/execution/failed",
        "PipelineExecutionStatus": "Failed",
    }
    monkeypatch.setattr(launcher, "pipeline_executions", lambda _: [previous])
    calls: list[list[str]] = []

    def start(arguments: list[str]) -> dict[str, str]:
        calls.append(arguments)
        return {"PipelineExecutionArn": "pipeline/execution/resumed"}

    monkeypatch.setattr(launcher, "aws_json", start)
    assert launcher.main() == 0
    assert calls[0][:2] == ["sagemaker", "start-pipeline-execution"]
    assert "--client-request-token" in calls[0]
    latest = json.loads(Path("artifacts/two_tower_fold/latest.json").read_text())
    assert latest["pipeline_execution_arn"].endswith("/resumed")


def test_successful_execution_is_not_started_again(
    launcher: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_options(launcher, monkeypatch, start=True)
    monkeypatch.setattr(
        launcher,
        "pipeline_executions",
        lambda _: [
            {
                "PipelineExecutionArn": "pipeline/execution/succeeded",
                "PipelineExecutionStatus": "Succeeded",
            }
        ],
    )
    calls: list[object] = []
    monkeypatch.setattr(launcher, "aws_json", lambda args: calls.append(args))
    assert launcher.main() == 0
    assert calls == []


def test_start_token_is_stable_for_concurrent_attempts_and_changes_after_failure(
    launcher: ModuleType,
) -> None:
    first = launcher.execution_token("run", [])
    assert first == launcher.execution_token("run", [])
    assert first != launcher.execution_token("run", [{"PipelineExecutionArn": "failed"}])


def test_invalid_metadata_never_reaches_aws(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = importlib.util.spec_from_file_location(
        "fold_registration", "scripts/launch_two_tower_fold.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "pipeline_definition.json"
    path.write_text(json.dumps({"Metadata": {"ValidationFold": 0}}))
    calls: list[object] = []
    monkeypatch.setattr(module, "pipeline_exists", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match=r"Metadata\.ValidationFold"):
        module.register_pipeline(
            name="test",
            role_arn="role",
            definition_path=path,
            run_id="a" * 64,
            commit="b" * 40,
            fold=0,
        )
    assert calls == []
