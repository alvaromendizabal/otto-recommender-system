from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def status_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "fold_status", Path("scripts/two_tower_fold_status.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def options() -> argparse.Namespace:
    return argparse.Namespace(
        bucket="test",
        fold=0,
        region=None,
        show_logs=False,
        publish_report=False,
        watch=True,
        interval_seconds=30.0,
        max_wait_seconds=100.0,
    )


@pytest.mark.parametrize("state", ["Failed", "Stopped", "Succeeded"])
def test_terminal_failure_or_missing_report_stops_watch(
    status_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    def s3_json(bucket: str, key: str) -> dict[str, str] | None:
        if key.endswith("latest.json"):
            return {"pipeline_execution_arn": "execution/123", "run_id": "test"}
        return None

    def aws_json(args: list[str]) -> dict[str, Any]:
        if args[1] == "describe-pipeline-execution":
            return {"PipelineExecutionStatus": state}
        return {"PipelineExecutionSteps": []}

    monkeypatch.setattr(status_module, "s3_json", s3_json)
    monkeypatch.setattr(status_module, "aws_json", aws_json)
    assert status_module.report_status(options()) == (1, True)


def test_watch_stops_after_terminal_state(
    status_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(status_module, "parse_args", options)
    results = iter([(0, False), (0, True)])
    monkeypatch.setattr(status_module, "report_status", lambda args: next(results))
    sleeps: list[float] = []
    monkeypatch.setattr(status_module.time, "sleep", sleeps.append)
    assert status_module.main() == 0
    assert len(sleeps) == 1


def test_watch_timeout_does_not_call_remote_stop(
    status_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = options()
    args.max_wait_seconds = 1
    monkeypatch.setattr(status_module, "parse_args", lambda: args)
    monkeypatch.setattr(status_module, "report_status", lambda args: (0, False))
    ticks = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(status_module.time, "perf_counter", lambda: next(ticks))
    calls: list[object] = []
    monkeypatch.setattr(status_module, "aws_json", lambda args: calls.append(args))
    assert status_module.main() == 2
    assert calls == []
