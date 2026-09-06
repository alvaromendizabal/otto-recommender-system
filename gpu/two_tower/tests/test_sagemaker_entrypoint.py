from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import sagemaker_entrypoint as entrypoint


@pytest.mark.parametrize("value", ["1", "true", "YES", "on", True])
def test_parse_bool_true(value: str | bool) -> None:
    assert entrypoint._parse_bool(value) is True


@pytest.mark.parametrize("value", ["0", "false", "NO", "off", False])
def test_parse_bool_false(value: str | bool) -> None:
    assert entrypoint._parse_bool(value) is False


def test_parse_bool_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="invalid boolean value"):
        entrypoint._parse_bool("maybe")


def test_failure_artifacts_persist_stage_and_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "output"
    output_data_dir = output_dir / "data"
    monkeypatch.setenv("SM_OUTPUT_DIR", str(output_dir))
    monkeypatch.setenv("SM_OUTPUT_DATA_DIR", str(output_data_dir))
    monkeypatch.setattr(
        entrypoint,
        "_runtime_snapshot",
        lambda: {"instance_type": "ml.g6.xlarge", "num_gpus": "1"},
    )

    entrypoint._write_failure_artifacts(
        stage="gpu_package_quality_gate",
        message="stage command exited with return code 1",
        code_commit="abc123",
        return_code=1,
        command=["python", "run_quality_gate.py"],
        elapsed_seconds=0.25,
    )

    payload = json.loads(
        (output_data_dir / "failure.json").read_text(encoding="utf-8")
    )
    assert payload["status"] == "failed"
    assert payload["stage"] == "gpu_package_quality_gate"
    assert payload["return_code"] == 1
    assert payload["code_commit"] == "abc123"
    assert payload["runtime"]["instance_type"] == "ml.g6.xlarge"

    failure_text = (output_dir / "failure").read_text(encoding="utf-8")
    assert "stage=gpu_package_quality_gate" in failure_text
    assert "return_code=1" in failure_text
