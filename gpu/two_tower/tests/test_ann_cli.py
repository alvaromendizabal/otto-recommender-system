from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from otto_two_tower.ann_cli import hyperparameters_to_argv, parse_args

BASE = [
    "--run-id",
    "run",
    "--code-commit",
    "commit",
    "--reference-input-id",
    "reference",
    "--reference-manifest-sha256",
    "hash",
]


def test_ann_argument_contract_is_dependency_free_and_strict() -> None:
    args = parse_args(BASE)
    assert args.probes == (32, 64, 128, 256)
    assert args.sample_sessions == 4096
    assert args.candidate_depth == 800
    with pytest.raises(SystemExit):
        parse_args([*BASE, "--unknown", "value"])


@pytest.mark.parametrize(
    "arguments",
    [
        ["--sample-sessions", "3"],
        ["--sample-sessions", "5"],
        ["--probes", "64,32"],
        ["--probes", "32,32"],
        ["--probes", "2048"],
        ["--candidate-depth", "19"],
        ["--target-overlap", "nan"],
        ["--target-overlap", "1.1"],
        ["--heartbeat-seconds", "inf"],
        ["--train-items", "1024"],
        ["--threads", "0"],
    ],
)
def test_invalid_benchmark_settings_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args([*BASE, *arguments])


def test_wrong_entrypoint_is_rejected() -> None:
    with pytest.raises(ValueError):
        hyperparameters_to_argv({"sagemaker_program": "train.py"})


@pytest.mark.parametrize("value", ["true", "True", "TRUE", "false", "False", "FALSE"])
def test_boolean_spellings_have_one_canonical_value(value: str) -> None:
    assert parse_args([*BASE, "--export-fold-predictions", value]).export_fold_predictions == (
        value.lower()
    )


@pytest.mark.parametrize("value", ["yes", "1", "0", "null", "[]", "tru"])
def test_nonboolean_values_fail_closed(value: str) -> None:
    with pytest.raises(SystemExit):
        parse_args([*BASE, "--export-fold-predictions", value])


def test_observed_managed_argv_passes_without_site_packages() -> None:
    source = Path(__file__).resolve().parents[1]
    observed = json.loads((source / "tests/fixtures/ann_launch.json").read_text())
    assert hyperparameters_to_argv(observed["hyperparameters"]) == observed["observed_argv"]
    assert parse_args(observed["observed_argv"]).export_fold_predictions == "true"
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "otto_two_tower.ann_cli"],
        cwd=source,
        input=json.dumps(observed["hyperparameters"]),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["argv"] == observed["observed_argv"]


def test_startup_argument_failure_records_timestamp_exit_and_total() -> None:
    completed = subprocess.run(
        [sys.executable, "-S", "benchmark.py", *BASE, "--export-fold-predictions", "yes"],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 2
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert records[0]["message"] == "ann_worker_start"
    assert records[-1]["message"] == "ann_worker_complete"
    assert records[-1]["exit_code"] == 2
    assert records[-1]["elapsed_seconds"] > 0
    assert all(row["timestamp"].endswith("+00:00") for row in records)
