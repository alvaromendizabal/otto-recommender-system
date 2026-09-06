from __future__ import annotations

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
