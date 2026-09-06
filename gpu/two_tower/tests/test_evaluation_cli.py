from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from otto_two_tower.evaluation_cli import hyperparameters_to_argv, parse_args

# Literal SM_USER_ARGS from the failed managed invocation. This regression
# fixture is independent of our serializer and contains no account identifiers.
OBSERVED_ARGUMENTS = [
    "--batch-size",
    "128",
    "--code-commit",
    "82779e442e34b12dea0b2ad753e21c6624a5fedd",
    "--expected-items-id",
    "cd28c437d4ad305fca9648b6fcb6a6ef8052795b584ca3a808685ba8daec676d",
    "--expected-ranking-id",
    "56a64d97c0586cc53c41d1e40c4fef9bb01b2fca0e7bb10c11db39f285462337",
    "--heartbeat-seconds",
    "30",
    "-k",
    "800",
    "--training-input-id",
    "0a0fdd1a83a839c9851285fc377467815e7374f986a2e74f9282502374d2ceb0",
]


def test_observed_sagemaker_invocation_is_accepted() -> None:
    args = parse_args(OBSERVED_ARGUMENTS)
    assert args.k == 800
    assert args.batch_size == 128
    assert args.heartbeat_seconds == 30
    assert args.model_dir == Path("/opt/ml/input/data/trained")
    assert not args.allow_cpu


@pytest.mark.parametrize("option", ["--candidate-depth", "--k", "-k"])
def test_candidate_depth_spellings(option: str) -> None:
    assert parse_args([*OBSERVED_ARGUMENTS, option, "20"]).k == 20


@pytest.mark.parametrize(
    "arguments",
    [
        ["--candidate-depth", "0"],
        ["--batch-size", "-1"],
        ["--chunk-size", "0"],
        ["--heartbeat-seconds", "0"],
        ["--heartbeat-seconds", "nan"],
        ["--heartbeat-seconds", "inf"],
        ["--candidate-dep", "800"],
        ["--unknown-setting", "value"],
    ],
)
def test_invalid_arguments_fail_closed(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        parse_args([*OBSERVED_ARGUMENTS, *arguments])
    assert error.value.code == 2


def test_launch_validation_needs_only_the_standard_library() -> None:
    parameters = dict(zip(OBSERVED_ARGUMENTS[::2], OBSERVED_ARGUMENTS[1::2], strict=True))
    hyperparameters = {key.lstrip("-"): value for key, value in parameters.items()}
    hyperparameters["sagemaker_program"] = "evaluate.py"
    hyperparameters["sagemaker_submit_directory"] = "s3://bucket/source.tar.gz"
    source = Path(__file__).resolve().parents[1]
    # -S disables site-packages, proving PyTorch and pytest are not required here.
    completed = subprocess.run(
        [sys.executable, "-S", "-m", "otto_two_tower.evaluation_cli"],
        input=json.dumps(hyperparameters),
        text=True,
        capture_output=True,
        cwd=source,
        check=False,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["candidate_depth"] == 800
    assert hyperparameters_to_argv(hyperparameters) == OBSERVED_ARGUMENTS
