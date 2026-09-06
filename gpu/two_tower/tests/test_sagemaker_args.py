from __future__ import annotations

import pytest

from otto_two_tower.sagemaker_args import worker_arguments


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", "True"),
        ("false", "False"),
        ('"true"', "true"),
        ("128", "128"),
        ("0.98", "0.98"),
        ("1e2", "100.0"),
        ("null", ""),
        ("32,64", "32,64"),
    ],
)
def test_scalar_json_decoding_precedes_cli_conversion(value: str, expected: str) -> None:
    assert worker_arguments(
        {"sagemaker_program": '"benchmark.py"', "k": value, "sagemaker_custom": "true"},
        program="benchmark.py",
    ) == ["-k", expected]


@pytest.mark.parametrize("value", ["[]", "{}", '{"key": 1}'])
def test_nonscalar_hyperparameters_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="scalar"):
        worker_arguments(
            {"sagemaker_program": "benchmark.py", "value": value}, program="benchmark.py"
        )
