"""Dependency-free CLI contract shared by the worker and launch preflight."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .sagemaker_args import worker_arguments


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export held-out full-catalogue predictions from a saved best model.",
        allow_abbrev=False,
    )
    parser.add_argument("--ranking-cache", type=Path, default=Path("/opt/ml/input/data/ranking"))
    parser.add_argument("--item-data", type=Path, default=Path("/opt/ml/input/data/items"))
    parser.add_argument("--model-dir", type=Path, default=Path("/opt/ml/input/data/trained"))
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/checkpoints"))
    parser.add_argument("--expected-ranking-id", required=True)
    parser.add_argument("--expected-items-id", required=True)
    parser.add_argument("--training-input-id", required=True)
    parser.add_argument("--code-commit", required=True)
    # SageMaker emits -k for a one-character hyperparameter. Keep both historic
    # spellings readable; new pipeline definitions use the descriptive name.
    parser.add_argument("--candidate-depth", "--k", "-k", dest="k", type=positive_int, default=800)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--chunk-size", type=positive_int, default=65536)
    parser.add_argument("--heartbeat-seconds", type=positive_seconds, default=30.0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args(argv)


def hyperparameters_to_argv(hyperparameters: Mapping[str, str]) -> list[str]:
    """Apply the toolkit's JSON decoding and scalar command-line conversion."""
    return worker_arguments(hyperparameters, program="evaluate.py")


def main() -> int:
    """Validate the exact serialized launch parameters without importing PyTorch."""
    hyperparameters = json.load(sys.stdin)
    arguments = hyperparameters_to_argv(hyperparameters)
    parsed = parse_args(arguments)
    if parsed.allow_cpu:
        raise ValueError("managed evaluation requires CUDA")
    print(
        json.dumps(
            {"status": "passed", "argv": arguments, "candidate_depth": parsed.k},
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
