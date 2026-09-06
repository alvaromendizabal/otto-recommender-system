"""Dependency-free managed ANN benchmark argument contract."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from .evaluation_cli import positive_int, positive_seconds
from .sagemaker_args import worker_arguments


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name, channel in (
        ("ranking-cache", "ranking"),
        ("item-data", "items"),
        ("model-dir", "trained"),
        ("reference-dir", "reference"),
    ):
        parser.add_argument("--" + name, type=Path, default=Path("/opt/ml/input/data") / channel)
    parser.add_argument("--output-dir", type=Path, default=Path("/opt/ml/ann"))
    for name in ("run-id", "code-commit", "reference-input-id", "reference-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--checkpoint-uri")
    parser.add_argument("--region", default="us-west-2")
    for name, default in (
        ("sample-sessions", 4096),
        ("nlist", 1024),
        ("train-items", 65536),
        ("train-iterations", 20),
        ("candidate-depth", 800),
        ("threads", 4),
        ("batch-size", 128),
        ("index-shard-rows", 262144),
        ("latency-queries", 128),
        ("latency-repeats", 3),
        ("warmup-queries", 32),
        ("seed", 20260906),
    ):
        parser.add_argument("--" + name, type=positive_int, default=default)
    parser.add_argument("--probes", default="32,64,128,256")
    parser.add_argument(
        "--export-fold-predictions", type=str.lower, choices=("true", "false"), default="true"
    )
    parser.add_argument("--target-overlap", type=float, default=0.98)
    parser.add_argument("--heartbeat-seconds", type=positive_seconds, default=30)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.probes = tuple(int(value) for value in args.probes.split(","))
    except ValueError:
        parser.error("probes must be comma-separated integers")
    if (
        not args.probes
        or tuple(sorted(set(args.probes))) != args.probes
        or args.probes[0] < 1
        or args.probes[-1] > args.nlist
    ):
        parser.error("probes must be unique, increasing, and within [1, nlist]")
    if args.sample_sessions < 4 or args.sample_sessions % 2:
        parser.error("sample-sessions must be even and at least four")
    if args.candidate_depth < 20:
        parser.error("candidate-depth must cover the official top-20 metric")
    if args.train_items < 39 * args.nlist:
        parser.error("train-items must provide at least 39 samples per centroid")
    if not math.isfinite(args.target_overlap) or not 0 < args.target_overlap <= 1:
        parser.error("target-overlap must be finite and in (0, 1]")
    return args


def hyperparameters_to_argv(parameters: Mapping[str, str]) -> list[str]:
    return worker_arguments(parameters, program="benchmark.py")


def main() -> int:
    argv = hyperparameters_to_argv(json.load(sys.stdin))
    args = parse_args(argv)
    if args.allow_cpu or not args.checkpoint_uri:
        raise ValueError("managed benchmark requires a GPU encoder and durable S3 output")
    print(
        json.dumps(
            {
                "status": "passed",
                "argv": argv,
                "export_fold_predictions": args.export_fold_predictions,
                "nlist": args.nlist,
                "probes": args.probes,
                "sample_sessions": args.sample_sessions,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
