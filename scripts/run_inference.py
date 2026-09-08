"""Run deployment history refresh and notebook-reproducible competition inference."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints
from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.deployment import prepare_history
from otto_recsys.research.inference import run_prediction
from otto_recsys.research.retrievers import build_retrievers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["prepare", "retrieval", "predict"], required=True)
    parser.add_argument("--train", type=Path, default=Path("data/source_audit/processed/train"))
    parser.add_argument("--test", type=Path, default=Path("data/source_audit/processed/test"))
    parser.add_argument("--models", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference"))
    parser.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-gib", type=int, default=None)
    parser.add_argument("--checkpoint-uri", default=None)
    parser.add_argument("--owner-account", default=None)
    parser.add_argument("--region", default="us-west-2")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    if args.memory_gib is not None:
        config["resources"]["memory_gib"] = args.memory_gib
    logger = configure_logging("competition_inference", log_dir=args.output / "logs")
    publish = None
    if args.checkpoint_uri:
        if not args.owner_account:
            raise ValueError("cloud checkpoint publication requires the expected bucket owner")
        storage = ResearchCheckpoints(
            args.output,
            args.checkpoint_uri,
            region=args.region,
            owner_account=args.owner_account,
            logger=logger,
        )
        publish = storage.publish
    if args.stage == "prepare":
        report = prepare_history(
            args.train,
            args.test,
            args.output / "corpus",
            logger=logger,
            threads=args.threads,
            memory_gib=config["resources"]["memory_gib"],
        )
    elif args.stage == "retrieval":
        report = build_retrievers(
            args.output / "corpus",
            args.output / "retrieval",
            config["retrieval"],
            logger=logger,
            threads=args.threads,
            memory_gib=config["resources"]["memory_gib"],
        )
    else:
        report = run_prediction(
            args.models,
            args.test,
            args.output / "retrieval",
            args.output / "prediction",
            workers=args.workers,
            threads=args.threads,
            logger=logger,
            publish=publish,
        )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"files", "feature_order"}}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
