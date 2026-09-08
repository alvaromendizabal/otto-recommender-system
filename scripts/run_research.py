"""Run the chronological OTTO research stages from verified durable inputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.configuration import read_config
from otto_recsys.research.dataset import build_cache
from otto_recsys.research.evaluation import run_evaluation
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.materialize import model_cache
from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus
from otto_recsys.research.retrievers import build_retrievers
from otto_recsys.research.screening import screen_cache
from otto_recsys.research.study import run_ablations


def load_protocol(path: Path) -> tuple[TemporalProtocol, dict]:
    config = read_config(path)
    values = dict(config["protocol"])
    for name in ("history_end", "fit_end", "selection_end", "evaluation_end"):
        timestamp = datetime.fromisoformat(values[name])
        if timestamp.utcoffset() is None:
            raise ValueError("protocol timestamps require an explicit timezone")
        values[name] = int(timestamp.timestamp() * 1000)
    return TemporalProtocol(**values), config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=[
            "prepare",
            "retrieval",
            "features",
            "screen",
            "model_features",
            "ablate",
            "evaluate",
        ],
        required=True,
    )
    parser.add_argument("--role", choices=["fit", "selection"], default="fit")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--memory-gib", type=int, default=None)
    parser.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    parser.add_argument("--source", type=Path, default=Path("data/source_audit/processed/train"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    protocol, config = load_protocol(args.config)
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("thread limit must be positive")
        config["training"]["threads"] = args.threads
    if args.memory_gib is not None:
        if args.memory_gib < 1:
            raise ValueError("memory limit must be positive")
        config["resources"]["memory_gib"] = args.memory_gib
    args.output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("research", log_dir=args.output / "logs")
    if args.stage == "prepare":
        result = build_temporal_corpus(
            args.source,
            args.output / "corpus",
            protocol,
            logger=logger,
            threads=config["training"]["threads"],
            memory_gib=config["resources"]["memory_gib"],
        )
    elif args.stage == "retrieval":
        result = build_retrievers(
            args.output / "corpus",
            args.output / "retrieval",
            config["retrieval"],
            logger=logger,
            threads=config["training"]["threads"],
            memory_gib=config["resources"]["memory_gib"],
        )
    elif args.stage == "features":
        result = build_cache(
            args.output / "corpus",
            args.output / "retrieval",
            args.output / "screening_cache",
            role="fit",
            names=tuple(f.name for f in feature_catalog()),
            candidate_budget=400,
            negative_budget=config["training"]["negative_budget"],
            seed=protocol.seed,
            max_rows=config["features"]["screening_rows"],
            logger=logger,
        )
    elif args.stage == "screen":
        result = screen_cache(
            args.output / "screening_cache",
            args.output / "screening",
            max_retained=config["features"]["max_retained"],
            seed=protocol.seed,
            logger=logger,
            threads=config["training"]["threads"],
        )
    elif args.stage == "model_features":
        selected = json.loads((args.output / "screening/selection.json").read_text())
        result = model_cache(
            args.output / "corpus",
            args.output / "retrieval",
            args.output / f"{args.role}_cache",
            role=args.role,
            names=tuple(selected["retained"]),
            budget=400,
            negatives=config["training"]["negative_budget"],
            seed=protocol.seed,
            workers=args.workers,
            logger=logger,
        )
    elif args.stage == "ablate":
        result = run_ablations(
            args.output, {**config["training"], "seed": protocol.seed}, logger=logger
        )
    else:
        result = run_evaluation(
            args.output,
            seed=protocol.seed,
            workers=args.workers,
            threads=config["training"]["threads"],
            logger=logger,
        )
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"files", "features", "pilots", "retained"}
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
