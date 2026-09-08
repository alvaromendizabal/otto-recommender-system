"""Run the chronological OTTO research stages from verified durable inputs."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.dataset import build_cache
from otto_recsys.research.features import feature_catalog
from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus
from otto_recsys.research.retrievers import build_retrievers
from otto_recsys.research.screening import screen_cache


def load_protocol(path: Path) -> tuple[TemporalProtocol, dict]:
    config = tomllib.loads(path.read_text())
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
        "--stage", choices=["prepare", "retrieval", "features", "screen"], required=True
    )
    parser.add_argument("--config", type=Path, default=Path("configs/research.toml"))
    parser.add_argument("--source", type=Path, default=Path("data/source_audit/processed/train"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/research"))
    args = parser.parse_args()
    protocol, config = load_protocol(args.config)
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
    else:
        result = screen_cache(
            args.output / "screening_cache",
            args.output / "screening",
            max_retained=config["features"]["max_retained"],
            seed=protocol.seed,
            logger=logger,
            threads=config["training"]["threads"],
        )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
