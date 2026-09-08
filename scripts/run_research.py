"""Run the chronological OTTO research stages from verified durable inputs."""

from __future__ import annotations

import argparse
import json
import tomllib
from datetime import datetime
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.research.protocol import TemporalProtocol, build_temporal_corpus
from otto_recsys.research.retrievers import build_retrievers


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
    parser.add_argument("--stage", choices=["prepare", "retrieval"], required=True)
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
    else:
        result = build_retrievers(
            args.output / "corpus",
            args.output / "retrieval",
            config["retrieval"],
            logger=logger,
            threads=config["training"]["threads"],
            memory_gib=config["resources"]["memory_gib"],
        )
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
