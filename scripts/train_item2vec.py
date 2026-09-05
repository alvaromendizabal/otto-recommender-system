from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.item2vec import Item2VecConfig, train_item2vec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sessions", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--negative", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", type=float, default=1e-4)
    parser.add_argument("--ns-exponent", type=float, default=0.75)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    config = Item2VecConfig(
        vector_size=args.vector_size,
        window=args.window,
        negative=args.negative,
        epochs=args.epochs,
        workers=args.workers,
        seed=args.seed,
        sample=args.sample,
        ns_exponent=args.ns_exponent,
    )

    logger = configure_logging("item2vec")
    started = time.perf_counter()
    manifest = train_item2vec(
        args.train_sessions,
        args.validation_manifest,
        args.output_dir,
        logger=logger,
        config=config,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"wall_elapsed_seconds={time.perf_counter() - started:.3f}")
    print("OTTO_ITEM2VEC_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
