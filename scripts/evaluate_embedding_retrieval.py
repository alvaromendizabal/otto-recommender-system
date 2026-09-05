from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.embedding_evaluation import evaluate_embedding_retrieval


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-cache", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--buckets", type=int, default=32)
    parser.add_argument("--ks", nargs="+", type=int, default=[20, 50, 100, 200])
    parser.add_argument("--ann-k", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    logger = configure_logging("embedding_evaluation")
    started = time.perf_counter()

    manifest = evaluate_embedding_retrieval(
        args.validation_cache,
        args.vectors,
        args.index,
        args.output_dir,
        logger=logger,
        buckets=args.buckets,
        ks=tuple(args.ks),
        ann_k=args.ann_k,
        ef_search=args.ef_search,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"wall_elapsed_seconds={time.perf_counter() - started:.3f}")
    print("OTTO_EMBEDDING_EVALUATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
