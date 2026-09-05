from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from otto_recsys.logging_utils import configure_logging
from otto_recsys.retrieval.faiss_index import FaissConfig, build_faiss_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    args = parser.parse_args()

    config = FaissConfig(
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        batch_size=args.batch_size,
        threads=args.threads,
    )

    logger = configure_logging("faiss_index")
    started = time.perf_counter()
    manifest = build_faiss_index(
        args.vectors,
        args.output_dir,
        logger=logger,
        config=config,
        heartbeat_seconds=args.heartbeat_seconds,
    )

    print(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    print(f"wall_elapsed_seconds={time.perf_counter() - started:.3f}")
    print("OTTO_FAISS_INDEX_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
