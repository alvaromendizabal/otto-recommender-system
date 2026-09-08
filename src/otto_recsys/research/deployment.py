"""Refresh deployment history from chronological, disjoint observed inputs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import duckdb

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.protocol import atomic_json, sql_path
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat


def source_identity(directory: Path) -> dict[str, str]:
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        raise ValueError("inference input requires canonical event partitions")
    return {path.name: sha256_file(path) for path in parts}


def prepare_history(
    source: Path,
    test: Path,
    output: Path,
    *,
    logger: logging.Logger,
    threads: int = 4,
    memory_gib: int = 12,
) -> dict[str, Any]:
    """Refresh the feature store after final evaluation, using no test targets."""
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        contract = {
            "training_files": source_identity(source),
            "test_files": source_identity(test),
            "code_sha256": sha256_file(Path(__file__)),
            "purpose": "post-evaluation competition inference; frozen model weights",
        }
        input_id = canonical_json_sha256(contract)
        contract_path = output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("inference history has a different source contract")
        atomic_json(contract_path, contract)
        connection = duckdb.connect()
        connection.execute(f"SET threads={int(threads)}")
        connection.execute(f"SET memory_limit='{int(memory_gib)}GB'")
        connection.execute("SET preserve_insertion_order=false")
        start = time.perf_counter()
        try:
            with Heartbeat(logger, stage="inference_history", interval_seconds=15):
                connection.execute(
                    "CREATE VIEW history AS SELECT * FROM read_parquet("
                    f"{sql_path(source / 'part-*.parquet')})"
                )
                connection.execute(
                    "CREATE VIEW observed AS SELECT * FROM read_parquet("
                    f"{sql_path(test / 'part-*.parquet')})"
                )
                bounds = connection.execute("""SELECT
                    (SELECT max(ts) FROM history), (SELECT min(ts) FROM observed),
                    (SELECT count(*) FROM history),
                    (SELECT count(DISTINCT session) FROM observed),
                    (SELECT count(*) FROM (SELECT DISTINCT session FROM history)
                     INNER JOIN (SELECT DISTINCT session FROM observed) USING(session))
                    """).fetchone()
                if bounds is None or bounds[0] is None or bounds[1] is None:
                    raise ValueError("inference history and test events must be nonempty")
                if bounds[0] >= bounds[1] or bounds[4]:
                    raise ValueError(
                        "competition history must precede disjoint observed test sessions"
                    )
                history = output / "history.parquet"
                if not verified_file(history, input_id):
                    temporary = history.with_suffix(".parquet.tmp")
                    connection.execute(
                        f"COPY history TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)"
                    )
                    temporary.replace(history)
                    atomic_json(
                        history.with_suffix(".json"),
                        {"input_id": input_id, "sha256": sha256_file(history)},
                    )
                report = {
                    "status": "passed",
                    "input_id": input_id,
                    "protocol": {"history_end": int(bounds[1])},
                    "history_max_ts": int(bounds[0]),
                    "history_events": int(bounds[2]),
                    "test_sessions": int(bounds[3]),
                    "overlapping_sessions": 0,
                    "files": {"history.parquet": sha256_file(history)},
                    "elapsed_seconds": time.perf_counter() - start,
                    "scope": contract["purpose"],
                }
                atomic_json(output / "manifest.json", report)
                return report
        finally:
            connection.close()
