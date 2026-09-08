"""Partitioned historical co-visitation and popularity fitted before query time."""

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
from otto_recsys.runtime import Heartbeat

HISTORY_HOURS = (1, 6, 12, 24, 72, 168, 336)
CHANNELS = ("time", "cart", "order")


def verified_file(path: Path, input_id: str) -> bool:
    try:
        receipt = json.loads(path.with_suffix(".json").read_text())
        return bool(receipt["input_id"] == input_id and receipt["sha256"] == sha256_file(path))
    except (OSError, ValueError, TypeError, KeyError):
        return False


def commit_file(path: Path, input_id: str, elapsed: float) -> None:
    atomic_json(
        path.with_suffix(".json"),
        {
            "input_id": input_id,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "elapsed_seconds": elapsed,
        },
    )


def build_retrievers(
    corpus: Path,
    output: Path,
    config: dict[str, Any],
    *,
    logger: logging.Logger,
    threads: int = 4,
    memory_gib: int = 6,
) -> dict[str, Any]:
    """Fit without any query labels, preserving each verified graph partition."""
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        corpus_manifest = json.loads((corpus / "manifest.json").read_text())
        history = corpus / "history.parquet"
        history_hash = sha256_file(history)
        if history_hash != corpus_manifest["files"]["history.parquet"]:
            raise ValueError("historical input checksum mismatch")
        cutoff = corpus_manifest["protocol"]["history_end"]
        tail = int(config["history_tail"])
        window = int(config["position_window"])
        partitions = int(config["source_partitions"])
        top_k = int(config["neighbors_per_score"])
        hours = int(config["time_window_hours"])
        if min(tail, window, partitions, top_k, hours, threads, memory_gib) < 1:
            raise ValueError("retrieval configuration must be positive")
        contract = {
            "history_sha256": history_hash,
            "history_end": cutoff,
            "config": config,
            "code_sha256": sha256_file(Path(__file__)),
            "duckdb": duckdb.__version__,
            "query_labels_used": False,
        }
        input_id = canonical_json_sha256(contract)
        contract_path = output / "contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("retriever directory has a different fitted-history contract")
        atomic_json(contract_path, contract)
        connection = duckdb.connect()
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit='{memory_gib}GB'")
        connection.execute("SET preserve_insertion_order=false")
        (output / "working").mkdir(exist_ok=True)
        connection.execute(f"SET temp_directory={sql_path(output / 'working')}")
        progress = {"phase": "history_validation", "bucket": 0, "buckets": partitions}
        start = time.perf_counter()
        try:
            with Heartbeat(
                logger,
                stage="research_retrieval",
                interval_seconds=15,
                progress_provider=progress.copy,
            ):
                latest = connection.execute(
                    f"SELECT max(ts) FROM read_parquet({sql_path(history)})"
                ).fetchone()
                if latest is None or latest[0] is None or latest[0] >= cutoff:
                    raise ValueError("retriever history crosses the query fitting cutoff")
                progress["phase"] = "historical_statistics"
                statistics = output / "history_statistics.parquet"
                if not verified_file(statistics, input_id):
                    stats_start = time.perf_counter()
                    expressions = []
                    for action, kind in (("all", None), ("clicks", 0), ("carts", 1), ("orders", 2)):
                        for horizon in (0, *HISTORY_HOURS):
                            conditions = ["true"] if kind is None else [f"event_type={kind}"]
                            if horizon:
                                conditions.append(f"ts >= {cutoff - horizon * 3_600_000}")
                            label = f"h{horizon}" if horizon else "all"
                            expressions.append(
                                f"count(*) FILTER (WHERE {' AND '.join(conditions)})"
                                f"::FLOAT AS hist_{action}_{label}"
                            )
                    temporary = statistics.with_suffix(".parquet.tmp")
                    temporary.unlink(missing_ok=True)
                    connection.execute(f"""COPY (
                        SELECT aid, max(ts) history_last_ts, min(ts) history_first_ts,
                          {",".join(expressions)} FROM read_parquet({sql_path(history)})
                        GROUP BY aid ORDER BY aid
                        ) TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                    temporary.replace(statistics)
                    commit_file(statistics, input_id, time.perf_counter() - stats_start)
                progress["phase"] = "history_tail"
                thin = output / "history_tail.parquet"
                if not verified_file(thin, input_id):
                    thin_start = time.perf_counter()
                    temporary = thin.with_suffix(".parquet.tmp")
                    temporary.unlink(missing_ok=True)
                    connection.execute(f"""COPY (
                        SELECT * FROM read_parquet({sql_path(history)})
                        QUALIFY row_number() OVER
                          (PARTITION BY session ORDER BY event_index DESC) <= {tail}
                        ) TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                    temporary.replace(thin)
                    commit_file(thin, input_id, time.perf_counter() - thin_start)
                parts = output / "parts"
                parts.mkdir(exist_ok=True)
                reused = 0
                for bucket in range(partitions):
                    progress.update(phase="graph_partition", bucket=bucket)
                    path = parts / f"part-{bucket:03d}.parquet"
                    if verified_file(path, input_id):
                        reused += 1
                        logger.info("research_graph_part_reused", extra={"bucket": bucket})
                        continue
                    part_start = time.perf_counter()
                    temporary = path.with_suffix(".parquet.tmp")
                    temporary.unlink(missing_ok=True)
                    connection.execute(f"""COPY (
                        WITH pairs AS (
                          SELECT a.session, a.aid source_aid, b.aid target_aid,
                            b.event_type target_type,
                            exp(-abs(b.ts-a.ts)/21600000.0)
                              / sqrt(abs(b.event_index::INTEGER-a.event_index::INTEGER)) weight
                          FROM read_parquet({sql_path(thin)}) a
                          JOIN read_parquet({sql_path(thin)}) b ON a.session=b.session
                          WHERE a.aid % {partitions} = {bucket} AND a.aid<>b.aid
                            AND abs(b.event_index::INTEGER-a.event_index::INTEGER)
                              BETWEEN 1 AND {window}
                            AND abs(b.ts-a.ts) <= {hours * 3_600_000}
                        ), unique_pairs AS (
                          SELECT session, source_aid,target_aid, max(weight) time_score,
                            max(weight * CASE target_type
                              WHEN 1 THEN 3.0 WHEN 2 THEN 1.5 ELSE 0.1 END) cart_score,
                            max(weight * CASE target_type
                              WHEN 2 THEN 4.0 WHEN 1 THEN 2.0 ELSE 0.05 END) order_score
                          FROM pairs GROUP BY session,source_aid,target_aid
                        ), totals AS (
                          SELECT source_aid,target_aid,sum(time_score)::FLOAT time_score,
                            sum(cart_score)::FLOAT cart_score, sum(order_score)::FLOAT order_score
                          FROM unique_pairs GROUP BY source_aid,target_aid
                        ), ranked AS (
                          SELECT *, row_number() OVER (PARTITION BY source_aid
                              ORDER BY time_score DESC,target_aid) time_rank,
                            row_number() OVER (PARTITION BY source_aid
                              ORDER BY cart_score DESC,target_aid) cart_rank,
                            row_number() OVER (PARTITION BY source_aid
                              ORDER BY order_score DESC,target_aid) order_rank
                          FROM totals
                        ) SELECT source_aid,target_aid,time_score,cart_score,order_score
                          FROM ranked WHERE least(time_rank,cart_rank,order_rank)<={top_k}
                          ORDER BY source_aid,target_aid
                        ) TO {sql_path(temporary)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                    temporary.replace(path)
                    elapsed = time.perf_counter() - part_start
                    commit_file(path, input_id, elapsed)
                    logger.info(
                        "research_graph_part_complete",
                        extra={"bucket": bucket, "elapsed_seconds": elapsed},
                    )
                files = {
                    str(p.relative_to(output)): sha256_file(p)
                    for p in [statistics, thin, *sorted(parts.glob("part-*.parquet"))]
                }
                receipt = {
                    "status": "passed",
                    "input_id": input_id,
                    "files": files,
                    "history_end": cutoff,
                    "observed_history_max_ts": latest[0],
                    "query_labels_used": False,
                    "completed_parts": partitions,
                    "reused_parts": reused,
                    "elapsed_seconds": time.perf_counter() - start,
                }
                atomic_json(output / "manifest.json", receipt)
                logger.info(
                    "research_retrievers_complete",
                    extra={"elapsed_seconds": receipt["elapsed_seconds"]},
                )
                return receipt
        finally:
            connection.close()
