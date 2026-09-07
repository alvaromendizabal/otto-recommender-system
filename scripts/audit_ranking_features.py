"""Independently reconcile persisted ranking features against frozen inputs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import duckdb

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.ranking.feature_cache import valid_part, write_json
from otto_recsys.runtime import Heartbeat


def audit(cache: Path, features: Path) -> dict[str, Any]:
    started = time.perf_counter()
    contract = json.loads((features / "feature_contract.json").read_text())
    manifest = json.loads((features / "manifest.json").read_text())
    input_id = canonical_json_sha256(contract)
    if manifest["input_id"] != input_id:
        raise ValueError("feature identity mismatch")
    for name, checksum in contract["input_sha256"].items():
        if sha256_file(cache / f"{name}.parquet") != checksum:
            raise ValueError("input checksum mismatch")
    receipts = [valid_part(features, bucket, input_id) for bucket in range(contract["buckets"])]
    if any(receipt is None for receipt in receipts):
        raise ValueError("incomplete feature checkpoint")
    if canonical_json_sha256([r["files"] for r in receipts if r]) != manifest["parts_sha256"]:
        raise ValueError("feature family checksum mismatch")
    with duckdb.connect() as con:
        con.execute("SET threads = 2")
        con.execute("SET memory_limit = '1GB'")
        for name in ("events", "labels", "examples"):
            con.execute(
                f"CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)",
                [str(cache / f"{name}.parquet")],
            )
        for name in ("sessions", "items", "queries"):
            con.execute(
                f"CREATE TEMP TABLE {name} AS SELECT * FROM read_parquet(?)",
                [str(features / "parts/part-*" / f"{name}.parquet")],
            )
        bad_sessions = con.execute("""
            SELECT count(*) FROM sessions s FULL JOIN (
                SELECT session, count(*) n, count(DISTINCT aid) u, min(ts) first_ts,
                       max(ts) last_ts, count(*) FILTER (event_type=0) clicks,
                       count(*) FILTER (event_type=1) carts, count(*) FILTER (event_type=2) orders
                FROM events GROUP BY session
            ) e USING(session)
            WHERE s.session IS NULL OR e.session IS NULL OR s.session_events != n
               OR s.session_unique_items != u OR s.first_ts != e.first_ts
               OR s.last_ts != e.last_ts OR s.session_duration_ms != e.last_ts-e.first_ts
               OR s.session_clicks != e.clicks OR s.session_carts != e.carts
               OR s.session_orders != e.orders
        """).fetchone()
        bad_items = con.execute("""
            SELECT count(*) FROM items i FULL JOIN (
                SELECT session, aid, count(*) n, max(ts) last_ts,
                       arg_max(event_type, event_index) last_type,
                       count(*) FILTER (event_type=0) clicks,
                       count(*) FILTER (event_type=1) carts, count(*) FILTER (event_type=2) orders
                FROM events GROUP BY session, aid
            ) e USING(session, aid) LEFT JOIN sessions s ON e.session=s.session
            WHERE i.session IS NULL OR e.session IS NULL OR i.item_events != n
               OR i.item_age_ms != s.last_ts-e.last_ts OR i.item_last_type != e.last_type
               OR abs(i.item_event_share - n::DOUBLE/s.session_events) > 1e-12
               OR i.item_clicks != e.clicks OR i.item_carts != e.carts OR i.item_orders != e.orders
        """).fetchone()
        bad_queries = con.execute("""
            SELECT count(*) FROM queries q FULL JOIN (
                SELECT e.session, o.objective, count(l.aid) true_items,
                       least(20, count(l.aid)) denominator, e.session::BIGINT*3+o.id query_id
                FROM examples e CROSS JOIN (VALUES ('clicks',0),('carts',1),('orders',2))
                     o(objective,id)
                LEFT JOIN labels l ON e.session=l.session AND o.objective=l.objective
                GROUP BY e.session,o.objective,o.id
            ) e USING(session, objective)
            WHERE q.session IS NULL OR e.session IS NULL OR q.true_items != e.true_items
               OR q.recall_denominator != e.denominator OR q.query_id != e.query_id
        """).fetchone()
        duplicate_keys = con.execute("""
            SELECT (SELECT count(*)-count(DISTINCT session) FROM sessions)
                 + (SELECT count(*)-count(DISTINCT (session,aid)) FROM items)
                 + (SELECT count(*)-count(DISTINCT query_id) FROM queries)
        """).fetchone()
        mismatches = dict(
            zip(
                ("sessions", "items", "queries", "duplicate_keys"),
                [
                    row[0] if row else -1
                    for row in (bad_sessions, bad_items, bad_queries, duplicate_keys)
                ],
                strict=True,
            )
        )
        if any(mismatches.values()):
            raise ValueError(f"feature reconciliation failed: {mismatches}")
        distribution = con.execute("""
            SELECT session_events, count(*) sessions FROM sessions
            GROUP BY session_events ORDER BY session_events
        """).fetchall()
    return {
        "status": "passed",
        "input_id": input_id,
        "timestamp": utc_now_iso(),
        "elapsed_seconds": time.perf_counter() - started,
        "verified_buckets": len(receipts),
        "mismatches": mismatches,
        "parts_sha256": manifest["parts_sha256"],
        "feature_contract_sha256": sha256_file(features / "feature_contract.json"),
        "session_length_distribution": [
            {"observed_events": n, "sessions": count} for n, count in distribution
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ranking-cache", type=Path, default=Path("data/interim/ranking_training_cache")
    )
    parser.add_argument("--features", type=Path, default=Path("data/interim/ranking_features"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/metrics/ranking_features_audit.json")
    )
    args = parser.parse_args()
    logger = configure_logging("ranking_features_audit")
    started = time.perf_counter()
    logger.info("ranking_features_audit_start")
    try:
        with Heartbeat(logger, stage="ranking_features_audit", interval_seconds=5):
            result = audit(args.ranking_cache, args.features)
        write_json(args.output, result)
        logger.info("ranking_features_audit_complete", extra={"status": "passed"})
    except Exception:
        logger.exception("ranking_features_audit_failed", extra={"status": "failed"})
        raise
    finally:
        logger.info(
            "ranking_features_audit_attempt_complete",
            extra={
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
    print("OTTO_RANKING_FEATURES_AUDIT_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
