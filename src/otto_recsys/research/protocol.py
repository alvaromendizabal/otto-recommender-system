"""Build auditable, chronological OTTO queries from immutable event Parquet files.

The old two-day benchmark remains unchanged. This protocol reserves earlier
periods for a new experiment and explicitly records prior dataset exposure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.runtime import Heartbeat

DAY_MS = 86_400_000


@dataclass(frozen=True)
class TemporalProtocol:
    """Exclusive boundaries; fit labels finish before selection queries begin."""

    history_end: int
    fit_end: int
    selection_end: int
    evaluation_end: int
    seed: int = 20260908
    fit_sessions: int = 100_000
    selection_sessions: int = 20_000

    def validate(self) -> None:
        if not 0 < self.history_end < self.fit_end < self.selection_end < self.evaluation_end:
            raise ValueError("temporal boundaries must be positive and strictly increasing")
        if self.fit_sessions < 1 or self.selection_sessions < 1 or self.seed < 0:
            raise ValueError("session budgets must be positive and seed nonnegative")

    def contract(self) -> dict[str, Any]:
        self.validate()
        return {
            **asdict(self),
            "boundaries": "left inclusive, right exclusive; whole session roles use first event",
            "labels": "next click, unique carts/orders; future timestamps and indices retained",
            "catalog": "no label-based filtering of unseen items",
            "query_sampling": "deterministic session hash; labels never used for inclusion",
            "evaluation_sampling": "all eligible sessions; no query or negative downsampling",
            "prior_exposure": (
                "Original data previously used by exploratory retrieval experiments. "
                "All learned components in this experiment must be refitted before history_end. "
                "This is a newly reserved temporal evaluation, not a claim of never-seen data."
            ),
        }


def sql_path(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_inventory(directory: Path) -> dict[str, Any]:
    """Verify downloaded source bytes against the captured inventory, if supplied."""
    parts = sorted(directory.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError("source requires part-*.parquet event files")
    hashes = {p.name: sha256_file(p) for p in parts}
    inventory_path = directory / "download_inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        expected = {Path(r["path"]).name: r["sha256"] for r in inventory}
        if expected != hashes:
            raise ValueError("source parts differ from the captured download inventory")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return {
        "parts": hashes,
        "conversion_manifest": manifest,
        "verification": "all Parquet bytes hashed; retained raw-source identity; no raw replay",
    }


def valid_corpus(output: Path, input_id: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads((output / "manifest.json").read_text())
        if receipt["input_id"] != input_id or receipt["status"] != "passed":
            return None
        if set(receipt["files"]) != {
            "history.parquet",
            "observed.parquet",
            "labels.parquet",
            "queries.parquet",
        }:
            return None
        for name, digest in receipt["files"].items():
            if sha256_file(output / name) != digest:
                return None
        return dict(receipt)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_temporal_corpus(
    source: Path,
    output: Path,
    protocol: TemporalProtocol,
    *,
    logger: logging.Logger,
    threads: int = 4,
    memory_gib: int = 6,
) -> dict[str, Any]:
    """Materialize strict time boundaries before training or viewing final scores."""
    protocol.validate()
    if threads < 1 or memory_gib < 1:
        raise ValueError("resource limits must be positive")
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        inventory = source_inventory(source)
        contract = {
            "schema_version": 1,
            "protocol": protocol.contract(),
            "source": inventory,
            "duckdb": duckdb.__version__,
            "code_sha256": sha256_file(Path(__file__)),
        }
        input_id = canonical_json_sha256(contract)
        old_contract = output / "contract.json"
        if old_contract.exists() and json.loads(old_contract.read_text()) != contract:
            raise ValueError("corpus directory belongs to a different protocol; preserve it")
        atomic_json(old_contract, contract)
        previous = valid_corpus(output, input_id)
        if previous is not None:
            logger.info("temporal_corpus_reused", extra={"input_id": input_id})
            return previous
        for interrupted in output.glob("*.parquet.tmp"):
            interrupted.unlink()
        started = time.perf_counter()
        connection = duckdb.connect()
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET memory_limit='{memory_gib}GB'")
        connection.execute("SET preserve_insertion_order=false")
        temporary = output / "working"
        temporary.mkdir(exist_ok=True)
        connection.execute(f"SET temp_directory={sql_path(temporary)}")
        source_glob = sql_path(source / "part-*.parquet")
        stage = {"phase": "session_boundaries"}
        try:
            with Heartbeat(
                logger,
                stage="temporal_corpus",
                interval_seconds=15,
                progress_provider=lambda: stage,
            ):
                connection.execute(
                    f"CREATE VIEW events AS SELECT * FROM read_parquet({source_glob})"
                )
                columns = {r[0] for r in connection.execute("DESCRIBE events").fetchall()}
                if columns != {"session", "aid", "ts", "event_type", "event_index"}:
                    raise ValueError("source event schema mismatch")
                bad = connection.execute("""SELECT count(*) FROM events WHERE session < 0
                    OR aid < 0 OR ts < 0 OR event_index < 0 OR event_type NOT IN (0,1,2)
                    OR session IS NULL OR aid IS NULL OR ts IS NULL
                    OR event_type IS NULL OR event_index IS NULL""").fetchone()
                if bad is None or bad[0]:
                    raise ValueError("invalid raw event values")
                rows = connection.execute("SELECT count(*) FROM events").fetchone()
                assert rows is not None
                declared = inventory["conversion_manifest"].get("events_processed", rows[0])
                if rows[0] != declared:
                    raise ValueError("source row count differs from conversion manifest")
                connection.execute(f"""CREATE TABLE sessions AS
                    WITH all_sessions AS (
                      SELECT session, min(ts) first_ts FROM events GROUP BY session
                    ), roles AS (
                      SELECT *, CASE
                        WHEN first_ts < {protocol.history_end} THEN 'history'
                        WHEN first_ts < {protocol.fit_end} THEN 'fit'
                        WHEN first_ts < {protocol.selection_end} THEN 'selection'
                        WHEN first_ts < {protocol.evaluation_end} THEN 'evaluation'
                        ELSE 'excluded' END split_role
                      FROM all_sessions
                    ) SELECT *, CASE split_role WHEN 'fit' THEN {protocol.fit_end}
                      WHEN 'selection' THEN {protocol.selection_end}
                      WHEN 'evaluation' THEN {protocol.evaluation_end}
                      ELSE {protocol.history_end} END period_end FROM roles""")
                stage["phase"] = "history"
                history = output / "history.parquet.tmp"
                history_receipt = output / "history.json"
                saved_history = None
                if history_receipt.exists() and (output / "history.parquet").exists():
                    saved_history = json.loads(history_receipt.read_text())
                history_valid = (
                    saved_history is not None
                    and saved_history.get("input_id") == input_id
                    and saved_history.get("sha256") == sha256_file(output / "history.parquet")
                )
                if not history_valid:
                    connection.execute(f"""COPY (SELECT e.* FROM events e
                        WHERE e.ts < {protocol.history_end}) TO {sql_path(history)}
                        (FORMAT PARQUET, COMPRESSION ZSTD)""")
                    history.replace(output / "history.parquet")
                    atomic_json(history_receipt, {
                        "input_id": input_id,
                        "sha256": sha256_file(output / "history.parquet"),
                    })
                else:
                    logger.info("temporal_history_reused")
                stage["phase"] = "query_prefixes"
                connection.execute("""CREATE TABLE eligible AS
                    SELECT e.*, s.split_role, s.period_end,
                    row_number() OVER (PARTITION BY e.session
                      ORDER BY e.event_index) event_position,
                    count(*) OVER (PARTITION BY e.session) n_events
                    FROM events e JOIN sessions s USING (session)
                    WHERE s.split_role IN ('fit','selection','evaluation')
                      AND e.ts < s.period_end""")
                connection.execute(f"""CREATE TABLE query_sessions AS
                    WITH distinct_sessions AS (
                      SELECT DISTINCT session, split_role, period_end, n_events,
                        1 + hash(session, {protocol.seed}, 'prefix') % (n_events-1) prefix_events
                      FROM eligible WHERE n_events >= 2
                    ), numbered AS (
                      SELECT *, row_number() OVER (PARTITION BY split_role
                        ORDER BY hash(session, {protocol.seed}, 'sample'), session) sample_rank
                      FROM distinct_sessions
                    ) SELECT * EXCLUDE (sample_rank) FROM numbered
                      WHERE split_role='evaluation'
                        OR (split_role='fit' AND sample_rank<={protocol.fit_sessions})
                        OR (split_role='selection'
                          AND sample_rank<={protocol.selection_sessions})""")
                connection.execute("""CREATE TABLE query_events AS SELECT e.*, q.prefix_events
                    FROM eligible e JOIN query_sessions q
                      USING (session, split_role, period_end, n_events)""")
                observed = output / "observed.parquet.tmp"
                connection.execute(f"""COPY (SELECT session, aid, ts, event_type,
                      event_index, split_role
                    FROM query_events WHERE event_position <= prefix_events
                    ORDER BY session,event_index)
                    TO {sql_path(observed)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                observed.replace(output / "observed.parquet")
                stage["phase"] = "timestamped_labels"
                labels = output / "labels.parquet.tmp"
                connection.execute(f"""COPY (
                    WITH future AS (SELECT *, row_number() OVER
                      (PARTITION BY session,event_type ORDER BY event_index) action_position
                      FROM query_events WHERE event_position > prefix_events)
                    SELECT session, split_role, CASE event_type WHEN 0 THEN 'clicks'
                      WHEN 1 THEN 'carts' ELSE 'orders' END objective, aid,
                      min(ts) label_ts, min(event_index) label_event_index
                    FROM future WHERE event_type<>0 OR action_position=1
                    GROUP BY session, split_role, event_type, aid ORDER BY session,objective,aid
                    ) TO {sql_path(labels)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                labels.replace(output / "labels.parquet")
                queries = output / "queries.parquet.tmp"
                connection.execute(f"""COPY (
                    WITH prefixes AS (SELECT session, split_role,
                      min(ts) first_ts, max(ts) query_ts,
                        max(event_index) observed_last_index, count(*) observed_events
                      FROM read_parquet({sql_path(output / "observed.parquet")})
                      GROUP BY session,split_role), counts AS (
                      SELECT session,objective,count(*) true_items
                      FROM read_parquet({sql_path(output / "labels.parquet")})
                      GROUP BY session,objective)
                    SELECT p.*, q.period_end, o.objective, coalesce(c.true_items,0) true_items,
                      least(20,coalesce(c.true_items,0)) recall_denominator
                    FROM prefixes p JOIN query_sessions q USING (session,split_role)
                    CROSS JOIN (VALUES ('clicks'),('carts'),('orders')) o(objective)
                    LEFT JOIN counts c ON c.session=p.session AND c.objective=o.objective
                    ORDER BY p.session,o.objective
                    ) TO {sql_path(queries)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
                queries.replace(output / "queries.parquet")
                invalid_labels = connection.execute(f"""SELECT count(*)
                    FROM read_parquet({sql_path(output / "labels.parquet")}) l
                    JOIN read_parquet({sql_path(output / "queries.parquet")}) q
                      USING (session,split_role,objective)
                    WHERE l.label_ts < q.query_ts OR l.label_ts >= q.period_end
                      OR l.label_event_index <= q.observed_last_index""").fetchone()
                if invalid_labels is None or invalid_labels[0]:
                    raise ValueError("future labels violate an observed prefix or time boundary")
                summaries = connection.execute(f"""SELECT split_role, count(DISTINCT session),
                    min(first_ts),max(query_ts),max(period_end),sum(recall_denominator)
                    FROM read_parquet({sql_path(output / "queries.parquet")}) GROUP BY split_role
                    ORDER BY split_role""").fetchall()
                if {r[0] for r in summaries} != {"fit", "selection", "evaluation"}:
                    raise ValueError("protocol has an empty split_role")
                files = {
                    name: sha256_file(output / name)
                    for name in (
                        "history.parquet",
                        "observed.parquet",
                        "labels.parquet",
                        "queries.parquet",
                    )
                }
                receipt = {
                    "status": "passed",
                    "input_id": input_id,
                    "files": files,
                    "source_rows": rows[0],
                    "protocol": protocol.contract(),
                    "roles": {
                        r[0]: {
                            "sessions": r[1],
                            "first_query_event": r[2],
                            "last_observed_event": r[3],
                            "exclusive_label_end": r[4],
                            "pooled_denominator": r[5],
                        }
                        for r in summaries
                    },
                    "elapsed_seconds": time.perf_counter() - started,
                }
                atomic_json(output / "manifest.json", receipt)
                logger.info(
                    "temporal_corpus_complete",
                    extra={"input_id": input_id, "elapsed_seconds": receipt["elapsed_seconds"]},
                )
                return receipt
        finally:
            connection.close()
