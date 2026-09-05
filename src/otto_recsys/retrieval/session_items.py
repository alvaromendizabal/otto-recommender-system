from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.runtime import Heartbeat

_MEMORY_RE = re.compile(r"^[1-9][0-9]*(MB|GB)$")


@dataclass(frozen=True)
class SessionItemManifest:
    split_ts: int
    max_items_per_session: int
    rows: int
    output_sha256: str


def _sql_literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _configure_connection(
    connection: duckdb.DuckDBPyConnection,
    *,
    threads: int,
    memory_limit: str,
    temp_directory: Path,
) -> None:
    if threads <= 0:
        raise ValueError("threads must be positive")

    if not _MEMORY_RE.fullmatch(memory_limit):
        raise ValueError(
            "memory_limit must look like '2GB' or '512MB'"
        )

    temp_directory.mkdir(parents=True, exist_ok=True)

    connection.execute(f"SET threads = {threads}")
    connection.execute(
        f"SET memory_limit = '{memory_limit}'"
    )
    connection.execute(
        "SET preserve_insertion_order = false"
    )
    connection.execute(
        "SET temp_directory = "
        f"'{_sql_literal(temp_directory)}'"
    )


def build_session_item_cache(
    processed_pattern: str | Path,
    validation_manifest_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    logger: logging.Logger,
    max_items_per_session: int = 20,
    threads: int = 2,
    memory_limit: str = "2GB",
    temp_directory: str | Path = "data/interim/duckdb",
    heartbeat_seconds: float = 30.0,
) -> SessionItemManifest:
    """Build a bounded, leakage-safe session-item retrieval table."""
    if max_items_per_session <= 1:
        raise ValueError(
            "max_items_per_session must be greater than one"
        )

    validation_payload = json.loads(
        Path(validation_manifest_path).read_text(
            encoding="utf-8"
        )
    )

    split_ts = validation_payload.get("split_ts")

    if not isinstance(split_ts, int):
        raise ValueError(
            "validation manifest does not contain integer split_ts"
        )

    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    output_manifest = Path(manifest_path).resolve()
    output_manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    temporary_output.unlink(missing_ok=True)

    connection = duckdb.connect(database=":memory:")

    _configure_connection(
        connection,
        threads=threads,
        memory_limit=memory_limit,
        temp_directory=Path(temp_directory).resolve(),
    )

    source_sql = _sql_literal(
        Path(processed_pattern).resolve()
    )
    output_sql = _sql_literal(temporary_output)

    query = f"""
    COPY (
        WITH deduplicated AS (
            SELECT
                session,
                aid,
                arg_max(ts, event_index) AS ts,
                arg_max(event_type, event_index) AS event_type,
                max(event_index) AS event_index
            FROM read_parquet('{source_sql}')
            WHERE ts < {split_ts}
            GROUP BY session, aid
        ),
        ranked AS (
            SELECT
                session,
                aid,
                ts,
                event_type,
                event_index,
                row_number() OVER (
                    PARTITION BY session
                    ORDER BY event_index DESC, aid
                ) AS recency_rank,
                count(*) OVER (
                    PARTITION BY session
                ) AS unique_item_count
            FROM deduplicated
        )
        SELECT
            session::INTEGER AS session,
            aid::INTEGER AS aid,
            ts::BIGINT AS ts,
            event_type::TINYINT AS event_type,
            event_index::USMALLINT AS event_index,
            recency_rank::USMALLINT AS recency_rank
        FROM ranked
        WHERE
            unique_item_count >= 2
            AND recency_rank <= {max_items_per_session}
        ORDER BY session, event_index
    )
    TO '{output_sql}'
    (
        FORMAT PARQUET,
        COMPRESSION ZSTD
    )
    """

    started = time.perf_counter()

    logger.info(
        "session_item_cache_start",
        extra={
            "event": "session_item_cache_start",
            "stage": "session_item_cache",
            "split_ts": split_ts,
            "max_items_per_session": max_items_per_session,
        },
    )

    try:
        with Heartbeat(
            logger,
            stage="session_item_cache",
            interval_seconds=heartbeat_seconds,
        ):
            connection.execute(query)

        os.replace(temporary_output, destination)

        count_row = connection.execute(
            "SELECT count(*) "
            f"FROM read_parquet('{_sql_literal(destination)}')"
        ).fetchone()

        if count_row is None:
            raise RuntimeError(
                "DuckDB did not return session-item row count"
            )

        rows = int(count_row[0])

    finally:
        connection.close()
        temporary_output.unlink(missing_ok=True)

    if rows <= 0:
        raise RuntimeError(
            "session-item cache contains no rows"
        )

    manifest = SessionItemManifest(
        split_ts=split_ts,
        max_items_per_session=max_items_per_session,
        rows=rows,
        output_sha256=sha256_file(destination),
    )

    temp_manifest = output_manifest.with_suffix(
        output_manifest.suffix + ".tmp"
    )

    temp_manifest.write_text(
        json.dumps(
            asdict(manifest),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(temp_manifest, output_manifest)

    logger.info(
        "session_item_cache_complete",
        extra={
            "event": "session_item_cache_complete",
            "stage": "session_item_cache",
            "status": "passed",
            "rows": rows,
            "elapsed_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        },
    )

    return manifest
