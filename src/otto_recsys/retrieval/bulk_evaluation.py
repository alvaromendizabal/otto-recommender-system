from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb

from otto_recsys.experiments.manifest import canonical_json_sha256
from otto_recsys.runtime import Heartbeat

_MEMORY_RE = re.compile(r"^[1-9][0-9]*(MB|GB)$")
_OBJECTIVES = ("clicks", "carts", "orders")
_ABLATIONS = (
    "revisit",
    "revisit_time",
    "revisit_time_type",
    "full_covisit",
)
_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


@dataclass(frozen=True)
class EvaluationConfig:
    buckets: int
    ks: tuple[int, ...]
    rrf_k: float
    threads: int
    memory_limit: str


@dataclass(frozen=True)
class RetrievalEvaluationResult:
    input_id: str
    config: EvaluationConfig
    completed_buckets: int
    elapsed_seconds: float
    metrics: dict[str, float]
    incremental_hits: dict[str, int]
    candidate_stats: dict[str, float]


def _literal(value: str | Path) -> str:
    return str(value).replace("'", "''")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _input_id(
    validation_manifest: Path,
    covisit_dir: Path,
    config: EvaluationConfig,
) -> str:
    payload = {
        "validation": _load_json(validation_manifest),
        "time": _load_json(covisit_dir / "time.json"),
        "type": _load_json(covisit_dir / "type.json"),
        "buy": _load_json(covisit_dir / "buy.json"),
        "config": asdict(config),
    }
    return canonical_json_sha256(payload)


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
        raise ValueError("invalid DuckDB memory limit")

    temp_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET threads = {threads}")
    connection.execute(f"SET memory_limit = '{memory_limit}'")
    connection.execute("SET preserve_insertion_order = false")
    connection.execute(
        "SET temp_directory = "
        f"'{_literal(temp_directory)}'"
    )


def _create_validation_tables(
    connection: duckdb.DuckDBPyConnection,
    *,
    items_path: Path,
    labels_path: Path,
    bucket: int,
) -> int:
    items_sql = _literal(items_path)
    labels_sql = _literal(labels_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE vitems AS
        SELECT session, aid, ts, event_type, event_index, recency_rank
        FROM read_parquet('{items_sql}')
        WHERE bucket = {bucket}
        """
    )
    connection.execute(
        f"""
        CREATE TEMP TABLE vlabels AS
        SELECT session, objective, aid
        FROM read_parquet('{labels_sql}')
        WHERE bucket = {bucket}
        """
    )

    row = connection.execute(
        "SELECT count(DISTINCT session) FROM vitems"
    ).fetchone()
    if row is None:
        raise RuntimeError("DuckDB did not return validation-session count")
    return int(row[0])


def _create_revisit_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    max_k: int,
) -> None:
    connection.execute(
        f"""
        CREATE TEMP TABLE revisit_candidates AS
        WITH scored AS (
            SELECT
                session,
                aid,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 2.0
                    ELSE 2.5
                END / (1.0 + 0.10 * (recency_rank - 1)) AS clicks_score,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 4.0
                    ELSE 3.0
                END / (1.0 + 0.10 * (recency_rank - 1)) AS carts_score,
                CASE event_type
                    WHEN 0 THEN 1.0
                    WHEN 1 THEN 3.0
                    ELSE 5.0
                END / (1.0 + 0.10 * (recency_rank - 1)) AS orders_score
            FROM vitems
        ),
        long_form AS (
            SELECT 'clicks' AS objective, session, aid, clicks_score AS score
            FROM scored
            UNION ALL
            SELECT 'carts', session, aid, carts_score
            FROM scored
            UNION ALL
            SELECT 'orders', session, aid, orders_score
            FROM scored
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM long_form
        )
        SELECT
            'revisit' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {max_k}
        """
    )


def _create_time_candidates(
    connection: duckdb.DuckDBPyConnection,
    matrix_path: Path,
    *,
    max_k: int,
) -> None:
    matrix_sql = _literal(matrix_path)

    connection.execute(
        f"""
        CREATE TEMP TABLE time_candidates AS
        WITH scored AS (
            SELECT
                v.session,
                m.target_aid AS aid,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 2.0
                        ELSE 2.5
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS clicks_score,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 4.0
                        ELSE 3.0
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS carts_score,
                sum(
                    m.score
                    * CASE v.event_type
                        WHEN 0 THEN 1.0
                        WHEN 1 THEN 3.0
                        ELSE 5.0
                    END
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS orders_score
            FROM vitems AS v
            INNER JOIN read_parquet('{matrix_sql}') AS m
                ON v.aid = m.source_aid
            WHERE m.objective = 'all'
            GROUP BY v.session, m.target_aid
        ),
        long_form AS (
            SELECT 'clicks' AS objective, session, aid, clicks_score AS score
            FROM scored
            UNION ALL
            SELECT 'carts', session, aid, carts_score
            FROM scored
            UNION ALL
            SELECT 'orders', session, aid, orders_score
            FROM scored
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM long_form
        )
        SELECT
            'time' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {max_k}
        """
    )


def _create_objective_matrix_candidates(
    connection: duckdb.DuckDBPyConnection,
    matrix_path: Path,
    *,
    source_name: str,
    objectives: Sequence[str],
    max_k: int,
) -> None:
    matrix_sql = _literal(matrix_path)
    objective_sql = ", ".join(f"'{value}'" for value in objectives)

    connection.execute(
        f"""
        CREATE TEMP TABLE {source_name}_candidates AS
        WITH scored AS (
            SELECT
                m.objective,
                v.session,
                m.target_aid AS aid,
                sum(
                    m.score
                    / (1.0 + 0.10 * (v.recency_rank - 1))
                ) AS score
            FROM vitems AS v
            INNER JOIN read_parquet('{matrix_sql}') AS m
                ON v.aid = m.source_aid
            WHERE m.objective IN ({objective_sql})
            GROUP BY m.objective, v.session, m.target_aid
        ),
        ranked AS (
            SELECT
                objective,
                session,
                aid,
                score,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY score DESC, aid
                ) AS source_rank
            FROM scored
        )
        SELECT
            '{source_name}' AS source,
            objective,
            session,
            aid,
            score,
            source_rank
        FROM ranked
        WHERE source_rank <= {max_k}
        """
    )


def _create_source_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    covisit_dir: Path,
    max_k: int,
) -> None:
    _create_revisit_candidates(connection, max_k=max_k)
    _create_time_candidates(
        connection,
        covisit_dir / "time.parquet",
        max_k=max_k,
    )
    _create_objective_matrix_candidates(
        connection,
        covisit_dir / "type.parquet",
        source_name="type",
        objectives=_OBJECTIVES,
        max_k=max_k,
    )
    _create_objective_matrix_candidates(
        connection,
        covisit_dir / "buy.parquet",
        source_name="buy",
        objectives=("carts", "orders"),
        max_k=max_k,
    )

    connection.execute(
        """
        CREATE TEMP TABLE source_candidates AS
        SELECT * FROM revisit_candidates
        UNION ALL
        SELECT * FROM time_candidates
        UNION ALL
        SELECT * FROM type_candidates
        UNION ALL
        SELECT * FROM buy_candidates
        """
    )


def _create_fused_candidates(
    connection: duckdb.DuckDBPyConnection,
    *,
    ablation: str,
    sources: Sequence[str],
    max_k: int,
    rrf_k: float,
) -> None:
    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive")
    if not sources:
        raise ValueError("sources must not be empty")

    allowed = {"revisit", "time", "type", "buy"}
    if any(source not in allowed for source in sources):
        raise ValueError("unsupported retrieval source")

    source_sql = ", ".join(f"'{source}'" for source in sources)

    connection.execute("DROP TABLE IF EXISTS ranked_candidates")
    connection.execute(
        f"""
        CREATE TEMP TABLE ranked_candidates AS
        WITH fused AS (
            SELECT
                objective,
                session,
                aid,
                sum(1.0 / ({rrf_k} + source_rank)) AS fused_score,
                count(*) AS source_count
            FROM source_candidates
            WHERE source IN ({source_sql})
            GROUP BY objective, session, aid
        ),
        ranked AS (
            SELECT
                '{ablation}' AS ablation,
                objective,
                session,
                aid,
                fused_score,
                source_count,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY fused_score DESC, source_count DESC, aid
                ) AS final_rank
            FROM fused
        )
        SELECT *
        FROM ranked
        WHERE final_rank <= {max_k}
        """
    )


def _metric_rows(
    connection: duckdb.DuckDBPyConnection,
    ks: Sequence[int],
    *,
    ablation: str,
) -> list[tuple[str, str, int, int, int, int, int]]:
    values = ", ".join(f"({k})" for k in ks)

    rows = connection.execute(
        f"""
        WITH ks(k) AS (VALUES {values}),
        objectives(objective) AS (
            VALUES ('clicks'), ('carts'), ('orders')
        ),
        label_counts AS (
            SELECT
                objective,
                session,
                count(*) AS label_count
            FROM vlabels
            GROUP BY objective, session
        ),
        denominators AS (
            SELECT
                o.objective,
                k.k,
                coalesce(sum(least(k.k, lc.label_count)), 0) AS denominator
            FROM objectives AS o
            CROSS JOIN ks AS k
            LEFT JOIN label_counts AS lc
                ON lc.objective = o.objective
            GROUP BY o.objective, k.k
        ),
        hits AS (
            SELECT
                r.objective,
                k.k,
                count(*) AS hits
            FROM ranked_candidates AS r
            INNER JOIN vlabels AS l
                ON r.session = l.session
               AND r.objective = l.objective
               AND r.aid = l.aid
            CROSS JOIN ks AS k
            WHERE r.final_rank <= k.k
            GROUP BY r.objective, k.k
        ),
        per_session_candidates AS (
            SELECT
                objective,
                session,
                max(final_rank) AS candidate_count
            FROM ranked_candidates
            GROUP BY objective, session
        ),
        candidate_summary AS (
            SELECT
                objective,
                count(*) AS sessions_with_candidates,
                sum(candidate_count) AS candidate_rows
            FROM per_session_candidates
            GROUP BY objective
        )
        SELECT
            '{ablation}' AS ablation,
            o.objective,
            k.k,
            coalesce(h.hits, 0) AS hits,
            d.denominator,
            coalesce(c.sessions_with_candidates, 0) AS sessions_with_candidates,
            coalesce(c.candidate_rows, 0) AS candidate_rows
        FROM objectives AS o
        CROSS JOIN ks AS k
        INNER JOIN denominators AS d
            ON d.objective = o.objective
           AND d.k = k.k
        LEFT JOIN hits AS h
            ON h.objective = o.objective
           AND h.k = k.k
        LEFT JOIN candidate_summary AS c
            ON c.objective = o.objective
        ORDER BY o.objective, k.k
        """
    ).fetchall()

    return [
        (
            str(row[0]),
            str(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[6]),
        )
        for row in rows
    ]

def _empty_state(input_id: str) -> dict[str, Any]:
    return {
        "input_id": input_id,
        "completed_buckets": [],
        "counts": {},
        "candidate_counts": {},
        "elapsed_seconds": 0.0,
        "status": "running",
    }


def _load_state(path: Path, input_id: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_state(input_id)

    state = _load_json(path)
    if state.get("input_id") != input_id:
        raise RuntimeError(
            "existing evaluation state does not match current inputs"
        )
    return state


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _accumulate_rows(
    state: dict[str, Any],
    rows: Sequence[tuple[str, str, int, int, int, int, int]],
    *,
    max_k: int,
) -> None:
    counts = state.setdefault("counts", {})
    candidate_counts = state.setdefault("candidate_counts", {})

    for (
        ablation,
        objective,
        k,
        hits,
        denominator,
        sessions_with_candidates,
        candidate_rows,
    ) in rows:
        metric_key = f"{ablation}|{objective}|{k}"
        metric = counts.setdefault(metric_key, {"hits": 0, "denominator": 0})
        metric["hits"] += hits
        metric["denominator"] += denominator

        if k == max_k:
            candidate_key = f"{ablation}|{objective}"
            candidate = candidate_counts.setdefault(
                candidate_key,
                {"sessions": 0, "rows": 0},
            )
            candidate["sessions"] += sessions_with_candidates
            candidate["rows"] += candidate_rows


def _finalize_result(
    state: dict[str, Any],
    config: EvaluationConfig,
) -> RetrievalEvaluationResult:
    counts: dict[str, dict[str, int]] = state["counts"]
    candidate_counts: dict[str, dict[str, int]] = state["candidate_counts"]

    metrics: dict[str, float] = {}
    incremental_hits: dict[str, int] = {}
    candidate_stats: dict[str, float] = {}

    for ablation in _ABLATIONS:
        for objective in _OBJECTIVES:
            for k in config.ks:
                key = f"{ablation}|{objective}|{k}"
                raw = counts[key]
                denominator = raw["denominator"]
                recall = raw["hits"] / denominator if denominator else 0.0
                metrics[f"{ablation}.{objective}.recall_{k}"] = recall

    for ablation in _ABLATIONS:
        for k in config.ks:
            weighted = sum(
                _WEIGHTS[objective]
                * metrics[f"{ablation}.{objective}.recall_{k}"]
                for objective in _OBJECTIVES
            )
            metrics[f"{ablation}.weighted_recall_{k}"] = weighted

    previous: str | None = None
    for ablation in _ABLATIONS:
        if previous is None:
            previous = ablation
            continue

        for objective in _OBJECTIVES:
            for k in config.ks:
                current_key = f"{ablation}|{objective}|{k}"
                previous_key = f"{previous}|{objective}|{k}"
                delta = (
                    counts[current_key]["hits"]
                    - counts[previous_key]["hits"]
                )
                incremental_hits[
                    f"{previous}->{ablation}.{objective}.hits_{k}"
                ] = delta

        previous = ablation

    for key, raw in candidate_counts.items():
        sessions = raw["sessions"]
        average = raw["rows"] / sessions if sessions else 0.0
        candidate_stats[f"{key}.average_candidates"] = average
        candidate_stats[f"{key}.sessions_with_candidates"] = float(sessions)

    return RetrievalEvaluationResult(
        input_id=str(state["input_id"]),
        config=config,
        completed_buckets=len(state["completed_buckets"]),
        elapsed_seconds=float(state["elapsed_seconds"]),
        metrics=metrics,
        incremental_hits=incremental_hits,
        candidate_stats=candidate_stats,
    )


def evaluate_covisit_retrieval(
    validation_cache_dir: str | Path,
    covisit_dir: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    ks: Sequence[int] = (20, 50, 100, 200, 500, 1200),
    rrf_k: float = 60.0,
    threads: int = 4,
    memory_limit: str = "8GB",
    temp_root: str | Path = "data/interim/duckdb_evaluation",
    heartbeat_seconds: float = 30.0,
) -> RetrievalEvaluationResult:
    """Evaluate co-visitation ablations with bounded, resumable DuckDB joins."""
    normalized_ks = tuple(sorted(set(int(k) for k in ks)))
    if not normalized_ks or any(k <= 0 for k in normalized_ks):
        raise ValueError("ks must contain positive integers")

    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")

    config = EvaluationConfig(
        buckets=buckets,
        ks=normalized_ks,
        rrf_k=rrf_k,
        threads=threads,
        memory_limit=memory_limit,
    )

    validation_root = Path(validation_cache_dir).resolve()
    graph_root = Path(covisit_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    validation_manifest = validation_root / "manifest.json"
    items_path = validation_root / "items.parquet"
    labels_path = validation_root / "labels.parquet"

    for path in (validation_manifest, items_path, labels_path):
        if not path.exists():
            raise FileNotFoundError(path)

    validation_payload = _load_json(validation_manifest)
    if validation_payload.get("buckets") != buckets:
        raise RuntimeError(
            "validation-cache bucket count does not match evaluation buckets"
        )

    for name in ("time", "type", "buy"):
        for suffix in (".json", ".parquet"):
            path = graph_root / f"{name}{suffix}"
            if not path.is_file():
                raise FileNotFoundError(path)

    input_id = _input_id(validation_manifest, graph_root, config)
    state_path = destination / "state.json"
    metrics_path = destination / "metrics.json"
    state = _load_state(state_path, input_id)

    completed = {int(bucket) for bucket in state["completed_buckets"]}
    max_k = max(normalized_ks)
    run_started = time.perf_counter()

    logger.info(
        "retrieval_evaluation_start",
        extra={
            "event": "retrieval_evaluation_start",
            "stage": "retrieval_evaluation",
            "input_id": input_id,
            "buckets": buckets,
            "completed_buckets": len(completed),
        },
    )

    for bucket in range(buckets):
        if bucket in completed:
            continue

        bucket_started = time.perf_counter()
        bucket_temp = Path(temp_root).resolve() / f"bucket-{bucket:03d}"
        shutil.rmtree(bucket_temp, ignore_errors=True)
        bucket_temp.mkdir(parents=True, exist_ok=True)

        connection = duckdb.connect(database=":memory:")
        _configure_connection(
            connection,
            threads=threads,
            memory_limit=memory_limit,
            temp_directory=bucket_temp,
        )

        logger.info(
            "retrieval_bucket_start",
            extra={
                "event": "retrieval_bucket_start",
                "stage": "retrieval_evaluation",
                "bucket": bucket,
                "buckets": buckets,
            },
        )

        try:
            sessions = _create_validation_tables(
                connection,
                items_path=items_path,
                labels_path=labels_path,
                bucket=bucket,
            )

            progress: dict[str, int | str] = {
                "bucket": bucket,
                "buckets": buckets,
                "sessions": sessions,
            }

            with Heartbeat(
                logger,
                stage="retrieval_sources",
                interval_seconds=heartbeat_seconds,
                progress_provider=progress.copy,
            ):
                _create_source_candidates(
                    connection,
                    covisit_dir=graph_root,
                    max_k=max_k,
                )

            ablation_sources = {
                "revisit": ("revisit",),
                "revisit_time": ("revisit", "time"),
                "revisit_time_type": ("revisit", "time", "type"),
                "full_covisit": ("revisit", "time", "type", "buy"),
            }

            bucket_rows: list[tuple[str, str, int, int, int, int, int]] = []

            for ablation, sources in ablation_sources.items():
                progress["ablation"] = ablation

                with Heartbeat(
                    logger,
                    stage=f"retrieval_{ablation}",
                    interval_seconds=heartbeat_seconds,
                    progress_provider=progress.copy,
                ):
                    _create_fused_candidates(
                        connection,
                        ablation=ablation,
                        sources=sources,
                        max_k=max_k,
                        rrf_k=rrf_k,
                    )
                    bucket_rows.extend(
                        _metric_rows(
                            connection,
                            normalized_ks,
                            ablation=ablation,
                        )
                    )

            _accumulate_rows(state, bucket_rows, max_k=max_k)
            state["completed_buckets"].append(bucket)

            bucket_elapsed = time.perf_counter() - bucket_started
            state["elapsed_seconds"] = round(
                float(state["elapsed_seconds"]) + bucket_elapsed,
                3,
            )
            _write_json_atomic(state, state_path)

            logger.info(
                "retrieval_bucket_complete",
                extra={
                    "event": "retrieval_bucket_complete",
                    "stage": "retrieval_evaluation",
                    "bucket": bucket,
                    "sessions": sessions,
                    "elapsed_seconds": round(bucket_elapsed, 3),
                    "completed_buckets": len(state["completed_buckets"]),
                },
            )

        finally:
            connection.close()
            shutil.rmtree(bucket_temp, ignore_errors=True)

    state["status"] = "complete"
    _write_json_atomic(state, state_path)

    result = _finalize_result(state, config)
    _write_json_atomic(
        {
            "input_id": result.input_id,
            "config": asdict(result.config),
            "completed_buckets": result.completed_buckets,
            "elapsed_seconds": result.elapsed_seconds,
            "metrics": result.metrics,
            "incremental_hits": result.incremental_hits,
            "candidate_stats": result.candidate_stats,
        },
        metrics_path,
    )

    logger.info(
        "retrieval_evaluation_complete",
        extra={
            "event": "retrieval_evaluation_complete",
            "stage": "retrieval_evaluation",
            "status": "passed",
            "completed_buckets": result.completed_buckets,
            "elapsed_seconds": result.elapsed_seconds,
            "wall_seconds": round(time.perf_counter() - run_started, 3),
            "full_weighted_recall_20": result.metrics[
                "full_covisit.weighted_recall_20"
            ],
        },
    )

    return result
