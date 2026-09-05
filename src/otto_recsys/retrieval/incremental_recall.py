from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import faiss
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import canonical_json_sha256
from otto_recsys.retrieval.candidate_union import (
    append_item2vec_candidates,
    candidate_count_rows,
    configure_connection,
    create_covisit_source_candidates,
    create_validation_tables,
)
from otto_recsys.runtime import Heartbeat

_OBJECTIVES = ("clicks", "carts", "orders")
_METRIC_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}
HitRow = tuple[
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]


@dataclass(frozen=True)
class IncrementalRecallConfig:
    buckets: int
    source_k: int
    ann_k: int
    ef_search: int
    threads: int
    memory_limit: str


@dataclass(frozen=True)
class IncrementalRecallResult:
    input_id: str
    config: IncrementalRecallConfig
    completed_buckets: int
    sessions: int
    elapsed_seconds: float
    metrics: dict[str, float]
    counts: dict[str, int]
    candidate_stats: dict[str, float]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _input_id(
    *,
    validation_manifest: Path,
    covisit_dir: Path,
    item2vec_manifest: Path,
    faiss_manifest: Path,
    config: IncrementalRecallConfig,
) -> str:
    payload = {
        "validation": _load_json(validation_manifest),
        "time": _load_json(covisit_dir / "time.json"),
        "type": _load_json(covisit_dir / "type.json"),
        "buy": _load_json(covisit_dir / "buy.json"),
        "item2vec": _load_json(item2vec_manifest),
        "faiss": _load_json(faiss_manifest),
        "config": asdict(config),
    }
    return canonical_json_sha256(payload)


def _empty_state(input_id: str) -> dict[str, Any]:
    return {
        "input_id": input_id,
        "completed_buckets": [],
        "sessions": 0,
        "elapsed_seconds": 0.0,
        "hit_counts": {},
        "candidate_counts": {},
        "status": "running",
    }


def _load_state(path: Path, input_id: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_state(input_id)

    state = _load_json(path)
    if state.get("input_id") != input_id:
        raise RuntimeError(
            "existing incremental-recall state does not match current inputs"
        )
    return state


def label_hit_rows(
    connection: duckdb.DuckDBPyConnection,
) -> list[HitRow]:
    """Aggregate exact label coverage and source-exclusive recovery."""
    rows = connection.execute(
        """
        WITH per_label AS (
            SELECT
                l.objective,
                l.session,
                l.aid,
                max(CASE WHEN c.source = 'revisit' THEN 1 ELSE 0 END) AS revisit,
                max(CASE WHEN c.source = 'time' THEN 1 ELSE 0 END) AS time,
                max(CASE WHEN c.source = 'type' THEN 1 ELSE 0 END) AS type,
                max(CASE WHEN c.source = 'buy' THEN 1 ELSE 0 END) AS buy,
                max(CASE WHEN c.source = 'item2vec' THEN 1 ELSE 0 END) AS item2vec
            FROM vlabels AS l
            LEFT JOIN source_candidates AS c
              ON c.session = l.session
             AND c.objective = l.objective
             AND c.aid = l.aid
            GROUP BY l.objective, l.session, l.aid
        ),
        flags AS (
            SELECT
                *,
                CASE WHEN revisit + time + type + buy > 0 THEN 1 ELSE 0 END
                    AS covisit,
                CASE WHEN revisit + time + type + buy + item2vec > 0
                    THEN 1 ELSE 0 END AS any_source
            FROM per_label
        )
        SELECT
            objective,
            count(*) AS labels,
            sum(revisit) AS revisit_hits,
            sum(time) AS time_hits,
            sum(type) AS type_hits,
            sum(buy) AS buy_hits,
            sum(item2vec) AS item2vec_hits,
            sum(covisit) AS covisit_hits,
            sum(any_source) AS union_hits,
            sum(CASE WHEN covisit = 1 AND item2vec = 1 THEN 1 ELSE 0 END)
                AS shared_hits,
            sum(CASE WHEN covisit = 1 AND item2vec = 0 THEN 1 ELSE 0 END)
                AS covisit_only_hits,
            sum(CASE WHEN covisit = 0 AND item2vec = 1 THEN 1 ELSE 0 END)
                AS item2vec_only_hits,
            sum(CASE WHEN any_source = 0 THEN 1 ELSE 0 END) AS missed,
            sum(CASE WHEN revisit = 1 AND time + type + buy + item2vec = 0
                THEN 1 ELSE 0 END) AS revisit_unique,
            sum(CASE WHEN time = 1 AND revisit + type + buy + item2vec = 0
                THEN 1 ELSE 0 END) AS time_unique,
            sum(CASE WHEN type = 1 AND revisit + time + buy + item2vec = 0
                THEN 1 ELSE 0 END) AS type_unique,
            sum(CASE WHEN buy = 1 AND revisit + time + type + item2vec = 0
                THEN 1 ELSE 0 END) AS buy_unique,
            sum(CASE WHEN item2vec = 1 AND revisit + time + type + buy = 0
                THEN 1 ELSE 0 END) AS item2vec_unique
        FROM flags
        GROUP BY objective
        ORDER BY objective
        """
    ).fetchall()

    return [
        (
            str(row[0]),
            int(row[1]),
            int(row[2]),
            int(row[3]),
            int(row[4]),
            int(row[5]),
            int(row[6]),
            int(row[7]),
            int(row[8]),
            int(row[9]),
            int(row[10]),
            int(row[11]),
            int(row[12]),
            int(row[13]),
            int(row[14]),
            int(row[15]),
            int(row[16]),
            int(row[17]),
        )
        for row in rows
    ]


def _accumulate_hit_rows(
    state: dict[str, Any],
    rows: list[HitRow],
) -> None:
    hit_counts = state.setdefault("hit_counts", {})
    names = (
        "labels",
        "revisit_hits",
        "time_hits",
        "type_hits",
        "buy_hits",
        "item2vec_hits",
        "covisit_hits",
        "union_hits",
        "shared_hits",
        "covisit_only_hits",
        "item2vec_only_hits",
        "missed",
        "revisit_unique",
        "time_unique",
        "type_unique",
        "buy_unique",
        "item2vec_unique",
    )

    for row in rows:
        objective = row[0]
        bucket_counts = dict(zip(names, row[1:], strict=True))
        objective_counts = hit_counts.setdefault(
            objective,
            {name: 0 for name in names},
        )
        for name, value in bucket_counts.items():
            objective_counts[name] += int(value)


def _accumulate_candidate_rows(
    state: dict[str, Any],
    rows: list[tuple[str, int, int, int]],
) -> None:
    candidate_counts = state.setdefault("candidate_counts", {})
    for objective, covisit_rows, item2vec_rows, union_rows in rows:
        raw = candidate_counts.setdefault(
            objective,
            {
                "sessions": 0,
                "covisit_rows": 0,
                "item2vec_rows": 0,
                "union_rows": 0,
            },
        )
        # Sessions are all validation sessions in the bucket for all objectives.
        # The caller sets the exact session count after this aggregation.
        raw["covisit_rows"] += covisit_rows
        raw["item2vec_rows"] += item2vec_rows
        raw["union_rows"] += union_rows


def _finalize_result(
    state: dict[str, Any],
    config: IncrementalRecallConfig,
) -> IncrementalRecallResult:
    hit_counts: dict[str, dict[str, int]] = state["hit_counts"]
    candidate_counts: dict[str, dict[str, int]] = state["candidate_counts"]

    metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    candidate_stats: dict[str, float] = {}

    for objective in _OBJECTIVES:
        raw = hit_counts[objective]
        labels = raw["labels"]
        if labels <= 0:
            raise RuntimeError(f"no labels found for objective {objective}")

        for name, value in raw.items():
            counts[f"{objective}.{name}"] = int(value)

        for source in ("revisit", "time", "type", "buy", "item2vec"):
            metrics[f"{objective}.{source}_recall_ceiling"] = (
                raw[f"{source}_hits"] / labels
            )

        metrics[f"{objective}.covisit_recall_ceiling"] = (
            raw["covisit_hits"] / labels
        )
        metrics[f"{objective}.union_recall_ceiling"] = raw["union_hits"] / labels
        metrics[f"{objective}.item2vec_marginal_recall"] = (
            raw["item2vec_only_hits"] / labels
        )
        metrics[f"{objective}.remaining_miss_rate"] = raw["missed"] / labels

        candidates = candidate_counts[objective]
        sessions = candidates["sessions"]
        if sessions <= 0:
            raise RuntimeError(f"no candidate sessions found for {objective}")
        candidate_stats[f"{objective}.covisit_average_candidates"] = (
            candidates["covisit_rows"] / sessions
        )
        candidate_stats[f"{objective}.item2vec_average_candidates"] = (
            candidates["item2vec_rows"] / sessions
        )
        candidate_stats[f"{objective}.union_average_candidates"] = (
            candidates["union_rows"] / sessions
        )

    for source in ("revisit", "time", "type", "buy"):
        metrics[f"weighted_{source}_recall_ceiling"] = sum(
            _METRIC_WEIGHTS[objective]
            * metrics[f"{objective}.{source}_recall_ceiling"]
            for objective in _OBJECTIVES
        )

    for family in ("covisit", "item2vec", "union"):
        metrics[f"weighted_{family}_recall_ceiling"] = sum(
            _METRIC_WEIGHTS[objective]
            * metrics[f"{objective}.{family}_recall_ceiling"]
            for objective in _OBJECTIVES
        )

    metrics["weighted_item2vec_marginal_recall"] = sum(
        _METRIC_WEIGHTS[objective]
        * metrics[f"{objective}.item2vec_marginal_recall"]
        for objective in _OBJECTIVES
    )

    return IncrementalRecallResult(
        input_id=str(state["input_id"]),
        config=config,
        completed_buckets=len(state["completed_buckets"]),
        sessions=int(state["sessions"]),
        elapsed_seconds=float(state["elapsed_seconds"]),
        metrics=metrics,
        counts=counts,
        candidate_stats=candidate_stats,
    )


def evaluate_incremental_recall(
    validation_cache_dir: str | Path,
    covisit_dir: str | Path,
    vectors_path: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    source_k: int = 1200,
    ann_k: int = 200,
    ef_search: int = 256,
    threads: int = 4,
    memory_limit: str = "8GB",
    temp_root: str | Path = "data/interim/duckdb_incremental_recall",
    heartbeat_seconds: float = 30.0,
) -> IncrementalRecallResult:
    """Measure exact Item2Vec marginal recall beyond co-visitation candidates."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")
    if source_k <= 0:
        raise ValueError("source_k must be positive")
    if ann_k <= 0:
        raise ValueError("ann_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")

    config = IncrementalRecallConfig(
        buckets=buckets,
        source_k=source_k,
        ann_k=ann_k,
        ef_search=ef_search,
        threads=threads,
        memory_limit=memory_limit,
    )

    validation_root = Path(validation_cache_dir).resolve()
    graph_root = Path(covisit_dir).resolve()
    vectors_file = Path(vectors_path).resolve()
    index_file = Path(index_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    items_path = validation_root / "items.parquet"
    labels_path = validation_root / "labels.parquet"
    validation_manifest = validation_root / "manifest.json"
    item2vec_manifest = vectors_file.parent / "manifest.json"
    faiss_manifest = index_file.parent / "manifest.json"

    for path in (
        items_path,
        labels_path,
        validation_manifest,
        graph_root / "time.parquet",
        graph_root / "type.parquet",
        graph_root / "buy.parquet",
        graph_root / "time.json",
        graph_root / "type.json",
        graph_root / "buy.json",
        vectors_file,
        item2vec_manifest,
        index_file,
        faiss_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    input_id = _input_id(
        validation_manifest=validation_manifest,
        covisit_dir=graph_root,
        item2vec_manifest=item2vec_manifest,
        faiss_manifest=faiss_manifest,
        config=config,
    )

    state_path = destination / "state.json"
    metrics_path = destination / "metrics.json"
    state = _load_state(state_path, input_id)
    completed = {int(value) for value in state["completed_buckets"]}

    vectors = KeyedVectors.load(str(vectors_file), mmap="r")
    index = faiss.read_index(str(index_file))
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", ef_search)

    progress: dict[str, int] = {
        "bucket": len(completed),
        "buckets": buckets,
        "sessions": int(state["sessions"]),
    }
    started = time.perf_counter()

    logger.info(
        "incremental_recall_start",
        extra={
            "event": "incremental_recall_start",
            "stage": "incremental_recall",
            "completed_buckets": len(completed),
            "buckets": buckets,
            "source_k": source_k,
            "ann_k": ann_k,
            "ef_search": ef_search,
            "input_id": input_id,
        },
    )

    with Heartbeat(
        logger,
        stage="incremental_recall",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for bucket in range(buckets):
            if bucket in completed:
                logger.info(
                    "incremental_bucket_skip",
                    extra={
                        "event": "incremental_bucket_skip",
                        "stage": "incremental_recall",
                        "bucket": bucket,
                        "status": "already_complete",
                    },
                )
                continue

            bucket_started = time.perf_counter()
            logger.info(
                "incremental_bucket_start",
                extra={
                    "event": "incremental_bucket_start",
                    "stage": "incremental_recall",
                    "bucket": bucket,
                    "buckets": buckets,
                },
            )

            temp_directory = Path(temp_root).resolve() / f"bucket_{bucket:03d}"
            connection = duckdb.connect(database=":memory:")
            try:
                configure_connection(
                    connection,
                    threads=threads,
                    memory_limit=memory_limit,
                    temp_directory=temp_directory,
                )
                sessions = create_validation_tables(
                    connection,
                    items_path=items_path,
                    labels_path=labels_path,
                    bucket=bucket,
                )
                create_covisit_source_candidates(
                    connection,
                    covisit_dir=graph_root,
                    source_k=source_k,
                )
                item2vec_rows = append_item2vec_candidates(
                    connection,
                    items_path=items_path,
                    vectors=vectors,
                    index=index,
                    bucket=bucket,
                    ann_k=ann_k,
                    ef_search=ef_search,
                )

                hits = label_hit_rows(connection)
                candidate_rows = candidate_count_rows(connection)
            finally:
                connection.close()

            _accumulate_hit_rows(state, hits)
            _accumulate_candidate_rows(state, candidate_rows)
            for objective in _OBJECTIVES:
                state["candidate_counts"].setdefault(
                    objective,
                    {
                        "sessions": 0,
                        "covisit_rows": 0,
                        "item2vec_rows": 0,
                        "union_rows": 0,
                    },
                )["sessions"] += sessions

            state["sessions"] = int(state["sessions"]) + sessions
            state["completed_buckets"].append(bucket)
            state["completed_buckets"] = sorted(
                {int(value) for value in state["completed_buckets"]}
            )
            state["elapsed_seconds"] = round(
                float(state["elapsed_seconds"])
                + (time.perf_counter() - bucket_started),
                3,
            )
            _write_json_atomic(state, state_path)

            progress["bucket"] = len(state["completed_buckets"])
            progress["sessions"] = int(state["sessions"])

            logger.info(
                "incremental_bucket_complete",
                extra={
                    "event": "incremental_bucket_complete",
                    "stage": "incremental_recall",
                    "bucket": bucket,
                    "buckets": buckets,
                    "sessions": sessions,
                    "events": item2vec_rows,
                    "elapsed_seconds": round(
                        time.perf_counter() - bucket_started,
                        3,
                    ),
                },
            )

    if len(state["completed_buckets"]) != buckets:
        raise RuntimeError("not all incremental-recall buckets completed")

    state["status"] = "completed"
    _write_json_atomic(state, state_path)
    result = _finalize_result(state, config)
    _write_json_atomic(asdict(result), metrics_path)

    logger.info(
        "incremental_recall_complete",
        extra={
            "event": "incremental_recall_complete",
            "stage": "incremental_recall",
            "status": "passed",
            "sessions": result.sessions,
            "elapsed_seconds": result.elapsed_seconds,
            "weighted_covisit_recall_ceiling": result.metrics[
                "weighted_covisit_recall_ceiling"
            ],
            "weighted_item2vec_recall_ceiling": result.metrics[
                "weighted_item2vec_recall_ceiling"
            ],
            "weighted_union_recall_ceiling": result.metrics[
                "weighted_union_recall_ceiling"
            ],
            "weighted_item2vec_marginal_recall": result.metrics[
                "weighted_item2vec_marginal_recall"
            ],
            "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return result
