from __future__ import annotations

import json
import logging
import os
import shutil
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
_COVISIT_SOURCES = ("revisit", "time", "type", "buy")
HistogramRow = tuple[str, int, int | None, int]


@dataclass(frozen=True)
class CandidateBudgetConfig:
    buckets: int
    source_k: int
    item2vec_quotas: tuple[int, ...]
    capture_fraction: float
    ef_search: int
    threads: int
    memory_limit: str


@dataclass(frozen=True)
class CandidateBudgetResult:
    input_id: str
    config: CandidateBudgetConfig
    completed_buckets: int
    sessions: int
    elapsed_seconds: float
    recommended_item2vec_k: dict[str, int]
    metrics: dict[str, float]
    candidate_stats: dict[str, float]


def normalize_quotas(values: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return a sorted, unique Item2Vec quota grid that always includes zero."""
    quotas = tuple(sorted({int(value) for value in values}))
    if not quotas:
        raise ValueError("at least one Item2Vec quota is required")
    if quotas[0] < 0:
        raise ValueError("Item2Vec quotas cannot be negative")
    if not any(value > 0 for value in quotas):
        raise ValueError("at least one positive Item2Vec quota is required")
    return quotas if quotas[0] == 0 else (0, *quotas)


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
    config: CandidateBudgetConfig,
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
        "label_histogram": {},
        "candidate_histogram": {},
        "status": "running",
    }


def _load_state(path: Path, input_id: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_state(input_id)

    state = _load_json(path)
    if state.get("input_id") != input_id:
        raise RuntimeError(
            "existing candidate-budget state does not match current inputs"
        )
    return state


def coverage_histogram_rows(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[list[HistogramRow], list[HistogramRow]]:
    """Build compact rank histograms for labels and unique candidate aids."""
    source_sql = ", ".join(f"'{source}'" for source in _COVISIT_SOURCES)

    label_rows = connection.execute(
        f"""
        WITH per_label AS (
            SELECT
                l.objective,
                l.session,
                l.aid,
                max(
                    CASE WHEN c.source IN ({source_sql}) THEN 1 ELSE 0 END
                ) AS covisit,
                min(
                    CASE WHEN c.source = 'item2vec' THEN c.source_rank END
                ) AS item2vec_rank
            FROM vlabels AS l
            LEFT JOIN source_candidates AS c
              ON c.objective = l.objective
             AND c.session = l.session
             AND c.aid = l.aid
            GROUP BY l.objective, l.session, l.aid
        )
        SELECT
            objective,
            covisit,
            item2vec_rank,
            count(*) AS rows
        FROM per_label
        GROUP BY objective, covisit, item2vec_rank
        ORDER BY objective, covisit DESC, item2vec_rank NULLS LAST
        """
    ).fetchall()

    candidate_rows = connection.execute(
        f"""
        WITH per_candidate AS (
            SELECT
                objective,
                session,
                aid,
                max(
                    CASE WHEN source IN ({source_sql}) THEN 1 ELSE 0 END
                ) AS covisit,
                min(
                    CASE WHEN source = 'item2vec' THEN source_rank END
                ) AS item2vec_rank
            FROM source_candidates
            GROUP BY objective, session, aid
        )
        SELECT
            objective,
            covisit,
            item2vec_rank,
            count(*) AS rows
        FROM per_candidate
        GROUP BY objective, covisit, item2vec_rank
        ORDER BY objective, covisit DESC, item2vec_rank NULLS LAST
        """
    ).fetchall()

    def convert(rows: list[tuple[Any, ...]]) -> list[HistogramRow]:
        return [
            (
                str(row[0]),
                int(row[1]),
                None if row[2] is None else int(row[2]),
                int(row[3]),
            )
            for row in rows
        ]

    return convert(label_rows), convert(candidate_rows)


def _histogram_key(row: HistogramRow) -> str:
    objective, covisit, rank, _ = row
    rank_text = "none" if rank is None else str(rank)
    return f"{objective}|{covisit}|{rank_text}"


def _accumulate_histogram(
    state: dict[str, Any],
    field: str,
    rows: list[HistogramRow],
) -> None:
    histogram = state.setdefault(field, {})
    for row in rows:
        key = _histogram_key(row)
        histogram[key] = int(histogram.get(key, 0)) + row[3]


def _decode_histogram(payload: dict[str, Any]) -> list[HistogramRow]:
    rows: list[HistogramRow] = []
    for key, raw_count in payload.items():
        objective, covisit_text, rank_text = key.split("|", maxsplit=2)
        rank = None if rank_text == "none" else int(rank_text)
        rows.append((objective, int(covisit_text), rank, int(raw_count)))
    return rows


def quota_summary(
    label_histogram: list[HistogramRow],
    candidate_histogram: list[HistogramRow],
    *,
    sessions: int,
    quotas: tuple[int, ...],
) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate exact candidate-set recall and size at every Item2Vec quota."""
    if sessions <= 0:
        raise ValueError("sessions must be positive")

    metrics: dict[str, float] = {}
    candidate_stats: dict[str, float] = {}

    for objective in _OBJECTIVES:
        label_rows = [row for row in label_histogram if row[0] == objective]
        candidate_rows = [
            row for row in candidate_histogram if row[0] == objective
        ]
        labels = sum(row[3] for row in label_rows)
        if labels <= 0:
            raise RuntimeError(f"no labels found for objective {objective}")

        covisit_hits = sum(row[3] for row in label_rows if row[1] == 1)
        covisit_candidates = sum(
            row[3] for row in candidate_rows if row[1] == 1
        )
        metrics[f"{objective}.covisit_recall_ceiling"] = covisit_hits / labels
        candidate_stats[f"{objective}.covisit_average_candidates"] = (
            covisit_candidates / sessions
        )

        for quota in quotas:
            item2vec_hits = sum(
                row[3]
                for row in label_rows
                if row[2] is not None and row[2] <= quota
            )
            item2vec_only_hits = sum(
                row[3]
                for row in label_rows
                if row[1] == 0
                and row[2] is not None
                and row[2] <= quota
            )
            union_hits = covisit_hits + item2vec_only_hits

            item2vec_candidates = sum(
                row[3]
                for row in candidate_rows
                if row[2] is not None and row[2] <= quota
            )
            item2vec_only_candidates = sum(
                row[3]
                for row in candidate_rows
                if row[1] == 0
                and row[2] is not None
                and row[2] <= quota
            )
            union_candidates = covisit_candidates + item2vec_only_candidates

            metrics[f"{objective}.item2vec_recall_{quota}"] = (
                item2vec_hits / labels
            )
            metrics[f"{objective}.item2vec_marginal_recall_{quota}"] = (
                item2vec_only_hits / labels
            )
            metrics[f"{objective}.union_recall_{quota}"] = union_hits / labels

            item2vec_average = item2vec_candidates / sessions
            union_average = union_candidates / sessions
            added_average = item2vec_only_candidates / sessions
            candidate_stats[f"{objective}.item2vec_average_candidates_{quota}"] = (
                item2vec_average
            )
            candidate_stats[f"{objective}.union_average_candidates_{quota}"] = (
                union_average
            )
            candidate_stats[f"{objective}.added_average_candidates_{quota}"] = (
                added_average
            )

            marginal = metrics[
                f"{objective}.item2vec_marginal_recall_{quota}"
            ]
            efficiency = 0.0 if added_average == 0 else 100.0 * marginal / added_average
            metrics[
                f"{objective}.marginal_recall_per_100_added_candidates_{quota}"
            ] = efficiency

    for quota in quotas:
        metrics[f"weighted_union_recall_{quota}"] = sum(
            _METRIC_WEIGHTS[objective]
            * metrics[f"{objective}.union_recall_{quota}"]
            for objective in _OBJECTIVES
        )
        metrics[f"weighted_item2vec_marginal_recall_{quota}"] = sum(
            _METRIC_WEIGHTS[objective]
            * metrics[f"{objective}.item2vec_marginal_recall_{quota}"]
            for objective in _OBJECTIVES
        )

    metrics["weighted_covisit_recall_ceiling"] = sum(
        _METRIC_WEIGHTS[objective]
        * metrics[f"{objective}.covisit_recall_ceiling"]
        for objective in _OBJECTIVES
    )
    return metrics, candidate_stats


def recommend_quotas(
    metrics: dict[str, float],
    *,
    quotas: tuple[int, ...],
    capture_fraction: float,
) -> dict[str, int]:
    """Select the smallest quota capturing a target fraction of max marginal recall."""
    if not 0 < capture_fraction <= 1:
        raise ValueError("capture_fraction must be in (0, 1]")

    largest_quota = max(quotas)
    recommended: dict[str, int] = {}
    for objective in _OBJECTIVES:
        maximum = metrics[
            f"{objective}.item2vec_marginal_recall_{largest_quota}"
        ]
        if maximum <= 0:
            recommended[objective] = 0
            continue

        threshold = capture_fraction * maximum
        eligible = [
            quota
            for quota in quotas
            if metrics[f"{objective}.item2vec_marginal_recall_{quota}"]
            >= threshold
        ]
        if not eligible:
            raise RuntimeError(f"no eligible quota found for {objective}")
        recommended[objective] = min(eligible)
    return recommended


def _finalize_result(
    state: dict[str, Any],
    config: CandidateBudgetConfig,
) -> CandidateBudgetResult:
    label_histogram = _decode_histogram(state["label_histogram"])
    candidate_histogram = _decode_histogram(state["candidate_histogram"])
    metrics, candidate_stats = quota_summary(
        label_histogram,
        candidate_histogram,
        sessions=int(state["sessions"]),
        quotas=config.item2vec_quotas,
    )
    recommended = recommend_quotas(
        metrics,
        quotas=config.item2vec_quotas,
        capture_fraction=config.capture_fraction,
    )

    metrics["recommended_weighted_union_recall_ceiling"] = sum(
        _METRIC_WEIGHTS[objective]
        * metrics[f"{objective}.union_recall_{recommended[objective]}"]
        for objective in _OBJECTIVES
    )

    for objective, quota in recommended.items():
        candidate_stats[f"{objective}.recommended_union_average_candidates"] = (
            candidate_stats[f"{objective}.union_average_candidates_{quota}"]
        )
        candidate_stats[f"{objective}.recommended_added_average_candidates"] = (
            candidate_stats[f"{objective}.added_average_candidates_{quota}"]
        )

    return CandidateBudgetResult(
        input_id=str(state["input_id"]),
        config=config,
        completed_buckets=len(state["completed_buckets"]),
        sessions=int(state["sessions"]),
        elapsed_seconds=float(state["elapsed_seconds"]),
        recommended_item2vec_k=recommended,
        metrics=metrics,
        candidate_stats=candidate_stats,
    )


def evaluate_candidate_budget(
    validation_cache_dir: str | Path,
    covisit_dir: str | Path,
    vectors_path: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    source_k: int = 1200,
    item2vec_quotas: tuple[int, ...] = (0, 10, 20, 50, 100, 150, 200),
    capture_fraction: float = 0.95,
    ef_search: int = 256,
    threads: int = 4,
    memory_limit: str = "8GB",
    temp_root: str | Path = "data/interim/duckdb_candidate_budget",
    heartbeat_seconds: float = 30.0,
) -> CandidateBudgetResult:
    """Measure the Item2Vec recall/candidate Pareto curve beyond co-visitation."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")
    if source_k <= 0:
        raise ValueError("source_k must be positive")
    if ef_search <= 0:
        raise ValueError("ef_search must be positive")
    if not 0 < capture_fraction <= 1:
        raise ValueError("capture_fraction must be in (0, 1]")

    quotas = normalize_quotas(item2vec_quotas)
    max_ann_k = max(quotas)
    config = CandidateBudgetConfig(
        buckets=buckets,
        source_k=source_k,
        item2vec_quotas=quotas,
        capture_fraction=capture_fraction,
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
        "candidate_budget_start",
        extra={
            "event": "candidate_budget_start",
            "stage": "candidate_budget",
            "completed_buckets": len(completed),
            "buckets": buckets,
            "source_k": source_k,
            "max_ann_k": max_ann_k,
            "capture_fraction": capture_fraction,
            "input_id": input_id,
        },
    )

    with Heartbeat(
        logger,
        stage="candidate_budget",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for bucket in range(buckets):
            if bucket in completed:
                logger.info(
                    "candidate_budget_bucket_skip",
                    extra={
                        "event": "candidate_budget_bucket_skip",
                        "stage": "candidate_budget",
                        "bucket": bucket,
                        "status": "already_complete",
                    },
                )
                continue

            bucket_started = time.perf_counter()
            logger.info(
                "candidate_budget_bucket_start",
                extra={
                    "event": "candidate_budget_bucket_start",
                    "stage": "candidate_budget",
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
                    ann_k=max_ann_k,
                    ef_search=ef_search,
                )
                label_rows, candidate_rows = coverage_histogram_rows(connection)
            finally:
                connection.close()
                shutil.rmtree(temp_directory, ignore_errors=True)

            _accumulate_histogram(state, "label_histogram", label_rows)
            _accumulate_histogram(state, "candidate_histogram", candidate_rows)
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
                "candidate_budget_bucket_complete",
                extra={
                    "event": "candidate_budget_bucket_complete",
                    "stage": "candidate_budget",
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
        raise RuntimeError("not all candidate-budget buckets completed")

    state["status"] = "completed"
    _write_json_atomic(state, state_path)
    result = _finalize_result(state, config)
    _write_json_atomic(asdict(result), metrics_path)

    logger.info(
        "candidate_budget_complete",
        extra={
            "event": "candidate_budget_complete",
            "stage": "candidate_budget",
            "status": "passed",
            "sessions": result.sessions,
            "elapsed_seconds": result.elapsed_seconds,
            "recommended_clicks_k": result.recommended_item2vec_k["clicks"],
            "recommended_carts_k": result.recommended_item2vec_k["carts"],
            "recommended_orders_k": result.recommended_item2vec_k["orders"],
            "recommended_weighted_union_recall": result.metrics[
                "recommended_weighted_union_recall_ceiling"
            ],
            "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return result
