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

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.retrieval.candidate_union import (
    append_item2vec_candidates,
    configure_connection,
    create_covisit_source_candidates,
    create_validation_tables,
    sql_literal,
)
from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class HardNegativeConfig:
    buckets: int
    source_k: int
    item2vec_k: int
    hard_negatives: int
    ef_search: int
    threads: int
    memory_limit: str


@dataclass(frozen=True)
class HardNegativeManifest:
    validation_manifest_id: str
    input_id: str
    config: HardNegativeConfig
    completed_buckets: int
    sessions: int
    positive_rows: int
    output_rows: int
    part_files: tuple[str, ...]
    family_sha256: str
    elapsed_seconds: float


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


def _manifest_id(payload: dict[str, Any], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{source} must contain a 64-character {key}")
    return value


def _validate_provenance(
    *,
    training_manifest: Path,
    item2vec_manifest: Path,
) -> str:
    training = _load_json(training_manifest)
    item2vec = _load_json(item2vec_manifest)
    training_id = _manifest_id(
        training,
        "validation_manifest_id",
        "ranking training manifest",
    )
    item2vec_id = _manifest_id(
        item2vec,
        "validation_manifest_id",
        "Item2Vec manifest",
    )
    if training_id != item2vec_id:
        raise RuntimeError(
            "ranking training prefixes and Item2Vec artifacts do not share the "
            "same frozen validation protocol"
        )
    return training_id


def _input_id(
    *,
    training_manifest: Path,
    covisit_dir: Path,
    item2vec_manifest: Path,
    faiss_manifest: Path,
    config: HardNegativeConfig,
) -> str:
    payload = {
        "training_cache": _load_json(training_manifest),
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
        "positive_rows": 0,
        "output_rows": 0,
        "elapsed_seconds": 0.0,
        "status": "running",
    }


def _load_state(path: Path, input_id: str) -> dict[str, Any]:
    if not path.is_file():
        return _empty_state(input_id)
    state = _load_json(path)
    if state.get("input_id") != input_id:
        raise RuntimeError("existing hard-negative state does not match current inputs")
    return state


def create_hard_negative_training_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    hard_negatives: int,
) -> None:
    """Create one compact row per positive with false-negative-safe hard negatives."""
    if hard_negatives <= 0:
        raise ValueError("hard_negatives must be positive")

    connection.execute(
        """
        CREATE TEMP TABLE candidate_features AS
        WITH collapsed AS (
            SELECT
                objective,
                session,
                aid,
                max(CASE WHEN source = 'revisit' THEN 1 ELSE 0 END) AS revisit,
                max(CASE WHEN source = 'time' THEN 1 ELSE 0 END) AS time,
                max(CASE WHEN source = 'type' THEN 1 ELSE 0 END) AS type,
                max(CASE WHEN source = 'buy' THEN 1 ELSE 0 END) AS buy,
                max(CASE WHEN source = 'item2vec' THEN 1 ELSE 0 END) AS item2vec,
                min(CASE WHEN source = 'revisit' THEN source_rank END) AS revisit_rank,
                min(CASE WHEN source = 'time' THEN source_rank END) AS time_rank,
                min(CASE WHEN source = 'type' THEN source_rank END) AS type_rank,
                min(CASE WHEN source = 'buy' THEN source_rank END) AS buy_rank,
                min(CASE WHEN source = 'item2vec' THEN source_rank END)
                    AS item2vec_rank,
                max(CASE WHEN source = 'item2vec' THEN score END) AS item2vec_score,
                count(DISTINCT source) AS source_count
            FROM source_candidates
            GROUP BY objective, session, aid
        )
        SELECT
            *,
            coalesce(1.0 / revisit_rank, 0.0)
              + coalesce(1.0 / time_rank, 0.0)
              + coalesce(1.0 / type_rank, 0.0)
              + coalesce(1.0 / buy_rank, 0.0)
              + coalesce(1.0 / item2vec_rank, 0.0) AS reciprocal_rank_sum
        FROM collapsed
        """
    )

    connection.execute(
        f"""
        CREATE TEMP TABLE hard_negative_rows AS
        WITH negatives AS (
            SELECT c.*
            FROM candidate_features AS c
            WHERE NOT EXISTS (
                SELECT 1
                FROM vlabels AS l
                WHERE l.session = c.session
                  AND l.objective = c.objective
                  AND l.aid = c.aid
            )
        ),
        ranked AS (
            SELECT
                *,
                row_number() OVER (
                    PARTITION BY objective, session
                    ORDER BY
                        source_count DESC,
                        reciprocal_rank_sum DESC,
                        coalesce(item2vec_score, -1e30) DESC,
                        aid
                ) AS negative_rank
            FROM negatives
        ),
        negative_lists AS (
            SELECT
                objective,
                session,
                list(aid ORDER BY negative_rank) AS hard_negative_aids,
                list(source_count ORDER BY negative_rank)
                    AS hard_negative_source_counts,
                list(reciprocal_rank_sum ORDER BY negative_rank)
                    AS hard_negative_rrf_scores
            FROM ranked
            WHERE negative_rank <= {hard_negatives}
            GROUP BY objective, session
        ),
        positive_features AS (
            SELECT
                l.objective,
                l.session,
                f.fold,
                l.aid AS positive_aid,
                coalesce(c.revisit, 0) AS positive_revisit,
                coalesce(c.time, 0) AS positive_time,
                coalesce(c.type, 0) AS positive_type,
                coalesce(c.buy, 0) AS positive_buy,
                coalesce(c.item2vec, 0) AS positive_item2vec,
                coalesce(c.source_count, 0) AS positive_source_count,
                c.revisit_rank AS positive_revisit_rank,
                c.time_rank AS positive_time_rank,
                c.type_rank AS positive_type_rank,
                c.buy_rank AS positive_buy_rank,
                c.item2vec_rank AS positive_item2vec_rank,
                c.item2vec_score AS positive_item2vec_score
            FROM vlabels AS l
            INNER JOIN vfolds AS f
              ON f.session = l.session
            LEFT JOIN candidate_features AS c
              ON c.objective = l.objective
             AND c.session = l.session
             AND c.aid = l.aid
        )
        SELECT
            p.session,
            p.fold,
            p.objective,
            p.positive_aid,
            p.positive_revisit,
            p.positive_time,
            p.positive_type,
            p.positive_buy,
            p.positive_item2vec,
            p.positive_source_count,
            p.positive_revisit_rank,
            p.positive_time_rank,
            p.positive_type_rank,
            p.positive_buy_rank,
            p.positive_item2vec_rank,
            p.positive_item2vec_score,
            n.hard_negative_aids,
            n.hard_negative_source_counts,
            n.hard_negative_rrf_scores,
            list_count(n.hard_negative_aids) AS negative_count
        FROM positive_features AS p
        INNER JOIN negative_lists AS n
          ON n.objective = p.objective
         AND n.session = p.session
        """
    )


def hard_negative_contract_rows(
    connection: duckdb.DuckDBPyConnection,
) -> tuple[int, int, int, int]:
    row = connection.execute(
        """
        SELECT
            count(*) AS rows,
            count(DISTINCT session) AS sessions,
            sum(CASE WHEN negative_count <= 0 THEN 1 ELSE 0 END) AS empty_negatives
        FROM hard_negative_rows
        """
    ).fetchone()
    if row is None:
        raise RuntimeError("DuckDB did not return hard-negative contract counts")

    false_negative_row = connection.execute(
        """
        SELECT count(*)
        FROM (
            SELECT DISTINCT
                r.session,
                r.objective,
                r.positive_aid
            FROM hard_negative_rows AS r,
                 unnest(r.hard_negative_aids) AS t(aid)
            INNER JOIN vlabels AS l
              ON l.session = r.session
             AND l.objective = r.objective
             AND l.aid = t.aid
        )
        """
    ).fetchone()
    if false_negative_row is None:
        raise RuntimeError("DuckDB did not return false-negative contract count")

    return (
        int(row[0] or 0),
        int(row[1] or 0),
        int(row[2] or 0),
        int(false_negative_row[0] or 0),
    )


def _family_hash(paths: list[Path]) -> str:
    payload = [
        {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths)
    ]
    return canonical_json_sha256(payload)


def mine_hard_negatives(
    training_cache_dir: str | Path,
    covisit_dir: str | Path,
    vectors_path: str | Path,
    index_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    source_k: int = 1200,
    item2vec_k: int = 800,
    hard_negatives: int = 64,
    ef_search: int = 1024,
    threads: int = 4,
    memory_limit: str = "8GB",
    temp_root: str | Path = "data/interim/duckdb_hard_negatives",
    heartbeat_seconds: float = 30.0,
) -> HardNegativeManifest:
    """Mine deterministic retrieval-hard negatives from frozen validation prefixes."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")
    if source_k <= 0 or item2vec_k <= 0 or hard_negatives <= 0:
        raise ValueError("candidate and negative depths must be positive")
    if item2vec_k < hard_negatives:
        raise ValueError("item2vec_k must be >= hard_negatives")
    if ef_search < item2vec_k:
        raise ValueError("ef_search must be >= item2vec_k")

    cache_root = Path(training_cache_dir).resolve()
    graph_root = Path(covisit_dir).resolve()
    vectors_file = Path(vectors_path).resolve()
    index_file = Path(index_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    items_path = cache_root / "items.parquet"
    labels_path = cache_root / "labels.parquet"
    examples_path = cache_root / "examples.parquet"
    training_manifest = cache_root / "manifest.json"
    item2vec_manifest = vectors_file.parent / "manifest.json"
    faiss_manifest = index_file.parent / "manifest.json"

    required = (
        items_path,
        labels_path,
        examples_path,
        training_manifest,
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
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    validation_id = _validate_provenance(
        training_manifest=training_manifest,
        item2vec_manifest=item2vec_manifest,
    )
    config = HardNegativeConfig(
        buckets=buckets,
        source_k=source_k,
        item2vec_k=item2vec_k,
        hard_negatives=hard_negatives,
        ef_search=ef_search,
        threads=threads,
        memory_limit=memory_limit,
    )
    input_id = _input_id(
        training_manifest=training_manifest,
        covisit_dir=graph_root,
        item2vec_manifest=item2vec_manifest,
        faiss_manifest=faiss_manifest,
        config=config,
    )

    state_path = destination / "state.json"
    manifest_path = destination / "manifest.json"
    parts_dir = destination / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state = _load_state(state_path, input_id)
    completed = {int(value) for value in state["completed_buckets"]}

    vectors = KeyedVectors.load(str(vectors_file), mmap="r")
    index = faiss.read_index(str(index_file))
    faiss.ParameterSpace().set_index_parameter(index, "efSearch", ef_search)

    progress: dict[str, int] = {
        "bucket": len(completed),
        "buckets": buckets,
        "sessions": int(state["sessions"]),
        "events": int(state["output_rows"]),
    }
    started = time.perf_counter()

    logger.info(
        "hard_negative_mining_start",
        extra={
            "event": "hard_negative_mining_start",
            "stage": "hard_negative_mining",
            "input_id": input_id,
            "validation_manifest_id": validation_id,
            "completed_buckets": len(completed),
            "item2vec_k": item2vec_k,
            "hard_negatives": hard_negatives,
        },
    )

    with Heartbeat(
        logger,
        stage="hard_negative_mining",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress.copy,
    ):
        for bucket in range(buckets):
            part_path = parts_dir / f"part-{bucket:03d}.parquet"
            if bucket in completed:
                if not part_path.is_file():
                    raise RuntimeError(
                        f"completed bucket {bucket} is missing {part_path}"
                    )
                logger.info(
                    "hard_negative_bucket_skip",
                    extra={
                        "event": "hard_negative_bucket_skip",
                        "stage": "hard_negative_mining",
                        "bucket": bucket,
                        "status": "already_complete",
                    },
                )
                continue

            bucket_started = time.perf_counter()
            logger.info(
                "hard_negative_bucket_start",
                extra={
                    "event": "hard_negative_bucket_start",
                    "stage": "hard_negative_mining",
                    "bucket": bucket,
                    "buckets": buckets,
                },
            )

            temp_directory = Path(temp_root).resolve() / f"bucket_{bucket:03d}"
            part_temp = parts_dir / f".part-{bucket:03d}.parquet.tmp"
            part_temp.unlink(missing_ok=True)
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
                examples_sql = sql_literal(examples_path)
                connection.execute(
                    f"""
                    CREATE TEMP TABLE vfolds AS
                    SELECT session, fold
                    FROM read_parquet('{examples_sql}')
                    WHERE bucket = {bucket}
                    """
                )
                create_covisit_source_candidates(
                    connection,
                    covisit_dir=graph_root,
                    source_k=source_k,
                )
                append_item2vec_candidates(
                    connection,
                    items_path=items_path,
                    vectors=vectors,
                    index=index,
                    bucket=bucket,
                    ann_k=item2vec_k,
                    ef_search=ef_search,
                )
                create_hard_negative_training_rows(
                    connection,
                    hard_negatives=hard_negatives,
                )
                positive_row = connection.execute(
                    "SELECT count(*) FROM vlabels"
                ).fetchone()
                if positive_row is None:
                    raise RuntimeError("DuckDB did not return positive-row count")
                positive_rows = int(positive_row[0])

                rows, output_sessions, empty_negatives, false_negative_rows = (
                    hard_negative_contract_rows(connection)
                )
                if rows <= 0 or output_sessions <= 0:
                    raise RuntimeError("hard-negative bucket produced no rows")
                if rows != positive_rows:
                    raise RuntimeError(
                        "hard-negative mining did not retain every positive label"
                    )
                if empty_negatives != 0:
                    raise RuntimeError("hard-negative bucket contains empty groups")
                if false_negative_rows != 0:
                    raise RuntimeError("future positives leaked into hard negatives")

                output_sql = sql_literal(part_temp)
                connection.execute(
                    f"""
                    COPY (
                        SELECT *
                        FROM hard_negative_rows
                        ORDER BY fold, session, objective, positive_aid
                    ) TO '{output_sql}' (
                        FORMAT PARQUET,
                        COMPRESSION ZSTD,
                        ROW_GROUP_SIZE 100000
                    )
                    """
                )
            finally:
                connection.close()
                shutil.rmtree(temp_directory, ignore_errors=True)

            if not part_temp.is_file() or part_temp.stat().st_size <= 0:
                raise RuntimeError(f"hard-negative part was not written: {part_temp}")
            os.replace(part_temp, part_path)

            state["sessions"] = int(state["sessions"]) + sessions
            state["positive_rows"] = int(state["positive_rows"]) + positive_rows
            state["output_rows"] = int(state["output_rows"]) + rows
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
            progress["events"] = int(state["output_rows"])
            logger.info(
                "hard_negative_bucket_complete",
                extra={
                    "event": "hard_negative_bucket_complete",
                    "stage": "hard_negative_mining",
                    "bucket": bucket,
                    "buckets": buckets,
                    "sessions": sessions,
                    "events": rows,
                    "elapsed_seconds": round(time.perf_counter() - bucket_started, 3),
                },
            )

    if len(state["completed_buckets"]) != buckets:
        raise RuntimeError("not all hard-negative buckets completed")

    part_paths = [parts_dir / f"part-{bucket:03d}.parquet" for bucket in range(buckets)]
    for path in part_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    state["status"] = "completed"
    _write_json_atomic(state, state_path)
    manifest = HardNegativeManifest(
        validation_manifest_id=validation_id,
        input_id=input_id,
        config=config,
        completed_buckets=buckets,
        sessions=int(state["sessions"]),
        positive_rows=int(state["positive_rows"]),
        output_rows=int(state["output_rows"]),
        part_files=tuple(path.name for path in part_paths),
        family_sha256=_family_hash(part_paths),
        elapsed_seconds=float(state["elapsed_seconds"]),
    )
    _write_json_atomic(asdict(manifest), manifest_path)

    logger.info(
        "hard_negative_mining_complete",
        extra={
            "event": "hard_negative_mining_complete",
            "stage": "hard_negative_mining",
            "status": "passed",
            "sessions": manifest.sessions,
            "events": manifest.output_rows,
            "elapsed_seconds": manifest.elapsed_seconds,
            "wall_elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )
    return manifest
