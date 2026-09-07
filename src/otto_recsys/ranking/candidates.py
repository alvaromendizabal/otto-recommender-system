"""Materialize label-blind baseline candidates without rebuilding observed features.

DuckDB bounds source-generation memory; feature joins stream complete session
batches to Parquet. Only checksum-committed buckets may be reused.
"""
from __future__ import annotations

import json
import logging
import math
import re
import shutil
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import duckdb
import faiss
import polars as pl
import pyarrow.parquet as pq
from gensim.models import KeyedVectors  # type: ignore[import-untyped]

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import utc_now_iso
from otto_recsys.ranking.feature_cache import valid_part as valid_features
from otto_recsys.ranking.feature_cache import workspace_lock, write_json
from otto_recsys.ranking.features import FEATURE_COLUMNS, OBJECTIVES, candidate_features
from otto_recsys.retrieval.candidate_union import (
    append_item2vec_candidates,
    configure_connection,
    create_covisit_source_candidates,
    sql_literal,
)
from otto_recsys.runtime import Heartbeat

FAMILIES = (*OBJECTIVES, "queries")


class CandidateCheckpoints(Protocol):
    def restore(self, directory: Path, input_id: str) -> None: ...
    def publish_part(self, directory: Path, bucket: int, input_id: str) -> None: ...
    def publish_summary(self, directory: Path) -> None: ...


@dataclass(frozen=True)
class CandidateConfig:
    source_k: int = 1200
    item2vec_k: int = 800
    ef_search: int = 1024
    candidate_k: int = 100
    batch_sessions: int = 128
    threads: int = 4
    memory_limit: str = "4GB"

    def validate(self) -> None:
        for name in ("source_k", "item2vec_k", "ef_search", "candidate_k",
                     "batch_sessions", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not re.fullmatch(r"[1-9][0-9]*(MB|GB)", self.memory_limit):
            raise ValueError("memory_limit must be a positive MB or GB budget")
        if self.candidate_k > 10000:
            raise ValueError("candidate_k exceeds the ranker's complete-query budget")
        if self.batch_sessions * self.candidate_k > 100000:
            raise ValueError("reduce batch_sessions: at most 100000 rows/objective per join")


def valid_part(directory: Path, bucket: int, input_id: str) -> dict[str, Any] | None:
    root = directory / "parts" / f"part-{bucket:03d}"
    try:
        receipt = json.loads(root.with_suffix(".json").read_text())
        if (receipt["input_id"] != input_id or receipt["bucket"] != bucket
                or set(receipt["files"]) != set(FAMILIES)):
            return None
        if not math.isfinite(receipt["compute_seconds"]) or receipt["compute_seconds"] < 0:
            return None
        for family in FAMILIES:
            path = root / f"{family}.parquet"
            evidence = receipt["files"][family]
            if (evidence["sha256"] != sha256_file(path)
                    or evidence["bytes"] != path.stat().st_size
                    or evidence["rows"] != pq.read_metadata(path).num_rows):
                return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError):
        return None


def compress_sources(connection: Any, candidate_k: int) -> None:
    """Freeze membership before labels exist in the connection.

    This is a fixed RRF/source-agreement compression baseline, not a selected
    optimum and not the previously measured uncompressed coverage ceiling.
    """
    if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError("candidate_k must be a positive integer")
    connection.execute(f"""
        CREATE TEMP TABLE selected_candidates AS
        WITH evidence AS (
            SELECT session, objective, aid, count(DISTINCT source) AS agreements,
                   sum(1.0 / source_rank) AS rrf,
                   max(CASE WHEN source='item2vec' THEN score END) AS embedding_score
            FROM source_candidates GROUP BY session, objective, aid
        ), ranked AS (
            SELECT session, objective, aid,
                   row_number() OVER (PARTITION BY session, objective ORDER BY
                       agreements DESC, rrf DESC, embedding_score DESC NULLS LAST, aid) AS rank
            FROM evidence
        )
        SELECT session, objective, aid FROM ranked WHERE rank <= {candidate_k}
    """)


def write_candidate_parts(
    connection: Any, feature_part: Path, labels: pl.DataFrame, destination: Path,
    *, config: CandidateConfig,
) -> dict[str, Any]:
    """Stream small, complete session groups through the canonical feature join."""
    config.validate()
    sessions = pl.read_parquet(feature_part / "sessions.parquet").sort("session")
    items = pl.read_parquet(feature_part / "items.parquet")
    queries = pl.read_parquet(feature_part / "queries.parquet")
    if sessions.is_empty():
        raise ValueError("empty observed-feature bucket")
    expected = queries.select("session", "objective", "true_items").sort(["session", "objective"])
    actual = (
        queries.select("session", "objective")
        .join(labels.group_by("session", "objective").len(name="true_items"),
              on=["session", "objective"], how="left")
        .with_columns(pl.col("true_items").fill_null(0))
        .sort(["session", "objective"])
    )
    if actual.to_dicts() != expected.to_dicts():
        raise ValueError("full label counts disagree with the observed query ledger")
    invalid = connection.execute("""
        SELECT count(*) FROM source_candidates WHERE
            source IS NULL OR source NOT IN ('revisit','time','type','buy','item2vec')
            OR objective IS NULL OR objective NOT IN ('clicks','carts','orders')
            OR session IS NULL OR session < 0 OR aid IS NULL OR aid < 0
            OR source_rank IS NULL OR source_rank < 1 OR source_rank != floor(source_rank)
            OR score IS NULL OR NOT isfinite(score)
    """).fetchone()
    if invalid is None or invalid[0]:
        raise ValueError("invalid or unsupported baseline source candidate")
    connection.register("known_sessions", sessions.select("session").to_arrow())
    unknown = connection.execute("""
        SELECT count(*) FROM source_candidates c ANTI JOIN known_sessions s USING (session)
    """).fetchone()
    connection.unregister("known_sessions")
    if unknown is None or unknown[0]:
        raise ValueError("candidate session is absent from observed features")
    compress_sources(connection, config.candidate_k)
    destination.mkdir(parents=True, exist_ok=True)
    writers: dict[str, Any] = {}
    rows = dict.fromkeys(OBJECTIVES, 0)
    try:
        for subset in sessions.iter_slices(config.batch_sessions):
            ids = subset.select("session")
            connection.register("feature_sessions", ids.to_arrow())
            sources = pl.from_arrow(connection.execute("""
                SELECT c.source, c.objective, c.session, c.aid, c.score, c.source_rank
                FROM source_candidates c JOIN selected_candidates s
                USING (session, objective, aid)
                JOIN feature_sessions f USING (session)
            """).to_arrow_table())
            connection.unregister("feature_sessions")
            features = candidate_features(
                sources, subset, items.join(ids, on="session", how="semi"),
                labels.join(ids, on="session", how="semi"),
            ).with_columns(
                pl.col(FEATURE_COLUMNS).cast(pl.Float32),
                pl.col("session", "aid", "query_id").cast(pl.Int64),
            )
            features = features.join(
                subset.select("session", "fold", "inner_partition"), on="session"
            )
            for objective in OBJECTIVES:
                table = features.filter(pl.col("objective") == objective).to_arrow()
                if objective not in writers:
                    writers[objective] = pq.ParquetWriter(
                        destination / f"{objective}.parquet.tmp", table.schema, compression="zstd"
                    )
                writers[objective].write_table(table)
                rows[objective] += table.num_rows
    finally:
        for writer in writers.values():
            writer.close()
    for objective in OBJECTIVES:
        (destination / f"{objective}.parquet.tmp").replace(destination / f"{objective}.parquet")
    shutil.copyfile(feature_part / "queries.parquet", destination / "queries.parquet.tmp")
    (destination / "queries.parquet.tmp").replace(destination / "queries.parquet")
    return {
        family: {"sha256": sha256_file(destination / f"{family}.parquet"),
                 "bytes": (destination / f"{family}.parquet").stat().st_size,
                 "rows": pq.read_metadata(destination / f"{family}.parquet").num_rows}
        for family in FAMILIES
    }


def make_contract(
    cache: Path, observed: Path, covisit: Path, vectors: Path, index: Path,
    config: CandidateConfig,
) -> dict[str, Any]:
    config.validate()
    ranking = json.loads((cache / "manifest.json").read_text())
    feature_contract = json.loads((observed / "feature_contract.json").read_text())
    feature_id = canonical_json_sha256(feature_contract)
    if (feature_contract["training_cache_input_id"] != ranking["input_id"]
            or feature_contract["validation_manifest_id"] != ranking["validation_manifest_id"]):
        raise ValueError("observed features do not belong to this frozen training cache")
    files = {}
    for family in ("items", "labels", "examples"):
        digest = sha256_file(cache / f"{family}.parquet")
        if digest != ranking[f"{family}_sha256"]:
            raise ValueError(f"frozen {family} checksum mismatch")
        files[f"ranking/{family}"] = digest
    baseline = [(f"covisit/{name}", covisit / name) for name in
                ("time.parquet", "type.parquet", "buy.parquet",
                 "time.json", "type.json", "buy.json")]
    baseline += [(f"vectors/{path.name}", path)
                 for path in sorted(vectors.parent.glob(vectors.name + "*")) if path.is_file()]
    baseline += [("vectors/manifest", vectors.parent / "manifest.json"),
                 ("index/data", index), ("index/manifest", index.parent / "manifest.json")]
    for key, path in baseline:
        files[key] = sha256_file(path)
    embedding_manifest = json.loads((vectors.parent / "manifest.json").read_text())
    if embedding_manifest.get("validation_manifest_id") != ranking["validation_manifest_id"]:
        raise ValueError("Item2Vec and frozen ranking cache use different validation protocols")
    receipts = []
    for bucket in range(feature_contract["buckets"]):
        receipt = valid_features(observed, bucket, feature_id)
        if receipt is None:
            raise ValueError(
                f"observed bucket {bucket} is not checksum-committed; restore it first"
            )
        receipts.append(receipt["files"])
    return {
        "schema_version": 1, "config": asdict(config),
        "buckets": feature_contract["buckets"], "folds": feature_contract["folds"],
        "observed_feature_id": feature_id, "observed_parts": receipts,
        "input_sha256": files, "feature_names": list(FEATURE_COLUMNS),
        "code_sha256": {str(path.name): sha256_file(path) for path in
                        (Path(__file__), Path(__file__).with_name("features.py"),
                         Path(__file__).parents[1] / "retrieval/candidate_union.py")},
        "runtime": {name: version(name) for name in
                    ("polars", "pyarrow", "duckdb", "numpy", "faiss-cpu", "gensim")},
        "candidate_policy": "source agreement, reciprocal-rank sum, Item2Vec score, ascending aid",
        "positive_insertion": False, "negative_sampling": False,
        "neural_candidates": False, "untouched_temporal_holdout": False,
        "retriever_fit_provenance_certified": False,
        "validation_scope": ("exploratory nested session validation; "
                             "frozen baseline provenance uncertified"),
    }


def build_candidates(
    cache: Path, observed: Path, covisit: Path, vectors: Path, index: Path, output: Path,
    *, config: CandidateConfig, logger: logging.Logger,
    checkpoints: CandidateCheckpoints | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    progress = {"bucket": 0, "reused_buckets": 0}
    with workspace_lock(output), Heartbeat(
        logger, stage="ranking_candidates", interval_seconds=15, progress_provider=progress.copy
    ):
        contract = make_contract(cache, observed, covisit, vectors, index, config)
        input_id = canonical_json_sha256(contract)
        contract_path = output / "candidate_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError(
                "candidate contract mismatch; preserve this run and use a new directory"
            )
        write_json(contract_path, contract)
        if checkpoints is not None:
            checkpoints.restore(output, input_id)
        receipts = []
        embedding = search_index = None
        for bucket in range(contract["buckets"]):
            progress["bucket"] = bucket
            receipt = valid_part(output, bucket, input_id)
            if receipt is not None:
                progress["reused_buckets"] += 1
                logger.info("candidate_bucket_reused", extra={"bucket": bucket})
            else:
                part_started = time.perf_counter()
                if embedding is None:
                    embedding = KeyedVectors.load(str(vectors), mmap="r")
                    search_index = faiss.read_index(str(index))
                    faiss.omp_set_num_threads(config.threads)
                temporary = output / "temporary" / f"bucket-{bucket:03d}"
                with duckdb.connect() as connection:
                    configure_connection(connection, threads=config.threads,
                                         memory_limit=config.memory_limit, temp_directory=temporary)
                    connection.execute(f"""
                        CREATE TEMP TABLE vitems AS
                        SELECT session, aid, ts, event_type, event_index, recency_rank
                        FROM read_parquet('{sql_literal(cache / 'items.parquet')}')
                        WHERE bucket = {bucket}
                    """)
                    # No labels or vlabels table exist during candidate generation/compression.
                    create_covisit_source_candidates(connection, covisit_dir=covisit,
                                                     source_k=config.source_k)
                    append_item2vec_candidates(
                        connection, items_path=cache / "items.parquet", vectors=embedding,
                        index=search_index, bucket=bucket, ann_k=config.item2vec_k,
                        ef_search=config.ef_search,
                    )
                    labels = pl.scan_parquet(cache / "labels.parquet").filter(
                        pl.col("bucket") == bucket
                    ).collect()
                    files = write_candidate_parts(
                        connection, observed / "parts" / f"part-{bucket:03d}", labels,
                        output / "parts" / f"part-{bucket:03d}", config=config,
                    )
                shutil.rmtree(temporary, ignore_errors=True)
                receipt = {"input_id": input_id, "bucket": bucket, "files": files,
                           "compute_seconds": time.perf_counter() - part_started,
                           "completed_at_utc": utc_now_iso()}
                write_json(output / "parts" / f"part-{bucket:03d}.json", receipt)
                logger.info("candidate_bucket_complete", extra={
                    "bucket": bucket, "elapsed_seconds": round(receipt["compute_seconds"], 3)
                })
            if checkpoints is not None:
                checkpoints.publish_part(output, bucket, input_id)
            receipts.append(receipt)
        summary = {
            "status": "passed", "input_id": input_id, "completed_buckets": len(receipts),
            "reused_buckets": progress["reused_buckets"], "completed_at_utc": utc_now_iso(),
            "attempt_elapsed_seconds": time.perf_counter() - started,
            "retained_compute_seconds": sum(part["compute_seconds"] for part in receipts),
            "rows": {family: sum(part["files"][family]["rows"] for part in receipts)
                     for family in FAMILIES},
            "candidate_k": config.candidate_k, "ranking_evaluation": "not yet measured",
            "validation_scope": contract["validation_scope"],
        }
        write_json(output / "manifest.json", summary)
        if checkpoints is not None:
            checkpoints.publish_summary(output)
        logger.info("ranking_candidates_complete", extra={
            "elapsed_seconds": round(summary["attempt_elapsed_seconds"], 3),
            "total_elapsed_seconds": round(summary["retained_compute_seconds"], 3),
        })
        return summary
