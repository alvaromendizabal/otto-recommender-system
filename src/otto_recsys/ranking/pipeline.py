"""Run objective-specific LambdaRank fits on the frozen candidate cache.

Results remain exploratory: independent ranker selection does not retroactively
certify the fit provenance of frozen upstream retrieval artifacts.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import polars as pl
import psutil

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import utc_now_iso
from otto_recsys.ranking.candidates import valid_part
from otto_recsys.ranking.feature_cache import workspace_lock, write_json
from otto_recsys.ranking.features import OBJECTIVES
from otto_recsys.ranking.lambdarank import (
    QueryBatch,
    RankerConfig,
    aggregate_official,
    evaluate,
    fit_ranker,
)
from otto_recsys.runtime import Heartbeat


class ModelCheckpoints(Protocol):
    def restore(self, directory: Path, input_id: str) -> None: ...
    def publish(self, directory: Path) -> None: ...


def role_filter(role: str, outer_fold: int) -> pl.Expr:
    if role == "outer":
        return pl.col("fold") == outer_fold
    if role == "inner":
        return (pl.col("fold") != outer_fold) & (pl.col("inner_partition") == 0)
    if role == "fit":
        return (pl.col("fold") != outer_fold) & (pl.col("inner_partition") != 0)
    raise ValueError("unsupported split role")


def load_batch(
    paths: list[Path], queries: pl.DataFrame, feature_names: tuple[str, ...],
    *, role: str, outer_fold: int,
) -> QueryBatch:
    rows = pl.scan_parquet(paths).filter(role_filter(role, outer_fold)).select(
        "session", "aid", "target", *feature_names
    ).collect()
    truth = queries.filter(role_filter(role, outer_fold))
    return QueryBatch.create(
        rows.select(feature_names).to_numpy(), rows["session"].to_numpy(),
        rows["aid"].to_numpy(), rows["target"].to_numpy(),
        {int(session): int(count) for session, count in
         truth.select("session", "true_items").iter_rows()}, feature_names,
    )


def baseline_scores(batch: QueryBatch) -> np.ndarray:
    """Matched source-agreement/RRF baseline on exactly the same candidate pool."""
    columns = {name: index for index, name in enumerate(batch.feature_names)}
    rrf = batch.features[:, columns["reciprocal_rank_sum"]]
    agreements = batch.features[:, columns["source_count"]]
    embedding = np.nan_to_num(batch.features[:, columns["item2vec_score"]], nan=-np.inf)
    order = np.lexsort((batch.aid, -embedding, -rrf, -agreements, batch.session))
    scores = np.empty(batch.session.size, dtype=np.float64)
    scores[order] = -np.arange(batch.session.size, dtype=np.float64)
    return scores


def pool_metrics(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        raise ValueError("no evaluation parts")
    totals: dict[str, Any] = {key: sum(part[key] for part in parts) for key in
              ("hits", "denominator", "labeled_queries", "all_queries", "candidate_rows")}
    denominator = totals["denominator"]
    labeled = totals["labeled_queries"]
    totals["recall_at_20"] = totals["hits"] / denominator if denominator else None
    for key in ("ndcg_at_20", "mrr_at_20", "hit_rate_at_20"):
        totals[key] = (sum((part[key] or 0.0) * part["labeled_queries"] for part in parts)
                       / labeled if labeled else None)
    totals["candidate_ceiling_at_20"] = (
        sum((part["candidate_ceiling_at_20"] or 0.0) * part["denominator"] for part in parts)
        / denominator if denominator else None
    )
    return totals


def training_memory_guard(rows: int, features: int, max_memory_gib: float) -> float:
    """Reject oversized in-memory fits before allocating; never silently sample."""
    if (not math.isfinite(max_memory_gib) or max_memory_gib <= 0
            or rows < 0 or features < 1):
        raise ValueError("invalid training-memory budget")
    # Raw features, immutable/sorted copies, IDs, and native dataset headroom.
    estimated = rows * (features * 4 * 3 + 64)
    available = min(max_memory_gib * 2**30, psutil.virtual_memory().available * 0.85)
    if estimated > available:
        raise MemoryError(
            f"fit/inner estimate {estimated / 2**30:.2f} GiB exceeds budget "
            f"{available / 2**30:.2f} GiB; use a larger existing CPU workspace or "
            "a separately versioned smaller candidate_k. No rows were silently sampled."
        )
    return estimated / 2**30


def _completed_objective(directory: Path, identity: str) -> dict[str, Any] | None:
    try:
        receipt = json.loads((directory / "evaluation_receipt.json").read_text())
        if receipt["input_id"] != identity:
            return None
        for name, digest in receipt["files"].items():
            if name not in {"evaluation.json", "model.txt"}:
                return None
            if sha256_file(directory / name) != digest:
                return None
        if set(receipt["files"]) != {"evaluation.json", "model.txt"}:
            return None
        return json.loads((directory / "evaluation.json").read_text())
    except (OSError, ValueError, TypeError, KeyError):
        return None


def run_ranking(
    candidates: Path, output: Path, *, outer_folds: tuple[int, ...],
    config: RankerConfig, logger: logging.Logger, max_memory_gib: float = 20,
    checkpoints: ModelCheckpoints | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    config.validate()
    progress: dict[str, Any] = {"stage": "verification", "completed_objectives": 0}
    with workspace_lock(output), Heartbeat(
        logger, stage="ranking_total", interval_seconds=15, progress_provider=progress.copy
    ):
        candidate_contract = json.loads((candidates / "candidate_contract.json").read_text())
        if candidate_contract.get("neural_candidates") is not False:
            raise ValueError("this baseline runner does not accept uncertified neural candidates")
        candidate_id = canonical_json_sha256(candidate_contract)
        if (not outer_folds or len(set(outer_folds)) != len(outer_folds)
                or any(isinstance(fold, bool) or not isinstance(fold, int)
                       or not 0 <= fold < candidate_contract["folds"] for fold in outer_folds)):
            raise ValueError("outer folds must be unique valid integers")
        receipts = []
        for bucket in range(candidate_contract["buckets"]):
            receipt = valid_part(candidates, bucket, candidate_id)
            if receipt is None:
                raise ValueError(f"candidate bucket {bucket} is missing or corrupt")
            receipts.append(receipt)
        names = tuple(name for name in candidate_contract["feature_names"]
                      if not name.startswith("two_tower_"))
        contract = {
            "schema_version": 1, "candidate_id": candidate_id,
            "candidate_parts": [receipt["files"] for receipt in receipts],
            "outer_folds": list(outer_folds), "config": asdict(config),
            "feature_names": list(names), "source_sha256": sha256_file(Path(__file__)),
            "ranker_sha256": sha256_file(Path(__file__).with_name("lambdarank.py")),
            "runtime": {name: version(name) for name in ("lightgbm", "numpy", "polars")},
            "validation_scope": candidate_contract["validation_scope"],
            "untouched_temporal_holdout": False, "neural_candidates": False,
        }
        run_id = canonical_json_sha256(contract)
        path = output / "run_contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("ranking run contract mismatch; select a new output directory")
        write_json(path, contract)
        if checkpoints is not None:
            checkpoints.restore(output, run_id)
        query_paths = [candidates / "parts" / f"part-{i:03d}" / "queries.parquet"
                       for i in range(candidate_contract["buckets"])]
        ledger = pl.read_parquet(query_paths)
        if ledger.select("session", "objective").is_duplicated().any():
            raise ValueError("duplicate full-query ledger entries")
        folds = []
        for outer in outer_folds:
            results = {}
            for objective in OBJECTIVES:
                progress.update(stage=f"fold_{outer}_{objective}")
                directory = output / f"fold-{outer}" / objective
                directory.mkdir(parents=True, exist_ok=True)
                identity = canonical_json_sha256({"run_id": run_id, "fold": outer,
                                                  "objective": objective})
                completed = _completed_objective(directory, identity)
                if completed is not None:
                    logger.info("ranking_objective_reused", extra={"stage": progress["stage"]})
                    results[objective] = completed
                    progress["completed_objectives"] += 1
                    continue
                queries = ledger.filter(pl.col("objective") == objective)
                files = [candidates / "parts" / f"part-{i:03d}" / f"{objective}.parquet"
                         for i in range(candidate_contract["buckets"])]
                training_rows = pl.scan_parquet(files).filter(
                    pl.col("fold") != outer
                ).select(pl.len()).collect().item()
                estimate = training_memory_guard(training_rows, len(names), max_memory_gib)
                logger.info("ranking_memory_preflight", extra={
                    "stage": progress["stage"], "estimated_gib": round(estimate, 3),
                    "training_rows": training_rows,
                })
                fit = load_batch(files, queries, names, role="fit", outer_fold=outer)
                inner = load_batch(files, queries, names, role="inner", outer_fold=outer)
                outer_ids = queries.filter(role_filter("outer", outer))["session"].to_list()

                def publish_model(_: Path) -> None:
                    if checkpoints is not None:
                        checkpoints.publish(output)

                model, state = fit_ranker(
                    fit, inner, outer_sessions=outer_ids, objective=objective,
                    directory=directory, config=config, logger=logger, publish=publish_model,
                )
                del fit, inner
                temporary_model = directory / "model.txt.tmp"
                model.save_model(str(temporary_model), num_iteration=state["best_iteration"])
                temporary_model.replace(directory / "model.txt")
                learned_parts, baseline_parts = [], []
                prediction_started = time.perf_counter()
                for bucket, file in enumerate(files):
                    batch_queries = queries.filter(pl.col("bucket") == bucket)
                    batch = load_batch([file], batch_queries, names, role="outer", outer_fold=outer)
                    scores = (model.predict(batch.features, num_iteration=state["best_iteration"],
                                            num_threads=config.threads)
                              if batch.session.size else np.empty(0))
                    learned_parts.append(evaluate(batch, scores))
                    baseline_parts.append(evaluate(batch, baseline_scores(batch)))
                    logger.info("ranking_evaluation_bucket", extra={
                        "stage": progress["stage"], "bucket": bucket,
                        "elapsed_seconds": round(time.perf_counter() - prediction_started, 3),
                    })
                result = {"learned": pool_metrics(learned_parts),
                          "baseline": pool_metrics(baseline_parts),
                          "best_iteration": state["best_iteration"],
                          "inner_recall_at_20": state["best_score"],
                          "retained_fit_seconds": state["retained_fit_seconds"],
                          "evaluation_seconds": time.perf_counter() - prediction_started,
                          "model_sha256": sha256_file(directory / "model.txt")}
                write_json(directory / "evaluation.json", result)
                write_json(directory / "evaluation_receipt.json", {
                    "input_id": identity, "files": {
                        name: sha256_file(directory / name)
                        for name in ("evaluation.json", "model.txt")
                    },
                })
                if checkpoints is not None:
                    checkpoints.publish(output)
                results[objective] = result
                progress["completed_objectives"] += 1
            folds.append({"outer_fold": outer, "objectives": results})
        summary = {
            "status": "passed", "run_id": run_id, "completed_at_utc": utc_now_iso(),
            "folds": folds, "candidate_k": candidate_contract["config"]["candidate_k"],
            "learned": aggregate_official([
                {name: fold["objectives"][name]["learned"] for name in OBJECTIVES} for fold in folds
            ]),
            "baseline": aggregate_official([
                {name: fold["objectives"][name]["baseline"] for name in OBJECTIVES}
                for fold in folds
            ]),
            "attempt_elapsed_seconds": time.perf_counter() - started,
            "retained_fit_seconds": sum(fold["objectives"][name]["retained_fit_seconds"]
                                        for fold in folds for name in OBJECTIVES),
            "validation_scope": contract["validation_scope"],
            "untouched_temporal_holdout": False, "kaggle_submission": "not generated",
        }
        write_json(output / "metrics.json", summary)
        if checkpoints is not None:
            checkpoints.publish(output)
        logger.info("ranking_complete", extra={
            "elapsed_seconds": round(summary["attempt_elapsed_seconds"], 3),
            "total_elapsed_seconds": round(summary["retained_fit_seconds"], 3),
        })
        return summary
