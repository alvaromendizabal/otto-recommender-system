"""Nested LambdaRank fits with full-query metrics and atomic iteration checkpoints.

This engine accepts already-materialized, label-blind candidate features. It does
not certify upstream retriever provenance or claim an untouched temporal holdout.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import lightgbm as lgb
import numpy as np
from numpy.typing import NDArray

from otto_recsys.runtime import Heartbeat

OBJECTIVE_WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}
EXCLUDED_FEATURES = frozenset({
    "session", "aid", "objective", "query_id", "fold", "bucket",
    "inner_partition", "target", "true_items", "recall_denominator",
})


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _integers(values: Any, name: str) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu" or (raw < 0).any():
        raise ValueError(f"{name} must be a one-dimensional nonnegative integer array")
    if raw.size and int(raw.max()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds int64")
    return np.array(raw, dtype=np.int64, copy=True)


@dataclass(frozen=True)
class QueryBatch:
    """One objective; full truth counts include sessions with no candidates.

    IDs and target are metadata, never model features. Query rows are sorted
    contiguously by session and item, and immutable copies prevent hash drift.
    """

    features: NDArray[np.float32]
    session: NDArray[np.int64]
    aid: NDArray[np.int64]
    target: NDArray[np.int64]
    truth_counts: Mapping[int, int]
    feature_names: tuple[str, ...]

    @classmethod
    def create(
        cls, features: Any, session: Any, aid: Any, target: Any,
        truth_counts: Mapping[int, int], feature_names: Sequence[str],
    ) -> QueryBatch:
        ids = _integers(session, "session")
        items = _integers(aid, "aid")
        labels = _integers(target, "target")
        names = tuple(feature_names)
        if not names or len(set(names)) != len(names) or EXCLUDED_FEATURES.intersection(names):
            raise ValueError("feature names must be unique, nonempty and exclude labels/IDs/splits")
        if any(not isinstance(name, str) or not name
               or any(c in name for c in '[]{}":,') for name in names):
            raise ValueError("feature names contain unsupported characters")
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape != (ids.size, len(names)):
            raise ValueError("feature matrix shape does not match rows and feature names")
        if items.shape != ids.shape or labels.shape != ids.shape or (labels > 1).any():
            raise ValueError("candidate arrays must align and targets must be binary")
        if np.isinf(x).any():
            raise ValueError("infinite features are not supported; missing features use NaN")
        counts = {}
        for key, count in truth_counts.items():
            if (isinstance(key, bool) or isinstance(count, bool)
                    or not isinstance(key, (int, np.integer))
                    or not isinstance(count, (int, np.integer))
                    or key < 0 or count < 0):
                raise ValueError("truth counts must map nonnegative integer sessions to counts")
            counts[int(key)] = int(count)
        if not set(ids.tolist()).issubset(counts):
            raise ValueError("candidate sessions are missing from the full query ledger")
        order = np.lexsort((items, ids))
        ids, items, labels = ids[order], items[order], labels[order]
        if ids.size > 1 and ((ids[1:] == ids[:-1]) & (items[1:] == items[:-1])).any():
            raise ValueError("duplicate session/item candidates")
        x = np.array(x[order], dtype=np.float32, copy=True, order="C")
        if ids.size:
            unique, starts = np.unique(ids, return_index=True)
            hits = np.add.reduceat(labels, starts)
            if any(int(hit) > counts[int(s)] for s, hit in zip(unique, hits, strict=True)):
                raise ValueError("candidate positives exceed full-query true item counts")
        for array in (x, ids, items, labels):
            array.setflags(write=False)
        return cls(x, ids, items, labels, MappingProxyType(counts), names)

    @property
    def groups(self) -> NDArray[np.int32]:
        _, counts = np.unique(self.session, return_counts=True)
        if counts.size and counts.max() > np.iinfo(np.int32).max:
            raise ValueError("query group exceeds LightGBM's int32 limit")
        return counts.astype(np.int32)

    def fingerprint(self) -> str:
        payload = {
            name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
            for name, value in (("features", self.features), ("session", self.session),
                                ("aid", self.aid), ("target", self.target))
        }
        payload["schema"] = _digest(list(self.feature_names))
        payload["ledger"] = _digest(sorted(self.truth_counts.items()))
        return _digest(payload)


def evaluate(batch: QueryBatch, scores: Any, *, k: int = 20) -> dict[str, Any]:
    """Official micro Recall@20 plus query-averaged NDCG/MRR/hit rate.

    Missing candidate queries contribute zero hits, not a smaller denominator.
    Ties are resolved by ascending item ID. Candidate ceiling is not ranked recall.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k != 20:
        raise ValueError("this evaluator implements the official k=20 metric only")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != batch.session.shape or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and aligned with all candidate rows")
    denominator = sum(min(k, count) for count in batch.truth_counts.values())
    labeled = sum(count > 0 for count in batch.truth_counts.values())
    hits = ceiling = 0
    dcg = reciprocal = hit_queries = 0.0
    order = np.lexsort((batch.aid, -scores, batch.session))
    sessions = batch.session[order]
    labels = batch.target[order]
    unique, starts, lengths = np.unique(sessions, return_index=True, return_counts=True)
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    for session, start, length in zip(unique, starts, lengths, strict=True):
        count = batch.truth_counts[int(session)]
        all_labels = labels[start:start + length]
        relevant = all_labels[:k]
        found = int(relevant.sum())
        hits += found
        ceiling += min(k, int(all_labels.sum()))
        if count:
            dcg += float(np.dot(relevant, discounts[:relevant.size])) / float(
                discounts[:min(k, count)].sum()
            )
            positions = np.flatnonzero(relevant)
            if positions.size:
                reciprocal += 1.0 / (int(positions[0]) + 1)
                hit_queries += 1
    return {
        "hits": hits, "denominator": denominator, "labeled_queries": labeled,
        "all_queries": len(batch.truth_counts), "candidate_rows": int(batch.session.size),
        "recall_at_20": hits / denominator if denominator else None,
        "candidate_ceiling_at_20": ceiling / denominator if denominator else None,
        "ndcg_at_20": dcg / labeled if labeled else None,
        "mrr_at_20": reciprocal / labeled if labeled else None,
        "hit_rate_at_20": hit_queries / labeled if labeled else None,
        "k": k,
    }


def aggregate_official(fold_metrics: Sequence[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Pool objective numerators/denominators across folds, then apply weights."""
    if not fold_metrics:
        raise ValueError("at least one evaluated fold is required")
    pooled = {}
    for objective in OBJECTIVE_WEIGHTS:
        hits = denominator = 0
        for fold in fold_metrics:
            if set(fold) != set(OBJECTIVE_WEIGHTS):
                raise ValueError("each fold must contain all three objectives")
            part = fold[objective]
            h, d = part["hits"], part["denominator"]
            if (isinstance(h, bool) or isinstance(d, bool)
                    or not isinstance(h, int) or not isinstance(d, int)
                    or not 0 <= h <= d):
                raise ValueError("invalid objective numerator/denominator")
            hits += h
            denominator += d
        if not denominator:
            raise ValueError(f"{objective} has no labeled queries")
        pooled[objective] = {"hits": hits, "denominator": denominator,
                             "recall_at_20": hits / denominator}
    return {"objectives": pooled, "weighted_recall_at_20": sum(
        weight * pooled[name]["recall_at_20"] for name, weight in OBJECTIVE_WEIGHTS.items()
    )}


@dataclass(frozen=True)
class RankerConfig:
    rounds: int = 300
    patience: int = 30
    checkpoint_every: int = 10
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    threads: int = 4
    seed: int = 20260907

    def validate(self) -> None:
        for name in ("rounds", "patience", "checkpoint_every", "num_leaves",
                     "min_data_in_leaf", "threads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (not isinstance(self.seed, int) or isinstance(self.seed, bool)
                or not 0 <= self.seed < 2**31):
            raise ValueError("seed must be an integer in [0, 2**31)")
        if self.num_leaves < 2 or not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("invalid tree size or learning rate")


def _fit_ranker(
    fit: QueryBatch, inner: QueryBatch, *, outer_sessions: Sequence[int],
    objective: str, directory: Path, config: RankerConfig, logger: logging.Logger,
    heartbeat_context: AbstractContextManager[Any] | None = None,
    publish: Callable[[Path], None] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Fit with inner-only early stopping and resume verified iteration snapshots.

    `publish` runs after an atomic local snapshot and must raise on upload failure.
    Remote restoration and a single remote writer are the orchestration layer's
    responsibility. The final outer labels are deliberately not accepted here.
    """
    config.validate()
    if objective not in OBJECTIVE_WEIGHTS:
        raise ValueError("unsupported objective")
    fit_ids, inner_ids = set(fit.truth_counts), set(inner.truth_counts)
    outer_ids = set(_integers(outer_sessions, "outer_sessions").tolist())
    if not fit_ids or not inner_ids or not outer_ids:
        raise ValueError("fit, inner and outer query sets must be nonempty")
    if fit_ids & inner_ids or fit_ids & outer_ids or inner_ids & outer_ids:
        raise ValueError("fit/inner/outer sessions overlap")
    if fit.feature_names != inner.feature_names:
        raise ValueError("fit and inner feature order differs")
    for name, batch in (("fit", fit), ("inner", inner)):
        if batch.session.size == 0 or np.unique(batch.target).size < 2:
            raise ValueError(f"{name} requires candidate rows with both target classes")
        if batch.groups.max() > 10000:
            raise ValueError("candidate group exceeds the supported 10000-row ranker budget")
        starts = np.r_[0, np.cumsum(batch.groups)[:-1]]
        positives = np.add.reduceat(batch.target, starts)
        if not ((positives > 0) & (positives < batch.groups)).any():
            raise ValueError(f"{name} has no query containing a positive/negative pair")
    params = {
        "objective": "lambdarank", "metric": "None", "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves, "min_data_in_leaf": config.min_data_in_leaf,
        "num_threads": config.threads, "seed": config.seed, "deterministic": True,
        "force_col_wise": True, "feature_pre_filter": False,
        "lambdarank_truncation_level": 25, "verbosity": 0,
    }
    contract = {
        "schema_version": 1, "objective": objective, "config": asdict(config),
        "parameters": params, "feature_names": list(fit.feature_names),
        "fit_sha256": fit.fingerprint(), "inner_sha256": inner.fingerprint(),
        "outer_sessions_sha256": _digest(sorted(outer_ids)),
        "lightgbm": lgb.__version__, "numpy": np.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selection_metric": "inner full-query Recall@20; first maximum wins ties",
        "validation_scope": "nested session split; upstream fit provenance not certified",
    }
    input_id = _digest(contract)
    directory.mkdir(parents=True, exist_ok=True)
    contract_path = directory / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("ranker contract mismatch; preserve this run and use a new directory")
    _atomic_json(contract_path, contract)
    state: dict[str, Any] = {
        "input_id": input_id, "iteration": 0, "best_iteration": 0,
        "best_score": -1.0, "retained_fit_seconds": 0.0, "complete": False,
    }
    initial = None
    for path in sorted((directory / "checkpoints").glob("*.json"), reverse=True):
        try:
            saved = json.loads(path.read_text())
            model = saved["model"]
            evidence = saved["state"]
            if (saved["checksum"] != _digest({"model": model, "state": evidence})
                    or evidence["input_id"] != input_id):
                continue
            candidate = lgb.Booster(model_str=model)
            if candidate.current_iteration() != evidence["iteration"]:
                continue
            state, initial = evidence, candidate
            break
        except (OSError, ValueError, TypeError, KeyError, lgb.basic.LightGBMError):
            continue
    if state["complete"]:
        logger.info("ranker_fit_reused", extra={"stage": objective, "input_id": input_id})
        if publish is not None:
            publish(directory)
        if initial is None:
            raise RuntimeError("completed checkpoint has no model")
        initial.best_iteration = int(state["best_iteration"])
        return initial, state
    started = time.perf_counter()
    retained = float(state["retained_fit_seconds"])
    logger.info("ranker_fit_start", extra={"stage": objective, "input_id": input_id,
                "resumed_iteration": state["iteration"]})
    train = lgb.Dataset(fit.features, label=fit.target, group=fit.groups,
                        feature_name=list(fit.feature_names), free_raw_data=False)
    valid = lgb.Dataset(inner.features, label=inner.target, group=inner.groups,
                        reference=train, feature_name=list(inner.feature_names),
                        free_raw_data=False)

    def metric(predictions: Any, dataset: Any) -> tuple[str, float, bool]:
        value = evaluate(inner, predictions)["recall_at_20"]
        if value is None:
            raise ValueError("inner split lacks an official metric denominator")
        return "full_query_recall_at_20", float(value), True

    def save(booster: lgb.Booster, complete: bool = False) -> None:
        state["complete"] = complete
        state["retained_fit_seconds"] = retained + time.perf_counter() - started
        payload = {"model": booster.model_to_string(num_iteration=-1), "state": dict(state)}
        payload["checksum"] = _digest(payload)
        _atomic_json(directory / "checkpoints" / f"{state['iteration']:06d}.json", payload)
        if publish is not None:
            publish(directory)
        logger.info("ranker_checkpoint_complete", extra={
            "stage": objective, "iteration": state["iteration"],
            "best_iteration": state["best_iteration"], "best_score": state["best_score"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "total_elapsed_seconds": round(state["retained_fit_seconds"], 3),
        })

    def progress(env: Any) -> None:
        if not env.evaluation_result_list:
            raise RuntimeError("inner metric was not evaluated")
        state["iteration"] = env.model.current_iteration()
        score = float(env.evaluation_result_list[0][2])
        if score > state["best_score"]:
            state["best_score"] = score
            state["best_iteration"] = state["iteration"]
        exhausted = env.iteration + 1 - int(state["best_iteration"]) >= config.patience
        finished = exhausted or env.iteration + 1 >= config.rounds
        if state["iteration"] % config.checkpoint_every == 0 or finished:
            save(env.model, complete=finished)
        if exhausted:
            raise lgb.callback.EarlyStopException(
                int(state["best_iteration"]) - 1, env.evaluation_result_list
            )

    try:
        with heartbeat_context if heartbeat_context is not None else Heartbeat(
            logger, stage=f"ranking_{objective}", interval_seconds=15,
            progress_provider=lambda: {
                "iteration": state["iteration"],
                "best_iteration": state["best_iteration"],
                "total_elapsed_seconds": round(retained + time.perf_counter() - started, 3),
            },
        ):
            booster = lgb.train(
                params, train, num_boost_round=config.rounds - int(state["iteration"]),
                valid_sets=[valid], valid_names=["inner"], feval=metric,
                init_model=initial, keep_training_booster=True, callbacks=[progress],
            )
        booster.best_iteration = int(state["best_iteration"])
        save(booster, complete=True)
    except BaseException:
        logger.exception("ranker_fit_failed", extra={"stage": objective,
                         "elapsed_seconds": round(time.perf_counter() - started, 3)})
        raise
    logger.info("ranker_fit_complete", extra={"stage": objective,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "total_elapsed_seconds": round(state["retained_fit_seconds"], 3)})
    return booster, state


def fit_ranker(
    fit: QueryBatch, inner: QueryBatch, *, outer_sessions: Sequence[int],
    objective: str, directory: Path, config: RankerConfig, logger: logging.Logger,
    heartbeat_context: AbstractContextManager[Any] | None = None,
    publish: Callable[[Path], None] | None = None,
) -> tuple[lgb.Booster, dict[str, Any]]:
    """Run one fit under a nonblocking Linux workspace lock.

    The project heartbeat is active by default. Configure the supplied logger
    with the project UTC formatter, and supply a durable upload callback as
    `publish` in cloud orchestration. Restore the contract and checkpoint
    directory before calling this function on a fresh worker.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("ranker workspace already has an active writer") from error
        try:
            return _fit_ranker(
                fit, inner, outer_sessions=outer_sessions, objective=objective,
                directory=directory, config=config, logger=logger,
                heartbeat_context=heartbeat_context, publish=publish,
            )
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
