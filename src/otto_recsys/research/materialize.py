"""Parallel, bounded materialization of the selected feature schema."""

from __future__ import annotations

import json
import logging
import multiprocessing
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.dataset import METADATA, OBJECTIVES, Queries, sampled_rows
from otto_recsys.research.features import FeatureEngine
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import verified_file
from otto_recsys.runtime import Heartbeat

MODEL_METADATA = (*METADATA, *[f"baseline_{objective}" for objective in OBJECTIVES])
_ENGINE: FeatureEngine | None = None
_QUERIES: Queries | None = None
_SCHEMA: tuple[str, ...] = ()
_BUDGET = 400
_NEGATIVES: int | None = None
_SEED = 0


def initialize_worker(
    corpus: Path,
    retrieval: Path,
    role: str,
    names: tuple[str, ...],
    budget: int,
    negatives: int | None,
    seed: int,
) -> None:
    global _ENGINE, _QUERIES, _SCHEMA, _BUDGET, _NEGATIVES, _SEED
    _ENGINE = FeatureEngine(retrieval)
    _QUERIES = Queries(corpus, role)
    _SCHEMA, _BUDGET, _NEGATIVES, _SEED = names, budget, negatives, seed


def transform_query(index: int) -> dict[str, Any]:
    if _ENGINE is None or _QUERIES is None:
        raise RuntimeError("feature worker is not initialized")
    prefix = _QUERIES.prefix(index)
    candidates = _ENGINE.candidates(prefix, _BUDGET)
    target = _QUERIES.targets(prefix.session, candidates)
    matrix = _ENGINE.transform(prefix, candidates, _SCHEMA)
    chosen = (
        np.arange(candidates.aid.size)
        if _NEGATIVES is None
        else sampled_rows(target, candidates.aid, prefix.session, _NEGATIVES, _SEED)
    )
    evidence = np.where(candidates.source_ranks > 0, 1 / (20 + candidates.source_ranks), 0)
    # Three fixed objective fusions; identical policy across all ranker ablations.
    weights = np.array(
        [[1.0, 0.1, 0.1, 1.0, 0.01], [0.3, 1.0, 0.3, 1.0, 0.01], [0.2, 0.5, 1.0, 1.0, 0.01]]
    )
    baseline = evidence @ weights.T
    return {
        "features": matrix[chosen],
        "target": target[chosen],
        "aid": candidates.aid[chosen],
        "session": prefix.session,
        "candidate_position": chosen,
        "baseline": baseline[chosen],
    }


def model_cache(
    corpus: Path,
    retrieval: Path,
    output: Path,
    *,
    role: str,
    names: tuple[str, ...],
    budget: int,
    negatives: int,
    seed: int,
    workers: int,
    logger: logging.Logger,
    publish: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    if role not in ("fit", "selection") or not 1 <= workers <= 16:
        raise ValueError("model cache requires fit/selection roles and 1-16 workers")
    queries = Queries(corpus, role)
    if not queries.session.size:
        raise ValueError("model cache requires at least one eligible session")
    negatives_for_role = negatives if role == "fit" else None
    fitted = json.loads((retrieval / "manifest.json").read_text())
    contract = {
        "corpus_id": queries.manifest["input_id"],
        "retrieval_id": fitted["input_id"],
        "role": role,
        "features": list(names),
        "candidate_budget": budget,
        "negative_budget": negatives_for_role,
        "seed": seed,
        "sessions_per_part": 256,
        "code": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in ("materialize.py", "features.py", "dataset.py")
        },
    }
    input_id = canonical_json_sha256(contract)
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        path = output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("model feature cache has a different fitted input contract")
        atomic_json(path, contract)
        if publish:
            publish(path)
        progress: dict[str, Any] = {"sessions": 0, "candidate_rows": 0}
        parts = []
        start = time.perf_counter()
        initializer_args = (corpus, retrieval, role, names, budget, negatives_for_role, seed)
        with (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=initializer_args,
            ) as pool,
            Heartbeat(
                logger,
                stage=f"features_{role}",
                interval_seconds=15,
                progress_provider=progress.copy,
            ),
        ):
            for bucket, begin in enumerate(range(0, queries.session.size, 256)):
                path = output / f"part-{bucket:04d}.parquet"
                indices = range(begin, min(begin + 256, queries.session.size))
                if verified_file(path, input_id):
                    receipt = json.loads(path.with_suffix(".json").read_text())
                    count = int(receipt["rows"])
                else:
                    batch_start = time.perf_counter()
                    rows = list(pool.map(transform_query, indices, chunksize=8))
                    matrix = np.concatenate([r["features"] for r in rows])
                    target = np.concatenate([r["target"] for r in rows])
                    baseline = np.concatenate([r["baseline"] for r in rows])
                    frame = pl.DataFrame(matrix, schema=list(names)).with_columns(
                        pl.Series(
                            "session",
                            np.concatenate(
                                [np.full(r["aid"].size, r["session"], dtype=np.int32) for r in rows]
                            ),
                        ),
                        pl.Series("aid", np.concatenate([r["aid"] for r in rows]).astype(np.int32)),
                        pl.Series(
                            "candidate_position",
                            np.concatenate([r["candidate_position"] for r in rows]).astype(
                                np.int16
                            ),
                        ),
                        *[pl.Series(f"target_{o}", target[:, j]) for j, o in enumerate(OBJECTIVES)],
                        *[
                            pl.Series(f"baseline_{o}", baseline[:, j])
                            for j, o in enumerate(OBJECTIVES)
                        ],
                    )
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.write_parquet(temporary, compression="zstd")
                    temporary.replace(path)
                    count = frame.height
                    receipt = {
                        "input_id": input_id,
                        "sha256": sha256_file(path),
                        "rows": count,
                        "sessions": len(rows),
                        "elapsed_seconds": time.perf_counter() - batch_start,
                    }
                    atomic_json(path.with_suffix(".json"), receipt)
                    logger.info(
                        "selected_features_part_complete",
                        extra={
                            "bucket": bucket,
                            "sessions": len(rows),
                            "elapsed_seconds": time.perf_counter() - batch_start,
                        },
                    )
                if publish:
                    publish(path)
                    publish(path.with_suffix(".json"))
                progress["sessions"] += len(indices)
                progress["candidate_rows"] += count
                parts.append(path.name)
        report = {
            "status": "passed",
            "input_id": input_id,
            "role": role,
            "features": list(names),
            "rows": progress["candidate_rows"],
            "sessions": progress["sessions"],
            "parts": parts,
            "files": {name: sha256_file(output / name) for name in parts},
            "elapsed_seconds": time.perf_counter() - start,
        }
        atomic_json(output / "manifest.json", report)
        if publish:
            publish(output / "manifest.json")
        return report
