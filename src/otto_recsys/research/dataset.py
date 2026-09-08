"""Bounded query batches with labels attached after candidate generation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.ranking.feature_cache import workspace_lock
from otto_recsys.research.features import Candidates, FeatureEngine, Prefix
from otto_recsys.research.protocol import atomic_json
from otto_recsys.research.retrievers import commit_file, verified_file
from otto_recsys.runtime import Heartbeat

OBJECTIVES = ("clicks", "carts", "orders")
METADATA = (
    "session",
    "aid",
    "candidate_position",
    "target_clicks",
    "target_carts",
    "target_orders",
)


def session_hash(values: Any, seed: int) -> Any:
    """Stable unsigned arithmetic, independent of Python's process hash seed."""
    ids = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        return (ids ^ np.uint64(seed)) * np.uint64(11400714819323198485)


class Queries:
    def __init__(self, corpus: Path, split_role: str) -> None:
        if split_role not in ("fit", "selection", "evaluation"):
            raise ValueError("invalid query role")
        manifest = json.loads((corpus / "manifest.json").read_text())
        for name in ("observed.parquet", "queries.parquet", "labels.parquet"):
            if sha256_file(corpus / name) != manifest["files"][name]:
                raise ValueError(f"corpus checksum mismatch: {name}")
        observed = (
            pl.scan_parquet(corpus / "observed.parquet")
            .filter(pl.col("split_role") == split_role)
            .sort("session", "event_index")
            .collect()
        )
        self.session, self.starts, self.counts = np.unique(
            observed["session"].to_numpy(), return_index=True, return_counts=True
        )
        self.aid = observed["aid"].to_numpy().astype(np.int64)
        self.ts = observed["ts"].to_numpy().astype(np.int64)
        self.kind = observed["event_type"].to_numpy().astype(np.int64)
        ledger = (
            pl.scan_parquet(corpus / "queries.parquet")
            .filter(pl.col("split_role") == split_role)
            .collect()
        )
        self.denominators = np.zeros((self.session.size, 3), dtype=np.int64)
        self.truth_counts = np.zeros_like(self.denominators)
        for j, objective in enumerate(OBJECTIVES):
            rows = ledger.filter(pl.col("objective") == objective).sort("session")
            if not np.array_equal(rows["session"].to_numpy(), self.session):
                raise ValueError("query ledger does not match observed prefixes")
            self.denominators[:, j] = rows["recall_denominator"].to_numpy()
            self.truth_counts[:, j] = rows["true_items"].to_numpy()
        labels = (
            pl.scan_parquet(corpus / "labels.parquet")
            .filter(pl.col("split_role") == split_role)
            .collect()
        )
        self.labels: dict[tuple[int, str], np.ndarray] = {}
        for objective in OBJECTIVES:
            groups = labels.filter(pl.col("objective") == objective).group_by("session").agg("aid")
            for session, aids in groups.iter_rows():
                self.labels[(session, objective)] = np.asarray(aids, dtype=np.int64)
        self.role = split_role
        self.manifest = manifest

    def prefix(self, index: int) -> Prefix:
        start = self.starts[index]
        stop = start + self.counts[index]
        return Prefix(
            int(self.session[index]),
            self.aid[start:stop],
            self.ts[start:stop],
            self.kind[start:stop],
        )

    def targets(self, session: int, candidates: Candidates) -> np.ndarray:
        return np.column_stack(
            [
                np.isin(candidates.aid, self.labels.get((session, objective), []))
                for objective in OBJECTIVES
            ]
        ).astype(np.int8)

    def indices(self, seed: int) -> np.ndarray:
        return np.argsort(session_hash(self.session, seed), kind="stable")


def sampled_rows(
    target: np.ndarray, aids: np.ndarray, session: int, budget: int, seed: int
) -> np.ndarray:
    """Fit only: all discovered positives plus a fixed mix of hard/random negatives."""
    positive = target.any(axis=1)
    negative = np.flatnonzero(~positive)
    hard = negative[: budget // 2]
    rest = negative[budget // 2 :]
    order = np.argsort(
        session_hash(rest.astype(np.int64) + aids[rest], seed + session), kind="stable"
    )
    random = rest[order[: budget - hard.size]]
    return np.sort(np.concatenate([np.flatnonzero(positive), hard, random]))


def build_cache(
    corpus: Path,
    retrieval: Path,
    output: Path,
    *,
    role: str,
    names: tuple[str, ...],
    candidate_budget: int,
    negative_budget: int | None,
    seed: int,
    logger: logging.Logger,
    max_rows: int | None = None,
    sessions_per_part: int = 256,
) -> dict[str, Any]:
    """Commit each complete query batch; verification precedes every resume."""
    if role not in ("fit", "selection") or (negative_budget is not None and role != "fit"):
        raise ValueError("only fitting queries may be negative-sampled; evaluation is streamed")
    output.mkdir(parents=True, exist_ok=True)
    with workspace_lock(output):
        queries = Queries(corpus, role)
        engine = FeatureEngine(retrieval)
        contract = {
            "corpus_id": queries.manifest["input_id"],
            "retrieval_id": engine.manifest["input_id"],
            "role": role,
            "features": names,
            "candidate_budget": candidate_budget,
            "negative_budget": negative_budget,
            "seed": seed,
            "max_rows": max_rows,
            "sessions_per_part": sessions_per_part,
            "code": {
                name: sha256_file(Path(__file__).with_name(name))
                for name in ("dataset.py", "features.py")
            },
        }
        input_id = canonical_json_sha256(contract)
        contract = json.loads(json.dumps(contract))
        path = output / "contract.json"
        if path.exists() and json.loads(path.read_text()) != contract:
            raise ValueError("feature cache has a different generation contract")
        atomic_json(path, contract)
        atomic_json(output / "catalog.json", engine.catalog())
        order = queries.indices(seed)
        progress: dict[str, Any] = {"phase": "features", "sessions": 0, "candidate_rows": 0}
        start = time.perf_counter()
        parts = []
        with Heartbeat(
            logger, stage="research_features", interval_seconds=15, progress_provider=progress.copy
        ):
            for bucket, begin in enumerate(range(0, order.size, sessions_per_part)):
                path = output / f"part-{bucket:04d}.parquet"
                indices = order[begin : begin + sessions_per_part]
                if verified_file(path, input_id):
                    metadata = json.loads(path.with_suffix(".json").read_text())
                    count = int(metadata["rows"])
                else:
                    matrices, sessions, aids, positions, labels = [], [], [], [], []
                    batch_start = time.perf_counter()
                    for index in indices:
                        prefix = queries.prefix(int(index))
                        candidates = engine.candidates(prefix, candidate_budget)
                        target = queries.targets(prefix.session, candidates)
                        chosen = (
                            np.arange(candidates.aid.size)
                            if negative_budget is None
                            else sampled_rows(
                                target, candidates.aid, prefix.session, negative_budget, seed
                            )
                        )
                        subset = Candidates(
                            candidates.aid[chosen],
                            candidates.source_scores[chosen],
                            candidates.source_ranks[chosen],
                            candidates.graph[:, chosen],
                        )
                        # Normalize on the complete candidate pool before sampling any fitting rows.
                        full = engine.transform(prefix, candidates, names)
                        matrices.append(full[chosen])
                        sessions.append(np.full(chosen.size, prefix.session, dtype=np.int32))
                        aids.append(subset.aid.astype(np.int32))
                        positions.append(chosen.astype(np.int16))
                        labels.append(target[chosen])
                    matrix = np.concatenate(matrices)
                    truth = np.concatenate(labels)
                    frame = pl.DataFrame(matrix, schema=list(names))
                    frame = frame.with_columns(
                        pl.Series("session", np.concatenate(sessions)),
                        pl.Series("aid", np.concatenate(aids)),
                        pl.Series("candidate_position", np.concatenate(positions)),
                        *[pl.Series(f"target_{o}", truth[:, j]) for j, o in enumerate(OBJECTIVES)],
                    )
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.write_parquet(temporary, compression="zstd")
                    temporary.replace(path)
                    commit_file(path, input_id, time.perf_counter() - batch_start)
                    metadata = json.loads(path.with_suffix(".json").read_text())
                    count = frame.height
                    metadata.update(rows=count, sessions=len(indices))
                    atomic_json(path.with_suffix(".json"), metadata)
                    logger.info(
                        "research_feature_part_complete",
                        extra={
                            "bucket": bucket,
                            "sessions": len(indices),
                            "elapsed_seconds": time.perf_counter() - batch_start,
                        },
                    )
                parts.append(path.name)
                progress["sessions"] += len(indices)
                progress["candidate_rows"] += count
                if max_rows is not None and progress["candidate_rows"] >= max_rows:
                    break
        report = {
            "status": "passed",
            "input_id": input_id,
            "role": role,
            "candidate_features": len(names),
            "rows": progress["candidate_rows"],
            "sessions": progress["sessions"],
            "parts": parts,
            "files": {name: sha256_file(output / name) for name in parts},
            "elapsed_seconds": time.perf_counter() - start,
        }
        atomic_json(output / "manifest.json", report)
        return report


def cache_batches(directory: Path, names: tuple[str, ...]) -> Iterator[pl.DataFrame]:
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in manifest["parts"]:
        path = directory / name
        if sha256_file(path) != manifest["files"][name]:
            raise ValueError("feature cache checksum mismatch")
        yield pl.read_parquet(path, columns=[*names, *METADATA])
