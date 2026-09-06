"""Sharded IVF construction, exact score reranking, and honest search timing."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .benchmark_artifacts import BenchmarkArtifacts
from .ranking_metrics import ranking_counts


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(handle, allow_pickle=False, **arrays)


def build_index(
    vectors: np.ndarray,
    item_ids: np.ndarray,
    args: argparse.Namespace,
    artifacts: BenchmarkArtifacts,
    objective: str,
) -> tuple[Any, dict[str, Any]]:
    prefix = f"indices/{objective}"
    final = artifacts.get(prefix + "/index.faiss")
    stats_path = artifacts.get(prefix + "/build.json")
    if final is not None and stats_path is not None:
        from .evaluation import read_json

        started = time.perf_counter()
        index = faiss.read_index(str(final))
        return index, {
            **read_json(stats_path),
            "load_seconds_this_attempt": time.perf_counter() - started,
        }
    if len(vectors) < 39 * args.nlist:
        raise ValueError("catalogue too small for the configured centroid count")

    def train(path: Path) -> None:
        index = faiss.IndexIVFFlat(
            faiss.IndexFlatIP(vectors.shape[1]),
            vectors.shape[1],
            args.nlist,
            faiss.METRIC_INNER_PRODUCT,
        )
        index.cp.seed = args.seed
        index.cp.niter = args.train_iterations
        sample = np.random.default_rng(args.seed).choice(
            len(vectors), min(args.train_items, len(vectors)), replace=False
        )
        index.train(np.ascontiguousarray(vectors[sample]))
        faiss.write_index(index, str(path))

    trained = artifacts.produce(prefix + "/trained.faiss", train)
    shards = []
    for start in range(0, len(vectors), args.index_shard_rows):
        end = min(start + args.index_shard_rows, len(vectors))

        def add(path: Path, start: int = start, end: int = end) -> None:
            index = faiss.read_index(str(trained))
            index.add_with_ids(
                np.ascontiguousarray(vectors[start:end]),
                np.ascontiguousarray(item_ids[start:end], dtype=np.int64),
            )
            faiss.write_index(index, str(path))

        shards.append(artifacts.produce(prefix + f"/shards/part-{start:08d}.faiss", add))

    def merge(path: Path) -> None:
        index = faiss.read_index(str(trained))
        for shard in shards:
            index.merge_from(faiss.read_index(str(shard)), 0)
        if index.ntotal != len(vectors):
            raise ValueError("merged ANN index row count mismatch")
        faiss.write_index(index, str(path))

    final = artifacts.produce(prefix + "/index.faiss", merge)
    stats = artifacts.json(
        prefix + "/build.json",
        lambda: {
            "method": "IVFFlat inner product",
            "nlist": args.nlist,
            "catalogue_items": len(vectors),
            "dimension": vectors.shape[1],
            "index_bytes": final.stat().st_size,
            "shards": len(shards),
            "retained_build_compute_seconds": sum(
                float(row["elapsed_seconds"])
                for name, row in artifacts.used.items()
                if name.startswith(prefix + "/")
            ),
        },
    )
    started = time.perf_counter()
    index = faiss.read_index(str(final))
    return index, {**stats, "load_seconds_this_attempt": time.perf_counter() - started}


def search(
    index: Any,
    queries: np.ndarray,
    vectors: np.ndarray,
    item_ids: np.ndarray,
    k: int,
    *,
    positional: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Rerank the returned candidate pool by FP32 score, then ascending aid.

    Ties outside this returned pool can still differ from global exact ties.
    Missing neighbours are errors, never silently treated as negative item IDs.
    """
    _, ids = index.search(np.ascontiguousarray(queries), min(k + 32, len(vectors)))
    output_ids, output_scores = [], []
    for query, found in zip(queries, ids, strict=True):
        found = found[found >= 0]
        if len(found) < k or len(np.unique(found)) != len(found):
            raise ValueError("ANN search returned insufficient or duplicate neighbours")
        positions = found if positional else np.searchsorted(item_ids, found)
        if np.any(positions >= len(item_ids)) or (
            not positional and not np.array_equal(item_ids[positions], found)
        ):
            raise ValueError("ANN index returned unknown catalogue IDs")
        aids = item_ids[positions]
        scores = np.einsum("nd,d->n", vectors[positions], query, optimize=False)
        order = np.lexsort((aids, -scores))[:k]
        output_ids.append(aids[order])
        output_scores.append(scores[order])
    return np.asarray(output_scores, dtype=np.float32), np.asarray(output_ids, dtype=np.int64)


def latency(
    index: Any,
    queries: np.ndarray,
    vectors: np.ndarray,
    item_ids: np.ndarray,
    args: argparse.Namespace,
    *,
    positional: bool = False,
) -> dict[str, Any]:
    subset = queries[: min(args.latency_queries, len(queries))]
    for j in range(args.warmup_queries):
        search(
            index,
            subset[j % len(subset) : j % len(subset) + 1],
            vectors,
            item_ids,
            args.candidate_depth,
            positional=positional,
        )
    observations = []
    order = np.random.default_rng(args.seed).permutation(len(subset))
    for _ in range(args.latency_repeats):
        for j in order:
            started = time.perf_counter()
            search(
                index,
                subset[j : j + 1],
                vectors,
                item_ids,
                args.candidate_depth,
                positional=positional,
            )
            observations.append(1000 * (time.perf_counter() - started))
    return {
        "scope": "warm batch-1 search plus FP32 reranking; excludes encoder, network and loading",
        "unique_queries": len(subset),
        "repeats": args.latency_repeats,
        "warmup_calls": args.warmup_queries,
        "samples": len(observations),
        "p50_ms": float(np.quantile(observations, 0.5)),
        "p95_ms": float(np.quantile(observations, 0.95)),
        "p99_ms": float(np.quantile(observations, 0.99)),
        "observations_ms": observations,
    }


def evaluate_queries(
    index: Any,
    vectors: np.ndarray,
    item_ids: np.ndarray,
    sessions: np.ndarray,
    queries: np.ndarray,
    exact_ids: np.ndarray,
    truth: dict[int, set[int]],
    args: argparse.Namespace,
    artifacts: BenchmarkArtifacts,
    prefix: str,
) -> dict[str, Any]:
    arrays = []
    depths = sorted({20, min(400, args.candidate_depth), args.candidate_depth})
    for start in range(0, len(sessions), args.batch_size):
        end = min(start + args.batch_size, len(sessions))

        def write(path: Path, start: int = start, end: int = end) -> None:
            begin = time.perf_counter()
            scores, ids = search(index, queries[start:end], vectors, item_ids, args.candidate_depth)
            duration = time.perf_counter() - begin
            write_npz(
                path,
                sessions=sessions[start:end],
                aids=ids,
                scores=scores,
                search_seconds=np.array(duration),
            )

        path = artifacts.produce(prefix + f"/parts/part-{start:06d}.npz", write)
        with np.load(path, allow_pickle=False) as part:
            if not np.array_equal(part["sessions"], sessions[start:end]):
                raise ValueError("query checkpoint session alignment mismatch")
            arrays.append({name: part[name] for name in part.files})
    found = np.concatenate([a["aids"] for a in arrays])
    if found.shape != (len(sessions), args.candidate_depth):
        raise ValueError("query checkpoint depth mismatch")
    ranked = np.array(
        [
            ranking_counts(truth.get(int(session), set()), row.tolist())
            for session, row in zip(sessions, found, strict=True)
        ]
    )
    fidelity, ceiling, exact_positive_retention = {}, {}, {}
    for depth in depths:
        overlaps, hits, recovered, reference_hits = [], [], 0, 0
        for session, row, exact in zip(sessions, found, exact_ids, strict=True):
            actual, expected = set(row[:depth]), set(exact[:depth])
            positives = truth.get(int(session), set())
            overlaps.append(len(actual & expected) / depth)
            hits.append(min(20, len(actual & positives)))
            reference_hits += len(expected & positives)
            recovered += len(expected & positives & actual)
        fidelity[str(depth)] = float(np.mean(overlaps))
        ceiling[str(depth)] = float(sum(hits) / ranked[:, 0].sum())
        exact_positive_retention[str(depth)] = (
            recovered / reference_hits if reference_hits else None
        )
    timing = artifacts.json(
        prefix + "/latency.json", lambda: latency(index, queries, vectors, item_ids, args)
    )
    seconds = sum(float(a["search_seconds"]) for a in arrays)
    return {
        "ranking_counts": ranked,
        "fidelity": fidelity,
        "candidate_ceiling": ceiling,
        "exact_positive_retention": exact_positive_retention,
        "latency": timing,
        "batch_throughput_queries_per_second": len(sessions) / seconds,
        "batch_search_seconds": seconds,
        "query_parts": len(arrays),
    }
