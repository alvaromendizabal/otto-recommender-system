"""Nested, label-blind candidate expansion from existing cutoff-certified graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

CHANNELS = ("time", "cart", "order")
ARMS = ("baseline_400", "wide_onehop_800", "wide_onehop_1200",
        "wide_twohop_800", "wide_twohop_1200")


@dataclass(frozen=True)
class Frontier:
    """Ranks refer to the complete discovery universe, not a sampled training pool."""

    aid: np.ndarray
    source_names: tuple[str, ...]
    source_scores: np.ndarray
    source_ranks: np.ndarray
    baseline_size: int


def integer_ids(values: np.ndarray) -> None:
    if values.ndim != 1 or values.dtype.kind not in "iu" or (values < 0).any():
        raise ValueError("IDs must be a one-dimensional nonnegative integer array")


def positive_ranks(scores: np.ndarray, aids: np.ndarray) -> np.ndarray:
    """Descending positive score, with item ID as the deterministic tie breaker."""
    if scores.shape != aids.shape or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Source scores must be aligned, finite and nonnegative")
    order = np.lexsort((aids, -scores))
    active = order[scores[order] > 0]
    result = np.zeros(scores.size, dtype=np.float32)
    result[active] = np.arange(1, active.size + 1)
    return result


class CandidateFrontier:
    """Reuse one-hop evidence once and bound two-hop expansion by historical bridges.

    The engine and graph objects expose their existing certified cutoff/CSR interfaces.
    No label, target or prediction argument is accepted by generation.
    """

    def __init__(self, engine: Any, wide: dict[str, Any], *, bridge_per_source: int = 64,
                 rrf_k: float = 20.0, max_discovery_items: int = 250000) -> None:
        if isinstance(bridge_per_source, bool) or not 1 <= bridge_per_source <= 64:
            raise ValueError("Bridge budget must be 1..64")
        if not np.isfinite(rrf_k) or rrf_k <= 0 or max_discovery_items < 1200:
            raise ValueError("RRF/discovery limits are invalid")
        if set(wide) != {"symmetric", "forward"}:
            raise ValueError("Both certified wide graph families are required")
        if any(g.cutoff != engine.cutoff for g in wide.values()):
            raise ValueError("Historical cutoff mismatch")
        self.engine = engine
        self.bridge_per_source = int(bridge_per_source)
        self.rrf_k = float(rrf_k)
        self.max_discovery_items = int(max_discovery_items)
        self.graphs = [(f"base_{c}", g) for c, g in zip(CHANNELS, engine.graphs, strict=True)]
        for family in ("symmetric", "forward"):
            self.graphs.extend((f"{family}_{c}", g)
                               for c, g in zip(CHANNELS, wide[family].graphs, strict=True))
        if any(g.shape[0] != g.shape[1] for _, g in self.graphs):
            raise ValueError("Graphs must have square item catalogues")
        one = tuple(name for name, _ in self.graphs)
        self.source_names = (*one, "revisit", "popularity", *(f"twohop_{n}" for n in one))
        self.weights = np.asarray(
            [0.4, 0.3, 0.6, 0.5, 0.4, 0.7, 0.55, 0.45, 0.8, 1.0, 0.01,
             0.15, 0.12, 0.2, 0.2, 0.16, 0.28, 0.22, 0.18, 0.32], dtype=np.float32)

    @staticmethod
    def _aggregate(graph: Any, ids: np.ndarray, weights: np.ndarray) -> Any:
        if not ids.size:
            return csr_matrix((1, graph.shape[1]), dtype=np.float32)
        result = (csr_matrix(weights.reshape(1, -1)) @ graph[ids]).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        return result

    def _select(self, prefix: Any, baseline: np.ndarray, evidence: list[Any]) -> Frontier:
        discoveries = [baseline, prefix.aid, *self.engine.popular]
        discoveries.extend(row.indices for _, row in evidence)
        universe = np.unique(np.concatenate(discoveries))
        if universe.size > self.max_discovery_items:
            raise ValueError("Discovery limit exceeded; preserve checkpoints, do not silently trim")
        scores = np.zeros((universe.size, len(self.source_names)), dtype=np.float32)
        columns = {name: i for i, name in enumerate(self.source_names)}
        for name, row in evidence:
            scores[np.searchsorted(universe, row.indices), columns[name]] = row.data
        np.add.at(scores[:, columns["revisit"]], np.searchsorted(universe, prefix.aid[::-1]),
                  (1 / np.sqrt(np.arange(1, prefix.aid.size + 1))).astype(np.float32))
        for weight, popular in zip((0.1, 0.3, 0.6), self.engine.popular, strict=True):
            scores[np.searchsorted(universe, popular), columns["popularity"]] += (
                weight / np.arange(1, popular.size + 1)).astype(np.float32)
        ranks = np.column_stack([positive_ranks(scores[:, j], universe)
                                 for j in range(scores.shape[1])])
        fusion = np.where(ranks > 0, self.weights / (self.rrf_k + ranks), 0).sum(axis=1)
        base_index = np.searchsorted(universe, baseline)
        order = np.lexsort((universe, -fusion))
        extra = order[~np.isin(order, base_index)]
        chosen = np.concatenate([base_index, extra])[:1200]
        return Frontier(universe[chosen], self.source_names, scores[chosen], ranks[chosen],
                        int(baseline.size))

    def frontiers(self, prefix: Any) -> dict[str, Frontier]:
        prefix.validate()
        integer_ids(prefix.aid)
        if prefix.ts[0] < self.engine.cutoff:
            raise ValueError("Query precedes historical cutoff")
        baseline = self.engine.candidates(prefix, 400).aid
        integer_ids(baseline)
        if baseline.size > 400 or np.unique(baseline).size != baseline.size:
            raise ValueError("Baseline candidate identity is invalid")
        seeds = prefix.aid[-50:][::-1]
        weights = (1 / np.sqrt(np.arange(1, seeds.size + 1))).astype(np.float32)
        first = []
        second = []
        for name, graph in self.graphs:
            valid = seeds < graph.shape[0]
            row = self._aggregate(graph, seeds[valid], weights[valid])
            first.append((name, row))
            order = np.lexsort((row.indices, -row.data))[:self.bridge_per_source]
            bridges = row.indices[order]
            mass = row.data[order].astype(np.float64)
            mass /= max(float(mass.sum()), 1e-12)
            second.append((f"twohop_{name}", self._aggregate(
                graph, bridges, mass.astype(np.float32))))
        one = self._select(prefix, baseline, first)
        two = self._select(prefix, baseline, first + second)
        result = {}
        for name, source, size in ((ARMS[0], one, baseline.size), (ARMS[1], one, 800),
                                   (ARMS[2], one, 1200), (ARMS[3], two, 800),
                                   (ARMS[4], two, 1200)):
            result[name] = Frontier(source.aid[:size], source.source_names,
                                    source.source_scores[:size], source.source_ranks[:size],
                                    source.baseline_size)
            if not np.array_equal(result[name].aid[:baseline.size], baseline):
                raise ValueError("Expansion lost the exact baseline prefix")
        return result


def candidate_ceiling(candidate_ids: np.ndarray, labels: tuple[np.ndarray, ...],
                      denominators: np.ndarray) -> np.ndarray:
    """Post-generation oracle, capped at 20, with the complete target denominator."""
    integer_ids(candidate_ids)
    if np.unique(candidate_ids).size != candidate_ids.size or len(labels) != 3:
        raise ValueError("Candidate/label identity is invalid")
    if denominators.shape != (3,) or denominators.dtype.kind not in "iu":
        raise ValueError("Three integer denominators are required")
    for j, truth in enumerate(labels):
        integer_ids(truth)
        if np.unique(truth).size != truth.size or denominators[j] != min(20, truth.size):
            raise ValueError("Complete distinct target denominator mismatch")
    return np.asarray([min(20, np.isin(truth, candidate_ids).sum()) for truth in labels],
                      dtype=np.int32)
