"""Label-blind nested candidates; reuse certified graphs without fitting new ones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from otto_recsys.research.features import FeatureEngine, Prefix
    from otto_recsys.research.graph_signals import GraphSignals

CHANNELS = ("time", "cart", "order")
ARMS = {
    "baseline_400": (400, 1),
    "wide_onehop_800": (800, 1),
    "wide_onehop_1200": (1200, 1),
    "wide_twohop_800": (800, 2),
    "wide_twohop_1200": (1200, 2),
}


def integer_ids(values: np.ndarray, *, unique: bool = False) -> np.ndarray:
    """Reject floating IDs, duplicates where forbidden, and silent int64 overflow."""
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError("IDs must be a one-dimensional integer array")
    if array.size and (array.min() < 0 or int(array.max()) > np.iinfo(np.int64).max):
        raise ValueError("IDs must be nonnegative and int64 representable")
    if unique and np.unique(array).size != array.size:
        raise ValueError("IDs must be unique")
    return array.astype(np.int64, copy=False)


def source_ranks(scores: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Positive evidence only; descending score, ascending item ID for exact ties."""
    order = np.lexsort((ids, -scores))
    positive = order[scores[order] > 0]
    result = np.zeros(scores.size, dtype=np.float32)
    result[positive] = np.arange(1, positive.size + 1, dtype=np.float32)
    return result


@dataclass(frozen=True)
class Frontier:
    aid: np.ndarray
    source_names: tuple[str, ...]
    source_scores: np.ndarray
    source_ranks: np.ndarray
    baseline_size: int
    evidence_available: bool = True


class CandidateFrontier:
    """Keep the exact baseline prefix; change neither legacy candidates nor rankers."""

    def __init__(
        self,
        engine: FeatureEngine,
        wide: dict[str, GraphSignals],
        *,
        bridge_per_source: int = 64,
        rrf_k: float = 20.0,
        max_discovery_items: int = 250_000,
    ) -> None:
        if type(bridge_per_source) is not int or not 1 <= bridge_per_source <= 64:
            raise ValueError("bridge_per_source must be an integer in 1..64")
        if not np.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError("RRF offset must be finite and positive")
        if type(max_discovery_items) is not int or max_discovery_items < 1600:
            raise ValueError("discovery cap must be an integer of at least 1600")
        if set(wide) != {"symmetric", "forward"}:
            raise ValueError("both symmetric and forward graphs are required")
        if any(value.cutoff != engine.cutoff for value in wide.values()):
            raise ValueError("all graphs must share the baseline cutoff")
        self.engine = engine
        self.bridge_per_source = bridge_per_source
        self.rrf_k = float(rrf_k)
        self.max_discovery_items = max_discovery_items
        self.graph_sources: list[tuple[str, csr_matrix]] = [
            (f"base_{channel}", graph)
            for channel, graph in zip(CHANNELS, engine.graphs, strict=True)
        ]
        for family in ("symmetric", "forward"):
            self.graph_sources.extend(
                (f"{family}_{channel}", graph)
                for channel, graph in zip(CHANNELS, wide[family].graphs, strict=True)
            )
        if any(graph.shape[0] != graph.shape[1] for _, graph in self.graph_sources):
            raise ValueError("item graphs must be square")
        names = tuple(name for name, _ in self.graph_sources)
        self.source_names = (*names, "revisit", "popularity", *(f"twohop_{n}" for n in names))
        # Fixed preregistered weights, not parameters optimized against pilot labels.
        self.weights = np.asarray(
            [0.4, 0.3, 0.6, 0.5, 0.4, 0.7, 0.55, 0.45, 0.8, 1.0, 0.01,
             0.15, 0.12, 0.2, 0.2, 0.16, 0.28, 0.22, 0.18, 0.32], dtype=np.float32
        )

    @staticmethod
    def _aggregate(graph: csr_matrix, ids: np.ndarray, weights: np.ndarray) -> csr_matrix:
        if not ids.size:
            return csr_matrix((1, graph.shape[1]), dtype=np.float32)
        rows = graph[ids]
        if not np.isfinite(rows.data).all() or (rows.data < 0).any():
            raise ValueError("encountered nonfinite or negative graph evidence")
        result = (csr_matrix(weights.reshape(1, -1).astype(np.float32)) @ rows).tocsr()
        result.sum_duplicates()
        result.eliminate_zeros()
        result.sort_indices()
        if not np.isfinite(result.data).all():
            raise ValueError("graph aggregation overflow")
        return result

    def _evidence(
        self, prefix: Prefix, *, twohop: bool
    ) -> list[tuple[str, np.ndarray, np.ndarray]]:
        seeds = integer_ids(prefix.aid)[-50:][::-1]
        weights = (1 / np.sqrt(np.arange(1, seeds.size + 1))).astype(np.float32)
        evidence = []
        for name, graph in self.graph_sources:
            valid = seeds < graph.shape[0]
            first = self._aggregate(graph, seeds[valid], weights[valid])
            evidence.append((name, first.indices, first.data))
            if twohop:
                order = np.lexsort((first.indices, -first.data))[:self.bridge_per_source]
                bridge_weights = first.data[order].astype(np.float64)
                bridge_weights /= max(float(bridge_weights.sum()), 1e-12)
                second = self._aggregate(graph, first.indices[order], bridge_weights)
                evidence.append((f"twohop_{name}", second.indices, second.data))
        return evidence

    def _expand(
        self, prefix: Prefix, baseline: np.ndarray, *, budget: int, hops: int
    ) -> Frontier:
        evidence = self._evidence(prefix, twohop=hops == 2)
        popular = [integer_ids(ids, unique=True) for ids in self.engine.popular]
        if len(popular) != 3:
            raise ValueError("three objective popularity lists are required")
        discovery = np.unique(np.concatenate(
            [baseline, prefix.aid, *popular, *[ids for _, ids, _ in evidence]]
        ))
        if discovery.size > self.max_discovery_items:
            raise ValueError("discovery resource cap exceeded; no silent truncation")
        scores = np.zeros((discovery.size, len(self.source_names)), dtype=np.float32)
        lookup = {name: i for i, name in enumerate(self.source_names)}
        for name, ids, values in evidence:
            scores[np.searchsorted(discovery, ids), lookup[name]] = values
        np.add.at(
            scores[:, lookup["revisit"]], np.searchsorted(discovery, prefix.aid[::-1]),
            (1 / np.sqrt(np.arange(1, prefix.aid.size + 1))).astype(np.float32),
        )
        for weight, ids in zip((0.1, 0.3, 0.6), popular, strict=True):
            scores[np.searchsorted(discovery, ids), lookup["popularity"]] += (
                weight / np.arange(1, ids.size + 1)
            ).astype(np.float32)
        ranks = np.column_stack([source_ranks(scores[:, j], discovery)
                                 for j in range(scores.shape[1])])
        fusion = np.where(ranks > 0, self.weights / (self.rrf_k + ranks), 0).sum(axis=1)
        base_indices = np.searchsorted(discovery, baseline)
        order = np.lexsort((discovery, -fusion))
        chosen = np.r_[base_indices, order[~np.isin(order, base_indices)]][:budget]
        return Frontier(discovery[chosen], self.source_names, scores[chosen], ranks[chosen],
                        baseline.size)

    def candidates(self, prefix: Prefix, *, budget: int, hops: int) -> Frontier:
        prefix.validate()
        integer_ids(prefix.aid)
        if prefix.ts[0] < self.engine.cutoff:
            raise ValueError("query precedes historical cutoff")
        if type(budget) is not int or not 400 <= budget <= 1600 or type(hops) is not int:
            raise ValueError("budget must be an integer in 400..1600; hops must be 1 or 2")
        if hops not in (1, 2):
            raise ValueError("hops must be 1 or 2")
        baseline = integer_ids(self.engine.candidates(prefix, 400).aid, unique=True).copy()
        if not 1 <= baseline.size <= 400:
            raise ValueError("baseline must contain 1..400 unique candidates")
        if budget == 400:
            zeros = np.zeros((baseline.size, len(self.source_names)), dtype=np.float32)
            # No source evidence is fabricated for this baseline identity-only control.
            return Frontier(baseline, self.source_names, zeros, zeros.copy(), baseline.size, False)
        return self._expand(prefix, baseline, budget=budget, hops=hops)

    def frontiers(self, prefix: Prefix) -> dict[str, Frontier]:
        """Build each 1200 frontier once; 800 is its exact prefix, not another graph pass."""
        baseline = self.candidates(prefix, budget=400, hops=1)
        result = {"baseline_400": baseline}
        for name, hops in (("onehop", 1), ("twohop", 2)):
            large = self._expand(prefix, baseline.aid, budget=1200, hops=hops)
            result[f"wide_{name}_1200"] = large
            result[f"wide_{name}_800"] = Frontier(
                large.aid[:800], large.source_names, large.source_scores[:800],
                large.source_ranks[:800], baseline.aid.size,
            )
        return {arm: result[arm] for arm in ARMS}


def candidate_ceiling(
    candidate_ids: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    denominators: np.ndarray,
) -> np.ndarray:
    """Oracle hits@20, not achieved Recall@20; reject inconsistent truth ledgers."""
    candidates = integer_ids(candidate_ids, unique=True)
    denominator = integer_ids(denominators)
    if len(labels) != 3 or denominator.shape != (3,):
        raise ValueError("three objective labels and denominators are required")
    truth = [integer_ids(label, unique=True) for label in labels]
    if truth[0].size > 1:
        raise ValueError("next-click truth must contain at most one item")
    expected = np.asarray([min(20, label.size) for label in truth], dtype=np.int64)
    if not np.array_equal(expected, denominator):
        raise ValueError("denominators must equal min(20, full distinct truth count)")
    return np.asarray([min(20, int(np.isin(label, candidates).sum())) for label in truth],
                      dtype=np.int32)
