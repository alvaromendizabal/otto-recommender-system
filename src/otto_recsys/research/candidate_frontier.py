"""Label-blind nested candidate frontiers from cutoff-safe one/two-hop graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore[import-untyped]

from otto_recsys.research.features import FeatureEngine, Prefix, ranks
from otto_recsys.research.graph_signals import CHANNELS, GraphSignals


@dataclass(frozen=True)
class Frontier:
    """Expanded candidate IDs plus label-blind source evidence."""

    aid: np.ndarray
    source_names: tuple[str, ...]
    source_scores: np.ndarray
    source_ranks: np.ndarray
    baseline_size: int


class CandidateFrontier:
    """Preserve the exact 400-candidate baseline, then add historical graph evidence."""

    def __init__(
        self,
        engine: FeatureEngine,
        wide: dict[str, GraphSignals],
        *,
        bridge_per_source: int = 64,
        rrf_k: float = 20.0,
        source_weights: dict[str, float] | None = None,
    ) -> None:
        if bridge_per_source < 1 or not np.isfinite(rrf_k) or rrf_k <= 0:
            raise ValueError("frontier bridge and RRF settings must be positive")
        if set(wide) != {"symmetric", "forward"}:
            raise ValueError("frontier requires certified symmetric and forward wide graphs")
        if any(graph.cutoff != engine.cutoff for graph in wide.values()):
            raise ValueError("all frontier graphs must share the baseline historical cutoff")
        self.engine = engine
        self.wide = wide
        self.bridge_per_source = bridge_per_source
        self.rrf_k = float(rrf_k)
        self.graph_sources: list[tuple[str, csr_matrix]] = [
            *(f"base_{channel}", graph)
            for channel, graph in zip(CHANNELS, engine.graphs, strict=True)
        ]
        for family in ("symmetric", "forward"):
            self.graph_sources.extend(
                (f"{family}_{channel}", graph)
                for channel, graph in zip(CHANNELS, wide[family].graphs, strict=True)
            )
        self.onehop_names = tuple(name for name, _ in self.graph_sources)
        self.twohop_names = tuple(f"twohop_{name}" for name in self.onehop_names)
        self.source_names = (*self.onehop_names, "revisit", "popularity", *self.twohop_names)
        default = {
            **{f"base_{c}": w for c, w in zip(CHANNELS, (0.4, 0.3, 0.6), strict=True)},
            **{f"symmetric_{c}": w for c, w in zip(CHANNELS, (0.5, 0.4, 0.7), strict=True)},
            **{f"forward_{c}": w for c, w in zip(CHANNELS, (0.55, 0.45, 0.8), strict=True)},
            "revisit": 1.0,
            "popularity": 0.01,
            **{f"twohop_base_{c}": w for c, w in zip(CHANNELS, (0.15, 0.12, 0.2), strict=True)},
            **{
                f"twohop_symmetric_{c}": w
                for c, w in zip(CHANNELS, (0.2, 0.16, 0.28), strict=True)
            },
            **{
                f"twohop_forward_{c}": w
                for c, w in zip(CHANNELS, (0.22, 0.18, 0.32), strict=True)
            },
        }
        self.source_weights = default if source_weights is None else source_weights
        if set(self.source_weights) != set(self.source_names) or any(
            not np.isfinite(value) or value < 0 for value in self.source_weights.values()
        ):
            raise ValueError("frontier source weights must cover every source with finite nonnegative values")

    @staticmethod
    def _aggregate(graph: csr_matrix, ids: np.ndarray, weights: np.ndarray) -> csr_matrix:
        if not ids.size:
            return csr_matrix((1, graph.shape[1]), dtype=np.float32)
        result = csr_matrix(weights.reshape(1, -1).astype(np.float32)) @ graph[ids]
        result.eliminate_zeros()
        return result.tocsr()

    def _graph_evidence(
        self, prefix: Prefix, *, include_twohop: bool
    ) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[np.ndarray]]:
        seeds = prefix.aid[-50:][::-1]
        recency = (1 / np.sqrt(np.arange(1, seeds.size + 1))).astype(np.float32)
        evidence: list[tuple[str, np.ndarray, np.ndarray]] = []
        discoveries: list[np.ndarray] = []
        first_hop: list[tuple[str, csr_matrix, csr_matrix]] = []
        for name, graph in self.graph_sources:
            valid = seeds < graph.shape[0]
            aggregate = self._aggregate(graph, seeds[valid], recency[valid])
            evidence.append((name, aggregate.indices.copy(), aggregate.data.copy()))
            discoveries.append(aggregate.indices.copy())
            first_hop.append((name, graph, aggregate))
        if include_twohop:
            for name, graph, aggregate in first_hop:
                if aggregate.nnz:
                    order = np.lexsort((aggregate.indices, -aggregate.data))[: self.bridge_per_source]
                    bridge_ids = aggregate.indices[order]
                    bridge_weight = aggregate.data[order].astype(np.float64)
                    bridge_weight /= max(float(bridge_weight.sum()), 1e-12)
                    second = self._aggregate(
                        graph,
                        bridge_ids,
                        bridge_weight.astype(np.float32),
                    )
                else:
                    second = csr_matrix((1, graph.shape[1]), dtype=np.float32)
                evidence.append(
                    (f"twohop_{name}", second.indices.copy(), second.data.copy())
                )
                discoveries.append(second.indices.copy())
        return evidence, discoveries

    def candidates(self, prefix: Prefix, *, budget: int, hops: int) -> Frontier:
        """Exact baseline at 400; larger frontiers are strict nested supersets when available."""
        prefix.validate()
        if prefix.ts[0] < self.engine.cutoff:
            raise ValueError("query precedes the fitted historical cutoff")
        if not 400 <= budget <= 1600 or hops not in (1, 2):
            raise ValueError("frontier budget must be 400..1600 and hops must be 1 or 2")
        baseline = self.engine.candidates(prefix, 400)
        if budget == 400:
            zeros = np.zeros((baseline.aid.size, len(self.source_names)), dtype=np.float32)
            return Frontier(
                baseline.aid.copy(), self.source_names, zeros, zeros.copy(), baseline.aid.size
            )

        evidence, discoveries = self._graph_evidence(prefix, include_twohop=hops == 2)
        discovery = np.unique(
            np.concatenate(
                [
                    baseline.aid,
                    prefix.aid,
                    *self.engine.popular,
                    *discoveries,
                ]
            )
        )
        scores = np.zeros((discovery.size, len(self.source_names)), dtype=np.float32)
        source_index = {name: i for i, name in enumerate(self.source_names)}
        for name, ids, values in evidence:
            if ids.size:
                present = np.searchsorted(discovery, ids)
                scores[present, source_index[name]] = values
        revisit = source_index["revisit"]
        np.add.at(
            scores[:, revisit],
            np.searchsorted(discovery, prefix.aid[::-1]),
            (1 / np.sqrt(np.arange(1, prefix.aid.size + 1))).astype(np.float32),
        )
        popularity = source_index["popularity"]
        for weight, popular in zip((0.1, 0.3, 0.6), self.engine.popular, strict=True):
            scores[np.searchsorted(discovery, popular), popularity] += (
                weight / np.arange(1, popular.size + 1)
            ).astype(np.float32)
        source_ranks = np.column_stack(
            [ranks(scores[:, j], discovery) for j in range(scores.shape[1])]
        )
        weights = np.asarray(
            [self.source_weights[name] for name in self.source_names], dtype=np.float32
        )
        fusion = np.sum(
            np.where(source_ranks > 0, weights / (self.rrf_k + source_ranks), 0), axis=1
        )
        baseline_index = np.searchsorted(discovery, baseline.aid)
        order = np.lexsort((discovery, -fusion))
        extras = order[~np.isin(order, baseline_index)]
        selected = np.concatenate([baseline_index, extras])[:budget]
        aid = discovery[selected]
        if (
            np.unique(aid).size != aid.size
            or not np.array_equal(aid[: baseline.aid.size], baseline.aid)
        ):
            raise ValueError("expanded frontier must preserve the exact baseline candidate prefix")
        return Frontier(
            aid,
            self.source_names,
            scores[selected],
            source_ranks[selected],
            baseline.aid.size,
        )


def candidate_ceiling(
    candidate_ids: np.ndarray,
    labels: tuple[np.ndarray, np.ndarray, np.ndarray],
    denominators: np.ndarray,
) -> np.ndarray:
    """Metric-only ceiling after label-blind candidate construction."""
    if candidate_ids.ndim != 1 or np.unique(candidate_ids).size != candidate_ids.size:
        raise ValueError("candidate ceiling requires unique one-dimensional candidate IDs")
    if denominators.shape != (3,) or (denominators < 0).any():
        raise ValueError("candidate ceiling requires three nonnegative pooled denominators")
    hits = np.asarray(
        [min(20, int(np.isin(label, candidate_ids).sum())) for label in labels],
        dtype=np.int32,
    )
    if (hits > denominators).any():
        raise ValueError("candidate ceiling hits cannot exceed capped denominators")
    return hits
