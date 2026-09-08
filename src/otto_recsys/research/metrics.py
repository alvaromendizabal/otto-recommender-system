"""Complete-query ranking statistics and paired session bootstrap intervals."""

from __future__ import annotations

from typing import Any

import numpy as np

from otto_recsys.research.dataset import OBJECTIVES

WEIGHTS = np.array([0.1, 0.3, 0.6])


def ranked_hits(
    scores: np.ndarray, aids: np.ndarray, targets: np.ndarray, groups: np.ndarray, k: int = 20
) -> np.ndarray:
    """Fast rectangular query batches, with a correct variable-size fallback."""
    if (
        scores.shape != aids.shape
        or scores.shape != targets.shape
        or groups.sum() != scores.size
        or (groups < 1).any()
        or not np.isfinite(scores).all()
        or k < 1
    ):
        raise ValueError("scores, candidates, targets and complete query groups must align")
    if np.all(groups == groups[0]):
        size = int(groups[0])
        grid = scores.reshape(-1, size)
        ids = aids.reshape(-1, size)
        order = np.lexsort((ids, -grid), axis=1)[:, :k]
        return np.take_along_axis(targets.reshape(-1, size), order, axis=1).sum(axis=1)
    hits = np.zeros(groups.size, dtype=np.int32)
    position = 0
    for i, count in enumerate(groups):
        stop = position + count
        order = np.lexsort((aids[position:stop], -scores[position:stop]))[:k]
        hits[i] = targets[position:stop][order].sum()
        position = stop
    return hits


def one_query(
    aids: np.ndarray, labels: np.ndarray, scores: np.ndarray, k: int = 20
) -> dict[str, float]:
    """Recall numerator, NDCG, MRR and hit-rate contribution for one query."""
    if len(set(aids)) != aids.size or aids.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("ranked candidates must be unique with aligned finite scores")
    order = np.lexsort((aids, -scores))[:k]
    relevant = np.isin(aids[order], labels)
    positions = np.flatnonzero(relevant)
    discount = 1 / np.log2(np.arange(2, relevant.size + 2))
    ideal = float((1 / np.log2(np.arange(2, min(labels.size, k) + 2))).sum())
    return {
        "hits": int(relevant.sum()),
        "denominator": min(k, labels.size),
        "candidate_hits": min(k, int(np.isin(aids, labels).sum())),
        "ndcg": float((relevant * discount).sum() / ideal) if ideal else 0.0,
        "mrr": float(1 / (positions[0] + 1)) if positions.size else 0.0,
        "hit_rate": float(positions.size > 0),
        "eligible": int(labels.size > 0),
    }


def official_score(hits: np.ndarray, denominators: np.ndarray) -> dict[str, Any]:
    if hits.shape != denominators.shape or hits.ndim != 2 or hits.shape[1] != 3:
        raise ValueError("official statistics require aligned session-by-objective arrays")
    if (hits < 0).any() or (hits > denominators).any() or (denominators < 0).any():
        raise ValueError("official hits must lie within the full capped denominators")
    total = denominators.sum(axis=0)
    if (total == 0).any():
        raise ValueError("each official objective needs a positive pooled denominator")
    recall = hits.sum(axis=0) / total
    return {
        "weighted_recall_at_20": float(recall @ WEIGHTS),
        "objectives": {
            o: {
                "recall_at_20": float(recall[j]),
                "hits": int(hits[:, j].sum()),
                "denominator": int(total[j]),
            }
            for j, o in enumerate(OBJECTIVES)
        },
    }


def paired_bootstrap(
    hits: np.ndarray,
    baseline_hits: np.ndarray,
    denominators: np.ndarray,
    *,
    seed: int,
    replicates: int = 1000,
) -> dict[str, Any]:
    """Resample whole sessions, preserving all three objectives and paired models."""
    official_score(hits, denominators)
    official_score(baseline_hits, denominators)
    if replicates < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    rng = np.random.default_rng(seed)
    n = hits.shape[0]
    point = (hits.sum(axis=0) / denominators.sum(axis=0)) @ WEIGHTS
    base = (baseline_hits.sum(axis=0) / denominators.sum(axis=0)) @ WEIGHTS
    scores, differences = [], []
    for _ in range(replicates):
        indices = rng.integers(0, n, size=n)
        den = denominators[indices].sum(axis=0)
        if (den == 0).any():
            continue
        value = (hits[indices].sum(axis=0) / den) @ WEIGHTS
        difference = (
            (hits[indices].sum(axis=0) - baseline_hits[indices].sum(axis=0)) / den
        ) @ WEIGHTS
        scores.append(value)
        differences.append(difference)
    if len(scores) < replicates * 0.95:
        raise ValueError("too many bootstrap samples lack an objective denominator")
    return {
        "method": "paired nonparametric session bootstrap, percentile 95% intervals",
        "seed": seed,
        "requested_replicates": replicates,
        "valid_replicates": len(scores),
        "score": float(point),
        "score_interval": np.quantile(scores, [0.025, 0.975]).tolist(),
        "absolute_gain": float(point - base),
        "gain_interval": np.quantile(differences, [0.025, 0.975]).tolist(),
        "scope": "conditional on this frozen model and temporal cohort; no training uncertainty",
    }
