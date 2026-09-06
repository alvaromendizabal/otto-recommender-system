"""Official OTTO Recall@20 and explicitly labeled ranking diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np

WEIGHTS = np.array([0.1, 0.3, 0.6])
OBJECTIVES = ("clicks", "carts", "orders")


def ranking_counts(truth: set[int], ranked: list[int]) -> np.ndarray:
    if len(ranked) != len(set(ranked)) or any(aid < 0 for aid in ranked):
        raise ValueError("predictions must have unique nonnegative item IDs")
    hits = np.array([aid in truth for aid in ranked[:20]], dtype=np.float64)
    denominator = min(20, len(truth))
    dcg = float((hits / np.log2(np.arange(len(hits)) + 2)).sum())
    ideal = float((1 / np.log2(np.arange(denominator) + 2)).sum())
    positions = np.flatnonzero(hits)
    return np.array(
        [
            denominator,
            hits.sum(),
            int(bool(truth)),
            dcg / ideal if ideal else 0,
            1 / (positions[0] + 1) if len(positions) else 0,
            int(bool(len(positions))),
            hits.sum() / 20,
        ],
        dtype=np.float64,
    )


def summarize_ranking(counts: np.ndarray) -> dict[str, Any]:
    if counts.ndim != 3 or counts.shape[1:] != (3, 7) or np.any(~np.isfinite(counts)):
        raise ValueError("invalid ranking counts")
    totals = counts.sum(axis=0)
    if np.any(totals[:, 0] <= 0):
        raise ValueError("every objective needs labels for the weighted metric")
    objectives = {}
    for i, name in enumerate(OBJECTIVES):
        row = totals[i]
        objectives[name] = {
            "recall_at_20": float(row[1] / row[0]),
            "hits_at_20": int(row[1]),
            "capped_denominator": int(row[0]),
            "labeled_sessions": int(row[2]),
            "ndcg_at_20": float(row[3] / row[2]),
            "mrr_at_20": float(row[4] / row[2]),
            "hit_rate_at_20": float(row[5] / row[2]),
            "precision_at_20": float(row[6] / row[2]),
        }
    return {
        "weighted_recall_at_20": float((totals[:, 1] / totals[:, 0]) @ WEIGHTS),
        "sessions": len(counts),
        "objectives": objectives,
        "diagnostic_averaging": (
            "NDCG/MRR/hit-rate/precision: mean over labeled sessions per objective"
        ),
        "official_metric": "0.10*Recall@20(clicks)+0.30*Recall@20(carts)+0.60*Recall@20(orders)",
    }


def paired_recall_interval(
    exact: np.ndarray, approximate: np.ndarray, *, iterations: int = 500, seed: int = 20260906
) -> dict[str, Any]:
    if exact.shape != approximate.shape or iterations < 2:
        raise ValueError("paired ranking arrays must match")
    if not np.array_equal(exact[:, :, 0], approximate[:, :, 0]):
        raise ValueError("paired ranking denominators differ")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(iterations):
        sampled = rng.integers(0, len(exact), size=len(exact))
        denom = exact[sampled, :, 0].sum(axis=0)
        if np.any(denom == 0):
            raise ValueError("bootstrap cohort lacks objective labels")
        values.append(
            float(
                ((approximate[sampled, :, 1] - exact[sampled, :, 1]).sum(axis=0) / denom) @ WEIGHTS
            )
        )
    return {
        "unit": "paired session",
        "iterations": iterations,
        "seed": seed,
        "weighted_recall_at_20_delta_ci95": np.quantile(values, [0.025, 0.975]).tolist(),
    }
