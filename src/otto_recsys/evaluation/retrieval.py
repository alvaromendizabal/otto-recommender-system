from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

OBJECTIVE_WEIGHTS: dict[str, float] = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


@dataclass(frozen=True)
class RecallResult:
    objective: str
    k: int
    hits: int
    denominator: int
    recall: float


@dataclass(frozen=True)
class IncrementalRecallResult:
    objective: str
    k: int
    base_hits: int
    neural_hits: int
    union_hits: int
    neural_unique_hits: int
    denominator: int
    base_recall: float
    neural_recall: float
    union_recall: float
    incremental_recall: float


def _dedupe_prefix(values: Sequence[int], k: int) -> tuple[int, ...]:
    if k <= 0:
        raise ValueError("k must be positive")
    seen: set[int] = set()
    output: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= k:
            break
    return tuple(output)


def _official_denominator(labels: Iterable[int]) -> int:
    return min(20, len(set(int(value) for value in labels)))


def _candidate_hits(labels: Iterable[int], candidates: Sequence[int], k: int) -> int:
    truth = set(int(value) for value in labels)
    if not truth:
        return 0
    retrieved = set(_dedupe_prefix(candidates, k))
    # Candidate-pool evaluation asks how much final Recall@20 could be recovered
    # by an ideal downstream ranker. The numerator is therefore capped at 20.
    return min(20, len(truth & retrieved))


def candidate_recall_ceiling_at_k(
    *,
    objective: str,
    labels: Mapping[int, Sequence[int]],
    candidates: Mapping[int, Sequence[int]],
    k: int,
) -> RecallResult:
    hits = 0
    denominator = 0
    for session_id, truth in labels.items():
        session_denominator = _official_denominator(truth)
        if session_denominator == 0:
            continue
        denominator += session_denominator
        hits += _candidate_hits(truth, candidates.get(session_id, ()), k)
    recall = hits / denominator if denominator else 0.0
    return RecallResult(
        objective=objective,
        k=k,
        hits=hits,
        denominator=denominator,
        recall=recall,
    )


def incremental_candidate_recall_at_k(
    *,
    objective: str,
    labels: Mapping[int, Sequence[int]],
    base_candidates: Mapping[int, Sequence[int]],
    neural_candidates: Mapping[int, Sequence[int]],
    k: int,
) -> IncrementalRecallResult:
    base_hits = 0
    neural_hits = 0
    union_hits = 0
    neural_unique_hits = 0
    denominator = 0

    for session_id, truth_values in labels.items():
        truth = set(int(value) for value in truth_values)
        session_denominator = min(20, len(truth))
        if session_denominator == 0:
            continue
        denominator += session_denominator

        base = set(_dedupe_prefix(base_candidates.get(session_id, ()), k))
        neural = set(_dedupe_prefix(neural_candidates.get(session_id, ()), k))
        base_truth = truth & base
        neural_truth = truth & neural
        union_truth = truth & (base | neural)

        base_hits += min(20, len(base_truth))
        neural_hits += min(20, len(neural_truth))
        union_hits += min(20, len(union_truth))
        neural_unique_hits += len(neural_truth - base_truth)

    base_recall = base_hits / denominator if denominator else 0.0
    neural_recall = neural_hits / denominator if denominator else 0.0
    union_recall = union_hits / denominator if denominator else 0.0
    return IncrementalRecallResult(
        objective=objective,
        k=k,
        base_hits=base_hits,
        neural_hits=neural_hits,
        union_hits=union_hits,
        neural_unique_hits=neural_unique_hits,
        denominator=denominator,
        base_recall=base_recall,
        neural_recall=neural_recall,
        union_recall=union_recall,
        incremental_recall=union_recall - base_recall,
    )


def weighted_objective_score(
    recalls: Mapping[str, float],
    *,
    weights: Mapping[str, float] = OBJECTIVE_WEIGHTS,
) -> float:
    missing = set(weights) - set(recalls)
    if missing:
        raise ValueError(f"missing objective recalls: {sorted(missing)}")
    weight_sum = float(sum(weights.values()))
    if not np.isclose(weight_sum, 1.0):
        raise ValueError(f"objective weights must sum to 1.0, observed={weight_sum}")
    return float(sum(float(weights[name]) * float(recalls[name]) for name in weights))


def paired_poisson_bootstrap_delta(
    *,
    base_numerators: np.ndarray,
    union_numerators: np.ndarray,
    denominators: np.ndarray,
    iterations: int = 500,
    seed: int = 20260906,
    confidence: float = 0.95,
    chunk_size: int = 16,
) -> tuple[float, float, float]:
    """Scalable paired Poisson-bootstrap CI for a ratio-of-sums recall delta.

    Each row is one session. The same random weight is applied to base, union,
    and denominator, preserving pairing while avoiding an O(iterations * N)
    integer-index matrix in memory. This is a standard large-sample bootstrap
    approximation and is deterministic for a fixed seed.
    """
    base = np.asarray(base_numerators, dtype=np.float64)
    union = np.asarray(union_numerators, dtype=np.float64)
    denom = np.asarray(denominators, dtype=np.float64)
    if base.shape != union.shape or base.shape != denom.shape:
        raise ValueError("base, union, and denominator arrays must have equal shape")
    if base.ndim != 1:
        raise ValueError("bootstrap inputs must be one-dimensional")
    if iterations <= 1:
        raise ValueError("iterations must be greater than 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if float(denom.sum()) <= 0:
        raise ValueError("denominator sum must be positive")

    point = float(union.sum() / denom.sum() - base.sum() / denom.sum())
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    deltas = np.empty(iterations, dtype=np.float64)

    offset = 0
    while offset < iterations:
        width = min(chunk_size, iterations - offset)
        weights = generator.poisson(1.0, size=(width, base.size)).astype(
            np.float64, copy=False
        )
        weighted_denominator = weights @ denom
        valid = weighted_denominator > 0
        if not np.all(valid):
            # Extremely unlikely for realistic OTTO fold sizes, but deterministic
            # retry keeps the estimator well-defined for tiny unit-test inputs.
            for row in np.flatnonzero(~valid):
                while weighted_denominator[row] <= 0:
                    replacement = generator.poisson(1.0, size=base.size).astype(
                        np.float64, copy=False
                    )
                    weights[row] = replacement
                    weighted_denominator[row] = replacement @ denom
        deltas[offset : offset + width] = (
            (weights @ union) / weighted_denominator
            - (weights @ base) / weighted_denominator
        )
        offset += width

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(deltas, alpha))
    upper = float(np.quantile(deltas, 1.0 - alpha))
    return point, lower, upper
