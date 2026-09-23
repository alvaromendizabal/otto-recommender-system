"""Candidate-preserving neural augmentation used by the OTTO frontier experiment."""
from __future__ import annotations

import numpy as np

NEURAL_FEATURES = (
    "nn_order_cosine", "nn_order_reciprocal_rank", "nn_order_retrieved",
    "nn_order_max_cosine", "nn_order_margin_to_max", "nn_candidate_known",
    "nn_query_known", "nn_in_original_pool",
)


def combine_candidates(original: np.ndarray, neural: np.ndarray, padding_id: int) -> np.ndarray:
    """Append novel neural items without discarding or reordering incumbent candidates."""
    original = np.asarray(original, dtype=np.int64)
    neural = np.asarray(neural, dtype=np.int64)
    if len(np.unique(original)) != len(original) or np.any(original < 0):
        raise ValueError("incumbent candidate IDs must be unique and nonnegative")
    seen = set(map(int, original))
    novel: list[int] = []
    for item in neural:
        value = int(item)
        if 0 <= value < padding_id and value not in seen:
            novel.append(value)
            seen.add(value)
    result = np.r_[original, np.asarray(novel, dtype=np.int64)]
    if not np.array_equal(result[: len(original)], original):
        raise AssertionError("candidate augmentation changed the incumbent prefix")
    return result


def neural_features(chosen: np.ndarray, incumbent: np.ndarray, query: np.ndarray,
                    vectors: np.ndarray, top_ids: np.ndarray, top_scores: np.ndarray) -> np.ndarray:
    """Create eight neural signals aligned to an arbitrary candidate list."""
    chosen = np.asarray(chosen, dtype=np.int64)
    safe = np.minimum(chosen, len(vectors) - 1)
    candidate_vectors = np.asarray(vectors[safe])
    known = (chosen < len(vectors) - 1) & (np.linalg.norm(candidate_vectors, axis=1) > 0)
    cosine = candidate_vectors @ query
    cosine[~known] = 0
    reciprocal_rank = {int(item): 1 / (i + 1) for i, item in enumerate(top_ids) if int(item) < len(vectors) - 1}
    rr = np.asarray([reciprocal_rank.get(int(item), 0.0) for item in chosen], dtype=np.float32)
    maximum = float(np.max(top_scores))
    result = np.column_stack((cosine, rr, rr > 0, np.full(len(chosen), maximum),
                              cosine - maximum, known, np.full(len(chosen), np.linalg.norm(query) > 0),
                              np.isin(chosen, incumbent))).astype(np.float32)
    if result.shape != (len(chosen), len(NEURAL_FEATURES)) or not np.isfinite(result).all():
        raise ValueError("invalid neural feature matrix")
    return result
