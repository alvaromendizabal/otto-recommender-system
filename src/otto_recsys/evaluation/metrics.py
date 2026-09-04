from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

ACTION_WEIGHTS: dict[str, float] = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


def recall_at_k(
    predictions: Mapping[int, Sequence[int]],
    targets: Mapping[int, Iterable[int]],
    *,
    k: int = 20,
) -> float:
    """Compute OTTO-style micro Recall@K."""
    if k <= 0:
        raise ValueError("k must be positive")

    hits = 0
    denominator = 0

    for session, target_iter in targets.items():
        target_items = set(target_iter)
        if not target_items:
            continue

        predicted_items = set(predictions.get(session, ())[:k])
        hits += len(predicted_items & target_items)
        denominator += min(k, len(target_items))

    return hits / denominator if denominator else 0.0


def weighted_recall_at_k(
    predictions_by_action: Mapping[str, Mapping[int, Sequence[int]]],
    targets_by_action: Mapping[str, Mapping[int, Iterable[int]]],
    *,
    k: int = 20,
) -> tuple[float, dict[str, float]]:
    """Compute weighted OTTO Recall@K and per-action recalls."""
    detail: dict[str, float] = {}
    score = 0.0

    for action, weight in ACTION_WEIGHTS.items():
        action_recall = recall_at_k(
            predictions_by_action.get(action, {}),
            targets_by_action.get(action, {}),
            k=k,
        )
        detail[action] = action_recall
        score += weight * action_recall

    return score, detail
