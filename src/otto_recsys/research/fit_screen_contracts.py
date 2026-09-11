"""Pure contracts for fitting-only shared-feature OOF screening."""

from __future__ import annotations

from typing import Any

import numpy as np

from otto_recsys.research.dataset import session_hash


def fold_assignment(session_ids: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    ids = np.asarray(session_ids, dtype=np.int64)
    if ids.ndim != 1 or ids.size < folds or folds < 2 or np.unique(ids).size != ids.size:
        raise ValueError("fold assignment requires unique sessions and at least two folds")
    fold = (session_hash(ids, seed) % np.uint64(folds)).astype(np.int8)
    if set(fold.tolist()) != set(range(folds)):
        raise ValueError("every fitting-only fold must be represented")
    return fold


def advancement(
    baseline: dict[str, Any],
    shared: dict[str, Any],
    fold_gains: list[float],
    *,
    minimum_gain: float,
    minimum_nonnegative_folds: int,
) -> dict[str, Any]:
    if (
        minimum_gain < 0
        or not np.isfinite(minimum_gain)
        or not fold_gains
        or not np.isfinite(fold_gains).all()
        or not 1 <= minimum_nonnegative_folds <= len(fold_gains)
    ):
        raise ValueError("fitting-only advancement settings are invalid")
    weighted_gain = (
        float(shared["weighted_recall_at_20"])
        - float(baseline["weighted_recall_at_20"])
    )
    order_gain = (
        float(shared["objectives"]["orders"]["recall_at_20"])
        - float(baseline["objectives"]["orders"]["recall_at_20"])
    )
    nonnegative = sum(value >= 0 for value in fold_gains)
    return {
        "advance_to_fitting_only_block_ablation": bool(
            weighted_gain >= minimum_gain
            and order_gain >= 0
            and nonnegative >= minimum_nonnegative_folds
        ),
        "weighted_gain": weighted_gain,
        "order_gain": order_gain,
        "fold_weighted_gains": [float(x) for x in fold_gains],
        "nonnegative_folds": nonnegative,
        "required_nonnegative_folds": minimum_nonnegative_folds,
        "minimum_weighted_gain": minimum_gain,
        "scope": (
            "Fitting-only OOF development gate. Passing permits fitting-only "
            "family ablation, not selection access or promotion."
        ),
    }
