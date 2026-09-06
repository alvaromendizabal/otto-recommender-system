"""Leakage-safe evaluation utilities for OTTO retrieval and ranking."""

from .retrieval import (
    OBJECTIVE_WEIGHTS,
    IncrementalRecallResult,
    RecallResult,
    candidate_recall_ceiling_at_k,
    incremental_candidate_recall_at_k,
    paired_poisson_bootstrap_delta,
    weighted_objective_score,
)

__all__ = [
    "OBJECTIVE_WEIGHTS",
    "IncrementalRecallResult",
    "RecallResult",
    "candidate_recall_ceiling_at_k",
    "incremental_candidate_recall_at_k",
    "paired_poisson_bootstrap_delta",
    "weighted_objective_score",
]
