"""Leakage-safe query-relative as-of demand features for OTTO ranking.

This module consumes a *precomputed* demand snapshot whose counts include only
events strictly before ``cutoff_ts``. It does not read labels or future suffixes.

The initial family represents:
- recent item demand by action over nested 1/6/24/72/168-hour windows;
- relative demand among the full candidate set before negative subsampling;
- short-vs-long demand drift;
- support-smoothed action mix (descriptive intensity, not causal conversion);
- action-specific last-seen recency.

The snapshot builder is deliberately separate: temporal source provenance must be
verified before a snapshot can be admitted to a fitting/selection experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

ACTIONS = ("clicks", "carts", "orders")
WINDOW_HOURS = (1, 6, 24, 72, 168)
TREND_PAIRS = ((1, 24), (6, 24), (24, 168))
MIX_WINDOWS = (24, 168)

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class DemandSnapshot:
    """One strict as-of cutoff with nested item/action window counts."""

    cutoff_ts: int
    aids: IntArray
    counts: FloatArray  # item, window, action
    last_event_ts: IntArray  # item, action; -1 means not seen before cutoff
    source_id: str

    def validate(self) -> None:
        if self.cutoff_ts <= 0 or not self.source_id:
            raise ValueError("snapshot cutoff and source identity are required")
        if (
            self.aids.ndim != 1
            or self.counts.shape != (self.aids.size, len(WINDOW_HOURS), len(ACTIONS))
            or self.last_event_ts.shape != (self.aids.size, len(ACTIONS))
        ):
            raise ValueError("snapshot arrays have incompatible shapes")
        if self.aids.dtype.kind not in "iu" or self.last_event_ts.dtype.kind not in "iu":
            raise ValueError("snapshot identifiers/timestamps must be integer arrays")
        if self.counts.dtype.kind not in "fiu":
            raise ValueError("snapshot counts must be numeric")
        if (
            (self.aids < 0).any()
            or (np.diff(self.aids) <= 0).any()
            or not np.isfinite(self.counts).all()
            or (self.counts < 0).any()
        ):
            raise ValueError("snapshot identifiers or counts are invalid")
        if (np.diff(self.counts, axis=1) < -1e-6).any():
            raise ValueError("nested as-of window counts are not monotone")
        observed_ts = self.last_event_ts[self.last_event_ts >= 0]
        if observed_ts.size and observed_ts.max() >= self.cutoff_ts:
            raise ValueError("snapshot last-event timestamp reaches or exceeds cutoff")


def _window_index(hours: int) -> int:
    try:
        return WINDOW_HOURS.index(hours)
    except ValueError as error:
        raise ValueError(f"unsupported demand window: {hours}") from error


def _candidate_percentile(values: FloatArray) -> FloatArray:
    """Tie-aware empirical percentile: high demand -> high value, all ties -> 0.5."""
    n = values.size
    if n == 0:
        return np.empty(0, dtype=np.float32)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float32)
    result = np.empty(n, dtype=np.float32)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    cumulative_less = np.cumsum(counts) - counts
    mid = cumulative_less + (counts - 1) / 2
    mapped = mid / (n - 1)
    result[:] = mapped[inverse]
    return result


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for action in ACTIONS:
        for hours in WINDOW_HOURS:
            stem = f"asof_{action}_h{hours}"
            names.extend(
                (
                    f"{stem}_log_count",
                    f"{stem}_candidate_percentile",
                    f"{stem}_candidate_mass_share",
                )
            )
        names.extend(
            (
                f"asof_{action}_last_seen",
                f"asof_{action}_last_age_log_hours",
            )
        )
        for short, long in TREND_PAIRS:
            names.extend(
                (
                    f"asof_{action}_trend_h{short}_h{long}",
                    f"asof_{action}_percentile_delta_h{short}_h{long}",
                )
            )
    for hours in MIX_WINDOWS:
        for action in ACTIONS:
            names.append(f"asof_action_mix_{action}_h{hours}")
    return tuple(names)


def _lookup(snapshot: DemandSnapshot, candidate_aids: IntArray) -> tuple[FloatArray, IntArray]:
    if candidate_aids.ndim != 1 or candidate_aids.dtype.kind not in "iu":
        raise ValueError("candidate identifiers must be a one-dimensional integer array")
    if (candidate_aids < 0).any() or np.unique(candidate_aids).size != candidate_aids.size:
        raise ValueError("candidate identifiers must be unique nonnegative integers")
    positions = np.searchsorted(snapshot.aids, candidate_aids)
    known = positions < snapshot.aids.size
    if snapshot.aids.size:
        known &= snapshot.aids[np.minimum(positions, snapshot.aids.size - 1)] == candidate_aids
    safe = np.minimum(positions, max(snapshot.aids.size - 1, 0))
    counts = np.zeros((candidate_aids.size, len(WINDOW_HOURS), len(ACTIONS)), dtype=np.float32)
    last = np.full((candidate_aids.size, len(ACTIONS)), -1, dtype=np.int64)
    if snapshot.aids.size:
        counts[known] = snapshot.counts[safe[known]].astype(np.float32)
        last[known] = snapshot.last_event_ts[safe[known]]
    return counts, last


def transform(
    snapshot: DemandSnapshot,
    candidate_aids: IntArray,
    *,
    query_ts: int,
    prior_strength: float = 20.0,
) -> FloatArray:
    """Create the complete candidate matrix before target-dependent subsampling."""
    snapshot.validate()
    if query_ts < snapshot.cutoff_ts:
        raise ValueError("query precedes the certified demand snapshot cutoff")
    if prior_strength <= 0 or not np.isfinite(prior_strength):
        raise ValueError("prior_strength must be a positive finite value")

    counts, last = _lookup(snapshot, candidate_aids)
    columns: dict[str, FloatArray] = {}
    percentiles: dict[tuple[int, int], FloatArray] = {}

    for ai, action in enumerate(ACTIONS):
        for wi, hours in enumerate(WINDOW_HOURS):
            values = counts[:, wi, ai]
            stem = f"asof_{action}_h{hours}"
            columns[f"{stem}_log_count"] = np.log1p(values).astype(np.float32)
            percentile = _candidate_percentile(values)
            percentiles[(ai, hours)] = percentile
            columns[f"{stem}_candidate_percentile"] = percentile
            mass = float(values.sum())
            columns[f"{stem}_candidate_mass_share"] = (
                values / mass if mass > 0 else np.zeros_like(values)
            ).astype(np.float32)

        seen = last[:, ai] >= 0
        age_hours = np.zeros(candidate_aids.size, dtype=np.float32)
        age_hours[seen] = ((query_ts - last[seen, ai]) / 3_600_000).astype(np.float32)
        if (age_hours[seen] < 0).any():
            raise ValueError("candidate last-event timestamp is after the query")
        columns[f"asof_{action}_last_seen"] = seen.astype(np.float32)
        columns[f"asof_{action}_last_age_log_hours"] = np.where(
            seen, np.log1p(age_hours), 0.0
        ).astype(np.float32)

        for short, long in TREND_PAIRS:
            short_values = counts[:, _window_index(short), ai]
            long_values = counts[:, _window_index(long), ai]
            short_rate = short_values / short
            long_rate = long_values / long
            columns[f"asof_{action}_trend_h{short}_h{long}"] = (
                np.log1p(short_rate) - np.log1p(long_rate)
            ).astype(np.float32)
            columns[f"asof_{action}_percentile_delta_h{short}_h{long}"] = (
                percentiles[(ai, short)] - percentiles[(ai, long)]
            ).astype(np.float32)

    for hours in MIX_WINDOWS:
        wi = _window_index(hours)
        candidate_counts = counts[:, wi]
        global_counts = snapshot.counts[:, wi].sum(axis=0, dtype=np.float64)
        global_total = float(global_counts.sum())
        prior = (
            global_counts / global_total
            if global_total > 0
            else np.full(len(ACTIONS), 1 / len(ACTIONS), dtype=np.float64)
        )
        total = candidate_counts.sum(axis=1)
        denominator = total + prior_strength
        for ai, action in enumerate(ACTIONS):
            columns[f"asof_action_mix_{action}_h{hours}"] = (
                (candidate_counts[:, ai] + prior_strength * prior[ai]) / denominator
            ).astype(np.float32)

    names = feature_names()
    matrix = np.column_stack([columns[name] for name in names]).astype(np.float32)
    if matrix.shape != (candidate_aids.size, len(names)) or not np.isfinite(matrix).all():
        raise ValueError("as-of demand feature matrix failed shape/finite checks")
    return matrix
