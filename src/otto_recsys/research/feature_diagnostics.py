"""Streaming, exact first/second-moment diagnostics for bounded feature studies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Reporting thresholds only; they never select or drop features."""

    constant_std: float = 1e-12
    near_duplicate_correlation: float = 0.995

    def validate(self) -> None:
        if (
            not np.isfinite(self.constant_std)
            or self.constant_std < 0
            or not np.isfinite(self.near_duplicate_correlation)
            or not 0 < self.near_duplicate_correlation <= 1
        ):
            raise ValueError("diagnostic thresholds are invalid")


class StreamingFeatureDiagnostics:
    """Accumulate exact moments without retaining the full candidate matrix."""

    def __init__(
        self,
        names: tuple[str, ...],
        *,
        baseline_count: int,
        family: tuple[str, ...] | None = None,
        thresholds: DiagnosticThresholds | None = None,
    ) -> None:
        if not names or len(set(names)) != len(names):
            raise ValueError("feature names must be nonempty and unique")
        if not 1 <= baseline_count < len(names):
            raise ValueError("baseline_count must split baseline and added features")
        if family is not None and len(family) != len(names):
            raise ValueError("feature family labels must align with names")
        self.names = names
        self.baseline_count = baseline_count
        self.family = family or tuple(
            "baseline" if i < baseline_count else "added"
            for i in range(len(names))
        )
        self.thresholds = thresholds or DiagnosticThresholds()
        self.thresholds.validate()

        width = len(names)
        added = width - baseline_count
        self.rows = 0
        self.sessions = 0
        self.sum = np.zeros(width, dtype=np.float64)
        self.sum_sq = np.zeros(width, dtype=np.float64)
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)
        self.nonzero = np.zeros(width, dtype=np.int64)
        self.session_nonzero = np.zeros(width, dtype=np.int64)
        self.baseline_added_cross = np.zeros((baseline_count, added), dtype=np.float64)
        self.added_cross = np.zeros((added, added), dtype=np.float64)

    def update(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[1] != len(self.names) or values.shape[0] < 1:
            raise ValueError("feature batch shape differs from diagnostic schema")
        if not np.isfinite(values).all():
            raise ValueError("non-finite feature values are forbidden")
        x = values.astype(np.float64, copy=False)
        self.rows += x.shape[0]
        self.sessions += 1
        self.sum += x.sum(axis=0)
        self.sum_sq += np.square(x).sum(axis=0)
        self.minimum = np.minimum(self.minimum, x.min(axis=0))
        self.maximum = np.maximum(self.maximum, x.max(axis=0))
        self.nonzero += np.count_nonzero(x, axis=0)
        self.session_nonzero += np.any(x != 0, axis=0)
        base = x[:, : self.baseline_count]
        added = x[:, self.baseline_count :]
        self.baseline_added_cross += base.T @ added
        self.added_cross += added.T @ added

    @staticmethod
    def _correlation(
        cross: np.ndarray,
        left_sum: np.ndarray,
        right_sum: np.ndarray,
        left_std: np.ndarray,
        right_std: np.ndarray,
        rows: int,
    ) -> np.ndarray:
        covariance = cross / rows - np.outer(left_sum / rows, right_sum / rows)
        denominator = np.outer(left_std, right_std)
        return np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0,
        )

    def finalize(self) -> dict[str, Any]:
        if self.rows == 0 or self.sessions == 0:
            raise ValueError("diagnostics require at least one complete session")
        mean = self.sum / self.rows
        variance = np.maximum(0.0, self.sum_sq / self.rows - np.square(mean))
        std = np.sqrt(variance)
        b = self.baseline_count
        baseline_added = self._correlation(
            self.baseline_added_cross,
            self.sum[:b],
            self.sum[b:],
            std[:b],
            std[b:],
            self.rows,
        )
        added_corr = self._correlation(
            self.added_cross,
            self.sum[b:],
            self.sum[b:],
            std[b:],
            std[b:],
            self.rows,
        )
        np.fill_diagonal(added_corr, 0.0)
        max_baseline = np.zeros(len(self.names), dtype=np.float64)
        max_added = np.zeros(len(self.names), dtype=np.float64)
        max_baseline[b:] = np.max(np.abs(baseline_added), axis=0, initial=0.0)
        max_added[b:] = np.max(np.abs(added_corr), axis=0, initial=0.0)

        rows = []
        for index, name in enumerate(self.names):
            rows.append(
                {
                    "feature": name,
                    "family": self.family[index],
                    "mean": float(mean[index]),
                    "std": float(std[index]),
                    "minimum": float(self.minimum[index]),
                    "maximum": float(self.maximum[index]),
                    "row_nonzero_share": float(self.nonzero[index] / self.rows),
                    "session_nonzero_share": float(
                        self.session_nonzero[index] / self.sessions
                    ),
                    "max_abs_corr_with_baseline": float(max_baseline[index]),
                    "max_abs_corr_with_added": float(max_added[index]),
                    "constant": bool(std[index] <= self.thresholds.constant_std),
                    "near_duplicate_baseline": bool(
                        index >= b
                        and max_baseline[index]
                        >= self.thresholds.near_duplicate_correlation
                    ),
                    "near_duplicate_added": bool(
                        index >= b
                        and max_added[index]
                        >= self.thresholds.near_duplicate_correlation
                    ),
                }
            )
        return {
            "rows": self.rows,
            "sessions": self.sessions,
            "thresholds": {
                "constant_std": self.thresholds.constant_std,
                "near_duplicate_correlation": self.thresholds.near_duplicate_correlation,
            },
            "features": rows,
            "interpretation": (
                "Unsupervised fitting-only engineering diagnostics. Flags are not "
                "feature-retention decisions."
            ),
        }
