from __future__ import annotations

from dataclasses import dataclass, field

TARGETS = ("clicks", "carts", "orders")

WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}


@dataclass
class RecallGrid:
    """Streaming multi-target Recall@K accumulator."""

    ks: tuple[int, ...]
    hits: dict[
        str,
        dict[int, int],
    ] = field(init=False)
    denominators: dict[
        str,
        dict[int, int],
    ] = field(init=False)

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(set(self.ks))
        )

        if not normalized:
            raise ValueError("ks must not be empty")

        if any(k <= 0 for k in normalized):
            raise ValueError(
                "all k values must be positive"
            )

        self.ks = normalized

        self.hits = {
            target: {
                k: 0
                for k in self.ks
            }
            for target in TARGETS
        }

        self.denominators = {
            target: {
                k: 0
                for k in self.ks
            }
            for target in TARGETS
        }

    def update(
        self,
        target: str,
        candidates: list[int],
        truth: set[int],
    ) -> None:
        if target not in TARGETS:
            raise ValueError(
                f"unknown target {target!r}"
            )

        for k in self.ks:
            predicted = set(
                candidates[:k]
            )

            self.hits[target][k] += len(
                predicted & truth
            )

            self.denominators[target][k] += min(
                k,
                len(truth),
            )

    def results(self) -> dict[str, float]:
        metrics: dict[str, float] = {}

        for k in self.ks:
            weighted = 0.0

            for target in TARGETS:
                denominator = (
                    self.denominators[target][k]
                )

                recall = (
                    self.hits[target][k]
                    / denominator
                    if denominator
                    else 0.0
                )

                metrics[
                    f"{target}_recall_{k}"
                ] = recall

                weighted += (
                    WEIGHTS[target] * recall
                )

            metrics[
                f"weighted_recall_{k}"
            ] = weighted

        return metrics
