from __future__ import annotations

import heapq
import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

from otto_recsys.runtime import Heartbeat

TARGETS = ("clicks", "carts", "orders")

METRIC_WEIGHTS = {
    "clicks": 0.10,
    "carts": 0.30,
    "orders": 0.60,
}

SESSION_EVENT_WEIGHTS = {
    "clicks": {
        "clicks": 1.0,
        "carts": 2.0,
        "orders": 2.5,
    },
    "carts": {
        "clicks": 1.0,
        "carts": 4.0,
        "orders": 3.0,
    },
    "orders": {
        "clicks": 1.0,
        "carts": 3.0,
        "orders": 5.0,
    },
}


@dataclass(frozen=True)
class PopularityIndex:
    clicks: tuple[int, ...]
    carts: tuple[int, ...]
    orders: tuple[int, ...]

    def for_target(self, target: str) -> tuple[int, ...]:
        if target == "clicks":
            return self.clicks
        if target == "carts":
            return self.carts
        if target == "orders":
            return self.orders
        raise ValueError(f"unknown target {target!r}")

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def read_json(
        cls,
        path: str | Path,
    ) -> PopularityIndex:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        return cls(
            clicks=tuple(payload["clicks"]),
            carts=tuple(payload["carts"]),
            orders=tuple(payload["orders"]),
        )


def _top_items(
    counter: Counter[int],
    limit: int,
) -> tuple[int, ...]:
    if limit <= 0:
        raise ValueError("limit must be positive")

    ranked = heapq.nsmallest(
        limit,
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    )

    return tuple(aid for aid, _ in ranked)


def fit_popularity(
    train_sessions_path: str | Path,
    *,
    logger: logging.Logger,
    top_n: int = 500,
    heartbeat_seconds: float = 30.0,
) -> PopularityIndex:
    """Fit leakage-safe action-specific popularity."""
    counters = {
        target: Counter[int]()
        for target in TARGETS
    }

    progress: dict[str, int] = {
        "sessions": 0,
        "events": 0,
    }

    started = time.perf_counter()

    def snapshot() -> dict[str, int | float]:
        elapsed = max(
            time.perf_counter() - started,
            1e-9,
        )
        return {
            **progress,
            "throughput": round(
                progress["events"] / elapsed,
                1,
            ),
        }

    with (
        Heartbeat(
            logger,
            stage="fit_popularity",
            interval_seconds=heartbeat_seconds,
            progress_provider=snapshot,
        ),
        Path(train_sessions_path).open("rb") as handle,
    ):
        for line in handle:
            record = orjson.loads(line)
            events: list[dict[str, Any]] = record["events"]

            progress["sessions"] += 1
            progress["events"] += len(events)

            for event in events:
                action = str(event["type"])
                aid = int(event["aid"])

                if action in counters:
                    counters[action][aid] += 1

    logger.info(
        "popularity_complete",
        extra={
            "event": "popularity_complete",
            "stage": "fit_popularity",
            "sessions": progress["sessions"],
            "events": progress["events"],
            "elapsed_seconds": round(
                time.perf_counter() - started,
                3,
            ),
        },
    )

    return PopularityIndex(
        clicks=_top_items(counters["clicks"], top_n),
        carts=_top_items(counters["carts"], top_n),
        orders=_top_items(counters["orders"], top_n),
    )


def session_candidates(
    events: list[dict[str, Any]],
    target: str,
    *,
    limit: int,
) -> list[tuple[int, float]]:
    """Rank observed items using target-conditioned action and recency signals."""
    if target not in SESSION_EVENT_WEIGHTS:
        raise ValueError(f"unknown target {target!r}")

    if limit <= 0:
        raise ValueError("limit must be positive")

    scores: dict[int, float] = {}
    last_rank: dict[int, int] = {}

    target_weights = SESSION_EVENT_WEIGHTS[target]

    for reverse_rank, event in enumerate(reversed(events)):
        aid = int(event["aid"])
        action = str(event["type"])

        action_weight = target_weights[action]
        recency_weight = 1.0 / (
            1.0 + 0.10 * reverse_rank
        )

        scores[aid] = scores.get(aid, 0.0) + (
            action_weight * recency_weight
        )

        last_rank.setdefault(aid, reverse_rank)

    ranked = sorted(
        scores,
        key=lambda aid: (
            -scores[aid],
            last_rank[aid],
            aid,
        ),
    )

    return [
        (aid, scores[aid])
        for aid in ranked[:limit]
    ]


def recommend(
    events: list[dict[str, Any]],
    target: str,
    popularity: PopularityIndex,
    *,
    k: int = 20,
) -> list[int]:
    """Build deterministic unique top-K recommendations."""
    session_ranked = session_candidates(
        events,
        target,
        limit=k,
    )

    recommendations = [
        aid
        for aid, _ in session_ranked
    ]
    used = set(recommendations)

    for aid in popularity.for_target(target):
        if aid in used:
            continue

        recommendations.append(aid)
        used.add(aid)

        if len(recommendations) >= k:
            break

    return recommendations[:k]


def target_set(
    labels: dict[str, Any],
    target: str,
) -> set[int]:
    value = labels.get(target)

    if value is None:
        return set()

    if target == "clicks":
        return {int(value)}

    if not isinstance(value, list):
        raise ValueError(
            f"{target} ground truth must be a list"
        )

    return {int(aid) for aid in value}


def evaluate(
    sessions_path: str | Path,
    labels_path: str | Path,
    popularity: PopularityIndex,
    *,
    logger: logging.Logger,
    k: int = 20,
    heartbeat_seconds: float = 30.0,
) -> dict[str, float]:
    """Evaluate the baseline in bounded memory."""
    hits = {target: 0 for target in TARGETS}
    denominators = {target: 0 for target in TARGETS}

    progress = {"sessions": 0}
    started = time.perf_counter()

    def snapshot() -> dict[str, int | float]:
        elapsed = max(
            time.perf_counter() - started,
            1e-9,
        )
        return {
            **progress,
            "throughput": round(
                progress["sessions"] / elapsed,
                1,
            ),
        }

    with (
        Heartbeat(
            logger,
            stage="baseline_evaluation",
            interval_seconds=heartbeat_seconds,
            progress_provider=snapshot,
        ),
        Path(sessions_path).open("rb") as sessions_handle,
        Path(labels_path).open("rb") as labels_handle,
    ):
        for session_line, label_line in zip(
            sessions_handle,
            labels_handle,
            strict=True,
        ):
            session_record = orjson.loads(session_line)
            label_record = orjson.loads(label_line)

            if session_record["session"] != label_record["session"]:
                raise RuntimeError(
                    "session and label streams are misaligned"
                )

            events: list[dict[str, Any]] = (
                session_record["events"]
            )
            labels: dict[str, Any] = (
                label_record["labels"]
            )

            for target in TARGETS:
                predictions = recommend(
                    events,
                    target,
                    popularity,
                    k=k,
                )
                truth = target_set(
                    labels,
                    target,
                )

                hits[target] += len(
                    set(predictions) & truth
                )
                denominators[target] += min(
                    k,
                    len(truth),
                )

            progress["sessions"] += 1

    recalls = {
        target: (
            hits[target] / denominators[target]
            if denominators[target]
            else 0.0
        )
        for target in TARGETS
    }

    weighted_recall = sum(
        METRIC_WEIGHTS[target] * recalls[target]
        for target in TARGETS
    )

    metrics = {
        "click_recall_20": recalls["clicks"],
        "cart_recall_20": recalls["carts"],
        "order_recall_20": recalls["orders"],
        "weighted_recall_20": weighted_recall,
    }

    logger.info(
        "baseline_evaluation_complete",
        extra={
            "event": "baseline_evaluation_complete",
            "stage": "baseline_evaluation",
            "status": "passed",
            "sessions": progress["sessions"],
            "elapsed_seconds": round(
                time.perf_counter() - started,
                3,
            ),
            **metrics,
        },
    )

    return metrics
