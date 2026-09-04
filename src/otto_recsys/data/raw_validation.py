from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import orjson

from otto_recsys.runtime import Heartbeat

VALID_ACTIONS = frozenset({"clicks", "carts", "orders"})


@dataclass(frozen=True)
class RawValidationSummary:
    """Summary of validated raw OTTO sessions."""

    sessions: int
    events: int
    clicks: int
    carts: int
    orders: int
    min_ts: int
    max_ts: int


def validate_jsonl(
    path: str | Path,
    *,
    logger: logging.Logger,
    max_sessions: int | None = None,
    min_events_per_session: int = 1,
    heartbeat_seconds: float = 30.0,
) -> RawValidationSummary:
    """Validate nested OTTO JSONL using bounded memory."""
    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")

    if min_events_per_session <= 0:
        raise ValueError("min_events_per_session must be positive")

    source = Path(path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    progress = {
        "sessions": 0,
        "events": 0,
    }
    counts = {
        "clicks": 0,
        "carts": 0,
        "orders": 0,
    }

    min_ts: int | None = None
    max_ts: int | None = None
    started = time.perf_counter()

    def progress_snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["events"] / elapsed, 1),
        }

    logger.info(
        "raw_validation_start",
        extra={
            "event": "raw_validation_start",
            "stage": "raw_validation",
            "file": source.name,
        },
    )

    with (
        Heartbeat(
            logger,
            stage="raw_validation",
            interval_seconds=heartbeat_seconds,
            progress_provider=progress_snapshot,
        ),
        source.open("rb") as handle,
    ):
        for line_number, line in enumerate(handle, start=1):
            if (
                max_sessions is not None
                and progress["sessions"] >= max_sessions
            ):
                break

            if not line.strip():
                raise ValueError(f"line {line_number}: empty line")

            record = orjson.loads(line)

            if not isinstance(record, dict):
                raise ValueError(
                    f"line {line_number}: record must be an object"
                )

            session = record.get("session")
            events = record.get("events")

            if not isinstance(session, int) or session < 0:
                raise ValueError(
                    f"line {line_number}: invalid session identifier"
                )

            if not isinstance(events, list):
                raise ValueError(
                    f"session {session}: events must be a list"
                )

            if len(events) < min_events_per_session:
                raise ValueError(
                    f"session {session}: expected at least "
                    f"{min_events_per_session} events"
                )

            previous_ts: int | None = None

            for event_index, event in enumerate(events):
                if not isinstance(event, dict):
                    raise ValueError(
                        f"session {session} event {event_index}: "
                        "event must be an object"
                    )

                aid = event.get("aid")
                ts = event.get("ts")
                action = event.get("type")

                if not isinstance(aid, int) or aid < 0:
                    raise ValueError(
                        f"session {session} event {event_index}: invalid aid"
                    )

                if not isinstance(ts, int) or ts < 0:
                    raise ValueError(
                        f"session {session} event {event_index}: invalid ts"
                    )

                if not isinstance(action, str) or action not in VALID_ACTIONS:
                    raise ValueError(
                        f"session {session} event {event_index}: "
                        f"invalid action {action!r}"
                    )

                if previous_ts is not None and ts < previous_ts:
                    raise ValueError(
                        f"session {session}: timestamps are not ordered"
                    )

                previous_ts = ts
                counts[action] += 1
                progress["events"] += 1

                min_ts = ts if min_ts is None else min(min_ts, ts)
                max_ts = ts if max_ts is None else max(max_ts, ts)

            progress["sessions"] += 1

    if progress["sessions"] == 0 or min_ts is None or max_ts is None:
        raise ValueError("dataset contains no validated sessions")

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "raw_validation_complete",
        extra={
            "event": "raw_validation_complete",
            "stage": "raw_validation",
            "status": "passed",
            "sessions": progress["sessions"],
            "events": progress["events"],
            "elapsed_seconds": elapsed,
            "throughput": round(
                progress["events"] / max(elapsed, 1e-9),
                1,
            ),
        },
    )

    return RawValidationSummary(
        sessions=progress["sessions"],
        events=progress["events"],
        clicks=counts["clicks"],
        carts=counts["carts"],
        orders=counts["orders"],
        min_ts=min_ts,
        max_ts=max_ts,
    )
