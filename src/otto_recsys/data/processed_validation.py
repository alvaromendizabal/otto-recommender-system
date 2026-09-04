from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from otto_recsys.data.schema import EVENT_SCHEMA
from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class ProcessedValidationSummary:
    """Independent integrity summary for flattened OTTO Parquet data."""

    status: str
    parts: int
    rows: int
    sessions: int
    min_ts: int
    max_ts: int


def validate_processed_dataset(
    root: str | Path,
    *,
    logger: logging.Logger,
    heartbeat_seconds: float = 30.0,
    expected_status: str | None = None,
) -> ProcessedValidationSummary:
    """Validate schema, ordering, counts, parts, and manifest consistency."""
    directory = Path(root).resolve()
    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("conversion manifest must contain a JSON object")

    status = payload.get("status")
    expected_parts = payload.get("parts_written")
    expected_rows = payload.get("events_processed")
    expected_sessions = payload.get("sessions_processed")

    if status not in {"running", "partial", "complete"}:
        raise ValueError(f"invalid conversion status: {status!r}")

    if expected_status is not None and status != expected_status:
        raise RuntimeError(
            f"expected conversion status={expected_status!r}, "
            f"observed {status!r}"
        )

    if not isinstance(expected_parts, int) or expected_parts < 0:
        raise ValueError("manifest parts_written must be a nonnegative integer")

    if not isinstance(expected_rows, int) or expected_rows < 0:
        raise ValueError("manifest events_processed must be a nonnegative integer")

    if not isinstance(expected_sessions, int) or expected_sessions < 0:
        raise ValueError(
            "manifest sessions_processed must be a nonnegative integer"
        )

    part_paths = sorted(directory.glob("part-*.parquet"))

    expected_names = [
        f"part-{index:06d}.parquet"
        for index in range(expected_parts)
    ]
    observed_names = [path.name for path in part_paths]

    if observed_names != expected_names:
        raise RuntimeError(
            "Parquet part sequence differs from conversion manifest"
        )

    if expected_rows > 0 and not part_paths:
        raise RuntimeError("manifest reports events but no Parquet parts exist")

    progress: dict[str, int] = {
        "part": 0,
        "events": 0,
        "sessions": 0,
    }

    started = time.perf_counter()

    min_ts: int | None = None
    max_ts: int | None = None

    previous_session: int | None = None
    previous_event_index: int | None = None
    previous_ts: int | None = None

    def progress_snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["events"] / elapsed, 1),
        }

    logger.info(
        "processed_validation_start",
        extra={
            "event": "processed_validation_start",
            "stage": "processed_validation",
            "status": status,
            "parts": expected_parts,
            "events": expected_rows,
            "sessions": expected_sessions,
        },
    )

    with Heartbeat(
        logger,
        stage="processed_validation",
        interval_seconds=heartbeat_seconds,
        progress_provider=progress_snapshot,
    ):
        for part_index, part_path in enumerate(part_paths):
            parquet_file = pq.ParquetFile(part_path)

            if parquet_file.schema_arrow != EVENT_SCHEMA:
                raise RuntimeError(
                    f"schema mismatch in {part_path.name}"
                )

            for batch in parquet_file.iter_batches(
                batch_size=250_000,
                columns=[
                    "session",
                    "ts",
                    "event_type",
                    "event_index",
                ],
            ):
                sessions = batch.column("session").to_pylist()
                timestamps = batch.column("ts").to_pylist()
                event_types = batch.column("event_type").to_pylist()
                event_indices = batch.column("event_index").to_pylist()

                for session, ts, event_type, event_index in zip(
                    sessions,
                    timestamps,
                    event_types,
                    event_indices,
                    strict=True,
                ):
                    if event_type not in {0, 1, 2}:
                        raise RuntimeError(
                            f"invalid event_type={event_type}"
                        )

                    if session != previous_session:
                        if event_index != 0:
                            raise RuntimeError(
                                f"session {session} does not begin "
                                "at event_index=0"
                            )

                        progress["sessions"] += 1
                        previous_session = session
                        previous_event_index = event_index
                        previous_ts = ts

                    else:
                        assert previous_event_index is not None
                        assert previous_ts is not None

                        if event_index != previous_event_index + 1:
                            raise RuntimeError(
                                f"session {session} has noncontiguous "
                                "event_index"
                            )

                        if ts < previous_ts:
                            raise RuntimeError(
                                f"session {session} has decreasing timestamps"
                            )

                        previous_event_index = event_index
                        previous_ts = ts

                    min_ts = ts if min_ts is None else min(min_ts, ts)
                    max_ts = ts if max_ts is None else max(max_ts, ts)

                    progress["events"] += 1

            progress["part"] = part_index + 1

            logger.info(
                "processed_part_validated",
                extra={
                    "event": "processed_part_validated",
                    "stage": "processed_validation",
                    "part": part_index,
                    "events": progress["events"],
                    "sessions": progress["sessions"],
                },
            )

    if progress["events"] != expected_rows:
        raise RuntimeError(
            f"manifest reports {expected_rows} events but "
            f"Parquet contains {progress['events']}"
        )

    if progress["sessions"] != expected_sessions:
        raise RuntimeError(
            f"manifest reports {expected_sessions} sessions but "
            f"Parquet contains {progress['sessions']}"
        )

    if min_ts is None or max_ts is None:
        raise RuntimeError("processed dataset contains no events")

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "processed_validation_complete",
        extra={
            "event": "processed_validation_complete",
            "stage": "processed_validation",
            "status": "passed",
            "events": progress["events"],
            "sessions": progress["sessions"],
            "elapsed_seconds": elapsed,
            "throughput": round(
                progress["events"] / max(elapsed, 1e-9),
                1,
            ),
        },
    )

    return ProcessedValidationSummary(
        status=status,
        parts=len(part_paths),
        rows=progress["events"],
        sessions=progress["sessions"],
        min_ts=min_ts,
        max_ts=max_ts,
    )
