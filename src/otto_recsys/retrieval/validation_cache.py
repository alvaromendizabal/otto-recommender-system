from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.runtime import Heartbeat

_ITEMS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.int8(), nullable=False),
        pa.field("event_index", pa.uint16(), nullable=False),
        pa.field("recency_rank", pa.uint16(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)

_LABELS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("objective", pa.string(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)

_ACTION_TO_ID = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}


@dataclass(frozen=True)
class RetrievalValidationManifest:
    """Immutable identity for the flattened retrieval-validation cache."""

    validation_manifest_id: str
    buckets: int
    sessions: int
    item_rows: int
    label_rows: int
    items_sha256: str
    labels_sha256: str


def _read_validation_manifest_id(validation_dir: Path) -> str:
    payload = json.loads(
        (validation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    manifest_id = payload.get("manifest_id")

    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("validation manifest does not contain manifest_id")

    return manifest_id


def _event_rows(
    session: int,
    events: list[dict[str, Any]],
    *,
    bucket: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    """Return most-recent unique observed items for one validation session."""
    seen: set[int] = set()
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    recency_rank = 0

    if len(events) > 65_535:
        raise ValueError(
            f"session {session} exceeds uint16 event-index range"
        )

    for event_index in range(len(events) - 1, -1, -1):
        event = events[event_index]
        aid = int(event["aid"])

        if aid in seen:
            continue

        action = str(event["type"])
        event_type = _ACTION_TO_ID.get(action)

        if event_type is None:
            raise ValueError(
                f"session {session} contains invalid action {action!r}"
            )

        seen.add(aid)
        recency_rank += 1

        rows.append(
            (
                session,
                aid,
                int(event["ts"]),
                event_type,
                event_index,
                recency_rank,
                bucket,
            )
        )

    return rows


def _label_rows(
    session: int,
    labels: dict[str, Any],
    *,
    bucket: int,
) -> list[tuple[int, str, int, int]]:
    rows: list[tuple[int, str, int, int]] = []

    click = labels.get("clicks")
    if click is not None:
        rows.append((session, "clicks", int(click), bucket))

    for objective in ("carts", "orders"):
        value = labels.get(objective)

        if value is None:
            continue

        if not isinstance(value, list):
            raise ValueError(
                f"{objective} labels for session {session} must be a list"
            )

        for aid in value:
            rows.append((session, objective, int(aid), bucket))

    return rows


def _flush_items(
    writer: pq.ParquetWriter,
    rows: list[tuple[int, int, int, int, int, int, int]],
) -> int:
    if not rows:
        return 0

    table = pa.Table.from_arrays(
        [
            pa.array([row[0] for row in rows], type=pa.int32()),
            pa.array([row[1] for row in rows], type=pa.int32()),
            pa.array([row[2] for row in rows], type=pa.int64()),
            pa.array([row[3] for row in rows], type=pa.int8()),
            pa.array([row[4] for row in rows], type=pa.uint16()),
            pa.array([row[5] for row in rows], type=pa.uint16()),
            pa.array([row[6] for row in rows], type=pa.uint16()),
        ],
        schema=_ITEMS_SCHEMA,
    )
    writer.write_table(table)
    return table.num_rows


def _flush_labels(
    writer: pq.ParquetWriter,
    rows: list[tuple[int, str, int, int]],
) -> int:
    if not rows:
        return 0

    table = pa.Table.from_arrays(
        [
            pa.array([row[0] for row in rows], type=pa.int32()),
            pa.array([row[1] for row in rows], type=pa.string()),
            pa.array([row[2] for row in rows], type=pa.int32()),
            pa.array([row[3] for row in rows], type=pa.uint16()),
        ],
        schema=_LABELS_SCHEMA,
    )
    writer.write_table(table)
    return table.num_rows


def build_retrieval_validation_cache(
    validation_dir: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    flush_sessions: int = 10_000,
    heartbeat_seconds: float = 30.0,
) -> RetrievalValidationManifest:
    """Flatten validation JSONL into bounded-memory Parquet retrieval inputs."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")

    if flush_sessions <= 0:
        raise ValueError("flush_sessions must be positive")

    source_dir = Path(validation_dir).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    sessions_path = source_dir / "test_sessions.jsonl"
    labels_path = source_dir / "test_labels.jsonl"

    if not sessions_path.is_file():
        raise FileNotFoundError(sessions_path)
    if not labels_path.is_file():
        raise FileNotFoundError(labels_path)

    validation_manifest_id = _read_validation_manifest_id(source_dir)

    items_path = destination / "items.parquet"
    labels_output_path = destination / "labels.parquet"
    manifest_path = destination / "manifest.json"

    if items_path.is_file() and labels_output_path.is_file() and manifest_path.is_file():
        existing: RetrievalValidationManifest | None
        try:
            existing_payload = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            existing = RetrievalValidationManifest(**existing_payload)
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
        ):
            existing = None

        if (
            existing is not None
            and existing.validation_manifest_id == validation_manifest_id
            and existing.buckets == buckets
            and sha256_file(items_path) == existing.items_sha256
            and sha256_file(labels_output_path) == existing.labels_sha256
        ):
            logger.info(
                "retrieval_validation_cache_reused",
                extra={
                    "event": "retrieval_validation_cache_reused",
                    "stage": "retrieval_validation_cache",
                    "sessions": existing.sessions,
                    "item_rows": existing.item_rows,
                    "label_rows": existing.label_rows,
                },
            )
            return existing

    items_temp = destination / ".items.parquet.tmp"
    labels_temp = destination / ".labels.parquet.tmp"
    manifest_temp = destination / ".manifest.json.tmp"

    for path in (items_temp, labels_temp, manifest_temp):
        path.unlink(missing_ok=True)

    item_buffer: list[tuple[int, int, int, int, int, int, int]] = []
    label_buffer: list[tuple[int, str, int, int]] = []

    sessions = 0
    item_rows = 0
    label_rows = 0
    started = time.perf_counter()

    progress: dict[str, int] = {
        "sessions": 0,
        "item_rows": 0,
        "label_rows": 0,
    }

    def snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["sessions"] / elapsed, 1),
        }

    logger.info(
        "retrieval_validation_cache_start",
        extra={
            "event": "retrieval_validation_cache_start",
            "stage": "retrieval_validation_cache",
            "buckets": buckets,
        },
    )

    items_writer = pq.ParquetWriter(
        items_temp,
        _ITEMS_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
    )
    labels_writer = pq.ParquetWriter(
        labels_temp,
        _LABELS_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=True,
    )

    try:
        with (
            Heartbeat(
                logger,
                stage="retrieval_validation_cache",
                interval_seconds=heartbeat_seconds,
                progress_provider=snapshot,
            ),
            sessions_path.open("rb") as sessions_handle,
            labels_path.open("rb") as labels_handle,
        ):
            for session_line, label_line in zip(
                sessions_handle,
                labels_handle,
                strict=True,
            ):
                session_record = orjson.loads(session_line)
                label_record = orjson.loads(label_line)

                session = int(session_record["session"])
                if session != int(label_record["session"]):
                    raise RuntimeError(
                        "validation session and label streams are misaligned"
                    )

                events = session_record["events"]
                labels = label_record["labels"]

                if not isinstance(events, list) or not events:
                    raise ValueError(
                        f"validation session {session} has no observed events"
                    )
                if not isinstance(labels, dict):
                    raise ValueError(
                        f"validation labels for session {session} are invalid"
                    )

                bucket = session % buckets
                item_buffer.extend(
                    _event_rows(session, events, bucket=bucket)
                )
                label_buffer.extend(
                    _label_rows(session, labels, bucket=bucket)
                )

                sessions += 1
                progress["sessions"] = sessions

                if sessions % flush_sessions == 0:
                    item_rows += _flush_items(items_writer, item_buffer)
                    label_rows += _flush_labels(labels_writer, label_buffer)
                    item_buffer.clear()
                    label_buffer.clear()

                    progress["item_rows"] = item_rows
                    progress["label_rows"] = label_rows

        item_rows += _flush_items(items_writer, item_buffer)
        label_rows += _flush_labels(labels_writer, label_buffer)
        progress["item_rows"] = item_rows
        progress["label_rows"] = label_rows

    finally:
        items_writer.close()
        labels_writer.close()

    if sessions <= 0 or item_rows <= 0 or label_rows <= 0:
        items_temp.unlink(missing_ok=True)
        labels_temp.unlink(missing_ok=True)
        raise RuntimeError("retrieval-validation cache is empty")

    os.replace(items_temp, items_path)
    os.replace(labels_temp, labels_output_path)

    manifest = RetrievalValidationManifest(
        validation_manifest_id=validation_manifest_id,
        buckets=buckets,
        sessions=sessions,
        item_rows=item_rows,
        label_rows=label_rows,
        items_sha256=sha256_file(items_path),
        labels_sha256=sha256_file(labels_output_path),
    )

    manifest_temp.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temp, manifest_path)

    logger.info(
        "retrieval_validation_cache_complete",
        extra={
            "event": "retrieval_validation_cache_complete",
            "stage": "retrieval_validation_cache",
            "status": "passed",
            "sessions": sessions,
            "item_rows": item_rows,
            "label_rows": label_rows,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        },
    )

    return manifest
