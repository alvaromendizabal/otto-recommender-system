from __future__ import annotations

import hashlib
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

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.runtime import Heartbeat

_ACTION_TO_ID = {
    "clicks": 0,
    "carts": 1,
    "orders": 2,
}

_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.int8(), nullable=False),
        pa.field("event_index", pa.uint16(), nullable=False),
        pa.field("fold", pa.uint8(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)

_ITEMS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.int8(), nullable=False),
        pa.field("event_index", pa.uint16(), nullable=False),
        pa.field("recency_rank", pa.uint16(), nullable=False),
        pa.field("fold", pa.uint8(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)

_LABELS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("objective", pa.string(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("fold", pa.uint8(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)

_EXAMPLES_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("observed_events", pa.uint16(), nullable=False),
        pa.field("observed_unique_items", pa.uint16(), nullable=False),
        pa.field("click_labels", pa.uint8(), nullable=False),
        pa.field("cart_labels", pa.uint16(), nullable=False),
        pa.field("order_labels", pa.uint16(), nullable=False),
        pa.field("first_ts", pa.int64(), nullable=False),
        pa.field("last_ts", pa.int64(), nullable=False),
        pa.field("fold", pa.uint8(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)


@dataclass(frozen=True)
class RankingTrainingCacheConfig:
    buckets: int
    folds: int
    fold_seed: int
    flush_sessions: int
    max_examples: int | None


@dataclass(frozen=True)
class RankingTrainingCacheManifest:
    validation_manifest_id: str
    input_id: str
    config: RankingTrainingCacheConfig
    sessions: int
    event_rows: int
    item_rows: int
    label_rows: int
    click_labels: int
    cart_labels: int
    order_labels: int
    fold_session_counts: tuple[int, ...]
    source_sessions_sha256: str
    source_labels_sha256: str
    events_sha256: str
    items_sha256: str
    labels_sha256: str
    examples_sha256: str
    elapsed_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _validation_manifest_id(validation_dir: Path) -> str:
    manifest_path = validation_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = _load_json(manifest_path)
    value = payload.get("manifest_id")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("validation manifest must contain a 64-character manifest_id")
    return value


def _stable_u64(*parts: int) -> int:
    payload = ":".join(str(part) for part in parts).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def fold_for_session(session: int, *, seed: int, folds: int) -> int:
    """Assign one session to a stable OOF fold without order dependence."""
    if folds < 2 or folds > 255:
        raise ValueError("folds must be between 2 and 255")
    return _stable_u64(seed, session, 1) % folds


def _event_type(event: dict[str, Any], session: int) -> int:
    action = str(event.get("type"))
    event_type = _ACTION_TO_ID.get(action)
    if event_type is None:
        raise ValueError(f"session {session} contains invalid action {action!r}")
    return event_type


def _event_rows(
    session: int,
    events: list[dict[str, Any]],
    *,
    fold: int,
    bucket: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    if not events:
        raise ValueError(f"validation session {session} has no observed events")
    if len(events) > 65_535:
        raise ValueError(f"session {session} exceeds uint16 event-index range")

    rows: list[tuple[int, int, int, int, int, int, int]] = []
    previous_ts: int | None = None
    for event_index, event in enumerate(events):
        ts = int(event["ts"])
        if previous_ts is not None and ts < previous_ts:
            raise ValueError(f"session {session} events are not timestamp ordered")
        previous_ts = ts
        rows.append(
            (
                session,
                int(event["aid"]),
                ts,
                _event_type(event, session),
                event_index,
                fold,
                bucket,
            )
        )
    return rows


def _item_rows(
    session: int,
    events: list[dict[str, Any]],
    *,
    fold: int,
    bucket: int,
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    seen: set[int] = set()
    rows: list[tuple[int, int, int, int, int, int, int, int]] = []
    recency_rank = 0

    for event_index in range(len(events) - 1, -1, -1):
        event = events[event_index]
        aid = int(event["aid"])
        if aid in seen:
            continue
        seen.add(aid)
        recency_rank += 1
        rows.append(
            (
                session,
                aid,
                int(event["ts"]),
                _event_type(event, session),
                event_index,
                recency_rank,
                fold,
                bucket,
            )
        )
    return rows


def _label_rows(
    session: int,
    labels: dict[str, Any],
    *,
    fold: int,
    bucket: int,
) -> list[tuple[int, str, int, int, int]]:
    rows: list[tuple[int, str, int, int, int]] = []

    click = labels.get("clicks")
    if click is not None:
        rows.append((session, "clicks", int(click), fold, bucket))

    for objective in ("carts", "orders"):
        raw = labels.get(objective)
        if raw is None:
            continue
        if not isinstance(raw, list):
            raise ValueError(f"{objective} labels for session {session} must be a list")
        seen: set[int] = set()
        for value in raw:
            aid = int(value)
            if aid in seen:
                continue
            seen.add(aid)
            rows.append((session, objective, aid, fold, bucket))

    return rows


def _flush(
    writer: pq.ParquetWriter,
    schema: pa.Schema,
    rows: list[tuple[Any, ...]],
) -> int:
    if not rows:
        return 0
    arrays = [
        pa.array([row[index] for row in rows], type=field.type)
        for index, field in enumerate(schema)
    ]
    table = pa.Table.from_arrays(arrays, schema=schema)
    writer.write_table(table)
    return table.num_rows


def _write_json_atomic(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _manifest_from_json(path: Path) -> RankingTrainingCacheManifest:
    payload = _load_json(path)
    raw_config = payload.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("ranking training manifest config must be an object")
    raw_fold_counts = payload.get("fold_session_counts")
    if not isinstance(raw_fold_counts, list):
        raise ValueError("ranking training manifest fold counts must be a list")
    normalized = dict(payload)
    normalized["config"] = RankingTrainingCacheConfig(**raw_config)
    normalized["fold_session_counts"] = tuple(int(value) for value in raw_fold_counts)
    return RankingTrainingCacheManifest(**normalized)


def build_ranking_training_cache(
    validation_dir: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    folds: int = 5,
    fold_seed: int = 20260905,
    flush_sessions: int = 5_000,
    max_examples: int | None = None,
    heartbeat_seconds: float = 30.0,
) -> RankingTrainingCacheManifest:
    """Materialize the frozen validation prefixes for leakage-safe OOF training."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")
    if folds < 2 or folds > 255:
        raise ValueError("folds must be between 2 and 255")
    if flush_sessions <= 0:
        raise ValueError("flush_sessions must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")

    source_dir = Path(validation_dir).resolve()
    destination = Path(output_dir).resolve()
    sessions_path = source_dir / "test_sessions.jsonl"
    labels_path = source_dir / "test_labels.jsonl"
    for path in (sessions_path, labels_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    destination.mkdir(parents=True, exist_ok=True)
    validation_id = _validation_manifest_id(source_dir)
    source_sessions_sha = sha256_file(sessions_path)
    source_labels_sha = sha256_file(labels_path)
    config = RankingTrainingCacheConfig(
        buckets=buckets,
        folds=folds,
        fold_seed=fold_seed,
        flush_sessions=flush_sessions,
        max_examples=max_examples,
    )
    input_id = canonical_json_sha256(
        {
            "validation_manifest_id": validation_id,
            "source_sessions_sha256": source_sessions_sha,
            "source_labels_sha256": source_labels_sha,
            "config": asdict(config),
        }
    )

    manifest_path = destination / "manifest.json"
    events_path = destination / "events.parquet"
    items_path = destination / "items.parquet"
    labels_output_path = destination / "labels.parquet"
    examples_path = destination / "examples.parquet"

    if manifest_path.is_file():
        try:
            existing = _manifest_from_json(manifest_path)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if (
            existing is not None
            and existing.input_id == input_id
            and len(existing.fold_session_counts) == folds
            and events_path.is_file()
            and items_path.is_file()
            and labels_output_path.is_file()
            and examples_path.is_file()
            and sha256_file(events_path) == existing.events_sha256
            and sha256_file(items_path) == existing.items_sha256
            and sha256_file(labels_output_path) == existing.labels_sha256
            and sha256_file(examples_path) == existing.examples_sha256
        ):
            logger.info(
                "ranking_training_cache_reused",
                extra={
                    "event": "ranking_training_cache_reused",
                    "stage": "ranking_training_cache",
                    "status": "passed",
                    "sessions": existing.sessions,
                    "events": existing.event_rows,
                    "input_id": existing.input_id,
                },
            )
            return existing

    temp_paths = {
        "events": destination / ".events.parquet.tmp",
        "items": destination / ".items.parquet.tmp",
        "labels": destination / ".labels.parquet.tmp",
        "examples": destination / ".examples.parquet.tmp",
    }
    for path in temp_paths.values():
        path.unlink(missing_ok=True)

    writers = {
        "events": pq.ParquetWriter(
            temp_paths["events"],
            _EVENTS_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        ),
        "items": pq.ParquetWriter(
            temp_paths["items"],
            _ITEMS_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        ),
        "labels": pq.ParquetWriter(
            temp_paths["labels"],
            _LABELS_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        ),
        "examples": pq.ParquetWriter(
            temp_paths["examples"],
            _EXAMPLES_SCHEMA,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
        ),
    }

    event_buffer: list[tuple[Any, ...]] = []
    item_buffer: list[tuple[Any, ...]] = []
    label_buffer: list[tuple[Any, ...]] = []
    example_buffer: list[tuple[Any, ...]] = []

    sessions = 0
    event_rows = 0
    item_rows = 0
    label_rows = 0
    click_labels = 0
    cart_labels = 0
    order_labels = 0
    fold_counts = [0] * folds
    started = time.perf_counter()
    progress: dict[str, int] = {"sessions": 0, "events": 0, "labels": 0}

    def snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["sessions"] / elapsed, 1),
        }

    def flush_buffers() -> None:
        nonlocal event_rows, item_rows, label_rows
        event_rows += _flush(writers["events"], _EVENTS_SCHEMA, event_buffer)
        item_rows += _flush(writers["items"], _ITEMS_SCHEMA, item_buffer)
        label_rows += _flush(writers["labels"], _LABELS_SCHEMA, label_buffer)
        _flush(writers["examples"], _EXAMPLES_SCHEMA, example_buffer)
        event_buffer.clear()
        item_buffer.clear()
        label_buffer.clear()
        example_buffer.clear()
        progress["events"] = event_rows
        progress["labels"] = label_rows

    logger.info(
        "ranking_training_cache_start",
        extra={
            "event": "ranking_training_cache_start",
            "stage": "ranking_training_cache",
            "validation_manifest_id": validation_id,
            "input_id": input_id,
            "folds": folds,
            "max_examples": max_examples,
        },
    )

    try:
        with (
            Heartbeat(
                logger,
                stage="ranking_training_cache",
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

                raw_events = session_record.get("events")
                raw_labels = label_record.get("labels")
                if not isinstance(raw_events, list) or not raw_events:
                    raise ValueError(
                        f"validation session {session} has no observed events"
                    )
                if not isinstance(raw_labels, dict):
                    raise ValueError(
                        f"validation labels for session {session} are invalid"
                    )

                fold = fold_for_session(session, seed=fold_seed, folds=folds)
                bucket = session % buckets
                events = [dict(event) for event in raw_events]
                event_buffer.extend(
                    _event_rows(session, events, fold=fold, bucket=bucket)
                )
                items = _item_rows(session, events, fold=fold, bucket=bucket)
                item_buffer.extend(items)
                labels = _label_rows(session, raw_labels, fold=fold, bucket=bucket)
                label_buffer.extend(labels)

                session_clicks = sum(1 for row in labels if row[1] == "clicks")
                session_carts = sum(1 for row in labels if row[1] == "carts")
                session_orders = sum(1 for row in labels if row[1] == "orders")
                first_ts = int(events[0]["ts"])
                last_ts = int(events[-1]["ts"])
                example_buffer.append(
                    (
                        session,
                        len(events),
                        len(items),
                        session_clicks,
                        session_carts,
                        session_orders,
                        first_ts,
                        last_ts,
                        fold,
                        bucket,
                    )
                )

                sessions += 1
                fold_counts[fold] += 1
                click_labels += session_clicks
                cart_labels += session_carts
                order_labels += session_orders
                progress["sessions"] = sessions

                if sessions % flush_sessions == 0:
                    flush_buffers()
                    logger.info(
                        "ranking_training_cache_flush",
                        extra={
                            "event": "ranking_training_cache_flush",
                            "stage": "ranking_training_cache",
                            "sessions": sessions,
                            "events": event_rows,
                            "labels": label_rows,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        },
                    )

                if max_examples is not None and sessions >= max_examples:
                    break

        flush_buffers()
    finally:
        for writer in writers.values():
            writer.close()

    if sessions <= 0 or event_rows <= 0 or item_rows <= 0 or label_rows <= 0:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise RuntimeError("ranking training cache is empty")
    if sum(fold_counts) != sessions:
        raise RuntimeError("ranking training fold counts do not match session count")

    os.replace(temp_paths["events"], events_path)
    os.replace(temp_paths["items"], items_path)
    os.replace(temp_paths["labels"], labels_output_path)
    os.replace(temp_paths["examples"], examples_path)

    elapsed = round(time.perf_counter() - started, 3)
    manifest = RankingTrainingCacheManifest(
        validation_manifest_id=validation_id,
        input_id=input_id,
        config=config,
        sessions=sessions,
        event_rows=event_rows,
        item_rows=item_rows,
        label_rows=label_rows,
        click_labels=click_labels,
        cart_labels=cart_labels,
        order_labels=order_labels,
        fold_session_counts=tuple(fold_counts),
        source_sessions_sha256=source_sessions_sha,
        source_labels_sha256=source_labels_sha,
        events_sha256=sha256_file(events_path),
        items_sha256=sha256_file(items_path),
        labels_sha256=sha256_file(labels_output_path),
        examples_sha256=sha256_file(examples_path),
        elapsed_seconds=elapsed,
    )
    _write_json_atomic(asdict(manifest), manifest_path)

    logger.info(
        "ranking_training_cache_complete",
        extra={
            "event": "ranking_training_cache_complete",
            "stage": "ranking_training_cache",
            "status": "passed",
            "sessions": sessions,
            "events": event_rows,
            "labels": label_rows,
            "folds": folds,
            "elapsed_seconds": elapsed,
        },
    )
    return manifest
