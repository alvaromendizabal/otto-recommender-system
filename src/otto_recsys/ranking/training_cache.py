from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NotRequired, TypedDict

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


class FutureLabels(TypedDict):
    """Typed OTTO future-label contract for one supervised prefix."""

    clicks: int
    carts: NotRequired[list[int]]
    orders: NotRequired[list[int]]

_EVENTS_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("aid", pa.int32(), nullable=False),
        pa.field("ts", pa.int64(), nullable=False),
        pa.field("event_type", pa.int8(), nullable=False),
        pa.field("event_index", pa.uint16(), nullable=False),
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

_EXAMPLES_SCHEMA = pa.schema(
    [
        pa.field("session", pa.int32(), nullable=False),
        pa.field("source_events", pa.uint16(), nullable=False),
        pa.field("cut_index", pa.uint16(), nullable=False),
        pa.field("prefix_events", pa.uint16(), nullable=False),
        pa.field("future_events", pa.uint16(), nullable=False),
        pa.field("bucket", pa.uint16(), nullable=False),
    ]
)


@dataclass(frozen=True)
class RankingTrainingCacheConfig:
    buckets: int
    seed: int
    sample_denominator: int
    sample_remainder: int
    min_prefix_events: int
    max_prefix_events: int
    flush_examples: int
    max_examples: int | None


@dataclass(frozen=True)
class RankingTrainingCacheManifest:
    source_manifest_id: str
    input_id: str
    config: RankingTrainingCacheConfig
    sessions_seen: int
    examples: int
    event_rows: int
    item_rows: int
    label_rows: int
    click_labels: int
    cart_labels: int
    order_labels: int
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


def _source_manifest_id(path: Path) -> str:
    payload = _load_json(path)
    value = payload.get("manifest_id")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("source manifest must contain a 64-character manifest_id")
    return value


def _stable_u64(*parts: int) -> int:
    payload = ":".join(str(part) for part in parts).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def session_is_selected(
    session: int,
    *,
    seed: int,
    denominator: int,
    remainder: int,
) -> bool:
    """Deterministically subsample sessions without order dependence."""
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    if remainder < 0 or remainder >= denominator:
        raise ValueError("remainder must be in [0, denominator)")
    return _stable_u64(seed, session, 1) % denominator == remainder


def deterministic_cut_index(
    session: int,
    event_count: int,
    *,
    seed: int,
    min_prefix_events: int,
) -> int:
    """Choose a deterministic observed-prefix boundary with hidden future events."""
    if min_prefix_events <= 0:
        raise ValueError("min_prefix_events must be positive")
    if event_count <= min_prefix_events:
        raise ValueError("event_count must leave at least one hidden future event")

    choices = event_count - min_prefix_events
    return min_prefix_events + (_stable_u64(seed, session, 2) % choices)


def _event_type(event: dict[str, Any], session: int) -> int:
    action = str(event.get("type"))
    event_type = _ACTION_TO_ID.get(action)
    if event_type is None:
        raise ValueError(f"session {session} contains invalid action {action!r}")
    return event_type


def _unique_future_aids(
    future_events: list[dict[str, Any]],
    *,
    objective: str,
) -> list[int]:
    seen: set[int] = set()
    aids: list[int] = []
    for event in future_events:
        if str(event.get("type")) != objective:
            continue
        aid = int(event["aid"])
        if aid not in seen:
            seen.add(aid)
            aids.append(aid)
    return aids


def future_labels(
    future_events: list[dict[str, Any]],
    *,
    session: int,
) -> FutureLabels:
    """Return OTTO-style future labels for one deterministic prefix."""
    if not future_events:
        raise ValueError("future_events cannot be empty")

    # Validate every hidden action before constructing objective-specific labels.
    for event in future_events:
        _event_type(event, session)

    labels: FutureLabels = {"clicks": int(future_events[0]["aid"])}

    cart_aids = _unique_future_aids(future_events, objective="carts")
    if cart_aids:
        labels["carts"] = cart_aids

    order_aids = _unique_future_aids(future_events, objective="orders")
    if order_aids:
        labels["orders"] = order_aids

    return labels


def split_training_example(
    record: dict[str, Any],
    *,
    seed: int,
    min_prefix_events: int,
    max_prefix_events: int,
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], int]:
    """Split one session into observed prefix and hidden future without leakage."""
    session = int(record["session"])
    raw_events = record.get("events")
    if not isinstance(raw_events, list):
        raise ValueError(f"session {session} events must be a list")
    if len(raw_events) <= min_prefix_events:
        raise ValueError("session is too short for a supervised prefix")
    if max_prefix_events < min_prefix_events:
        raise ValueError("max_prefix_events must be >= min_prefix_events")
    if len(raw_events) > 65_535:
        raise ValueError(f"session {session} exceeds uint16 event-index range")

    events = [dict(event) for event in raw_events]
    cut_index = deterministic_cut_index(
        session,
        len(events),
        seed=seed,
        min_prefix_events=min_prefix_events,
    )
    prefix = events[:cut_index]
    future = events[cut_index:]
    if len(prefix) > max_prefix_events:
        prefix = prefix[-max_prefix_events:]

    if not prefix or not future:
        raise RuntimeError("prefix/future split invariant failed")
    return session, prefix, future, cut_index


def _event_rows(
    session: int,
    prefix: list[dict[str, Any]],
    *,
    bucket: int,
) -> list[tuple[int, int, int, int, int, int]]:
    rows: list[tuple[int, int, int, int, int, int]] = []
    for event_index, event in enumerate(prefix):
        rows.append(
            (
                session,
                int(event["aid"]),
                int(event["ts"]),
                _event_type(event, session),
                event_index,
                bucket,
            )
        )
    return rows


def _item_rows(
    session: int,
    prefix: list[dict[str, Any]],
    *,
    bucket: int,
) -> list[tuple[int, int, int, int, int, int, int]]:
    seen: set[int] = set()
    rows: list[tuple[int, int, int, int, int, int, int]] = []
    recency_rank = 0

    for event_index in range(len(prefix) - 1, -1, -1):
        event = prefix[event_index]
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
                bucket,
            )
        )
    return rows


def _label_rows(
    session: int,
    labels: FutureLabels,
    *,
    bucket: int,
) -> list[tuple[int, str, int, int]]:
    rows = [(session, "clicks", labels["clicks"], bucket)]
    rows.extend((session, "carts", aid, bucket) for aid in labels.get("carts", []))
    rows.extend(
        (session, "orders", aid, bucket) for aid in labels.get("orders", [])
    )
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
    payload = dict(payload)
    payload["config"] = RankingTrainingCacheConfig(**raw_config)
    return RankingTrainingCacheManifest(**payload)


def build_ranking_training_cache(
    source_path: str | Path,
    source_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    logger: logging.Logger,
    buckets: int = 32,
    seed: int = 20260905,
    sample_denominator: int = 8,
    sample_remainder: int = 0,
    min_prefix_events: int = 2,
    max_prefix_events: int = 50,
    flush_examples: int = 5_000,
    max_examples: int | None = None,
    heartbeat_seconds: float = 30.0,
) -> RankingTrainingCacheManifest:
    """Build deterministic pre-validation supervised prefixes for ranking/retrieval."""
    if buckets <= 0 or buckets > 65_535:
        raise ValueError("buckets must be between 1 and 65535")
    if sample_denominator <= 0:
        raise ValueError("sample_denominator must be positive")
    if sample_remainder < 0 or sample_remainder >= sample_denominator:
        raise ValueError("sample_remainder must be in [0, sample_denominator)")
    if min_prefix_events <= 0:
        raise ValueError("min_prefix_events must be positive")
    if max_prefix_events < min_prefix_events:
        raise ValueError("max_prefix_events must be >= min_prefix_events")
    if flush_examples <= 0:
        raise ValueError("flush_examples must be positive")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when provided")

    source = Path(source_path).resolve()
    source_manifest = Path(source_manifest_path).resolve()
    destination = Path(output_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not source_manifest.is_file():
        raise FileNotFoundError(source_manifest)

    destination.mkdir(parents=True, exist_ok=True)
    source_id = _source_manifest_id(source_manifest)
    config = RankingTrainingCacheConfig(
        buckets=buckets,
        seed=seed,
        sample_denominator=sample_denominator,
        sample_remainder=sample_remainder,
        min_prefix_events=min_prefix_events,
        max_prefix_events=max_prefix_events,
        flush_examples=flush_examples,
        max_examples=max_examples,
    )
    input_id = canonical_json_sha256(
        {
            "source_manifest_id": source_id,
            "source_size_bytes": source.stat().st_size,
            "config": asdict(config),
        }
    )

    manifest_path = destination / "manifest.json"
    events_path = destination / "events.parquet"
    items_path = destination / "items.parquet"
    labels_path = destination / "labels.parquet"
    examples_path = destination / "examples.parquet"

    if manifest_path.is_file():
        try:
            existing = _manifest_from_json(manifest_path)
        except (TypeError, ValueError, json.JSONDecodeError):
            existing = None
        if (
            existing is not None
            and existing.input_id == input_id
            and events_path.is_file()
            and items_path.is_file()
            and labels_path.is_file()
            and examples_path.is_file()
            and sha256_file(events_path) == existing.events_sha256
            and sha256_file(items_path) == existing.items_sha256
            and sha256_file(labels_path) == existing.labels_sha256
            and sha256_file(examples_path) == existing.examples_sha256
        ):
            logger.info(
                "ranking_training_cache_reused",
                extra={
                    "event": "ranking_training_cache_reused",
                    "stage": "ranking_training_cache",
                    "status": "passed",
                    "sessions": existing.examples,
                    "events": existing.event_rows,
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

    sessions_seen = 0
    examples = 0
    event_rows = 0
    item_rows = 0
    label_rows = 0
    click_labels = 0
    cart_labels = 0
    order_labels = 0
    started = time.perf_counter()

    progress: dict[str, int] = {
        "sessions": 0,
        "examples": 0,
        "events": 0,
    }

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

    logger.info(
        "ranking_training_cache_start",
        extra={
            "event": "ranking_training_cache_start",
            "stage": "ranking_training_cache",
            "source_manifest_id": source_id,
            "input_id": input_id,
            "sample_denominator": sample_denominator,
            "sample_remainder": sample_remainder,
            "max_examples": max_examples,
        },
    )

    try:
        with Heartbeat(
            logger,
            stage="ranking_training_cache",
            interval_seconds=heartbeat_seconds,
            progress_provider=snapshot,
        ), source.open("rb") as handle:
            for line in handle:
                record = orjson.loads(line)
                session = int(record["session"])
                sessions_seen += 1
                progress["sessions"] = sessions_seen

                if not session_is_selected(
                    session,
                    seed=seed,
                    denominator=sample_denominator,
                    remainder=sample_remainder,
                ):
                    continue

                raw_events = record.get("events")
                if not isinstance(raw_events, list) or len(raw_events) <= min_prefix_events:
                    continue

                session, prefix, future, cut_index = split_training_example(
                    record,
                    seed=seed,
                    min_prefix_events=min_prefix_events,
                    max_prefix_events=max_prefix_events,
                )
                labels = future_labels(future, session=session)
                bucket = session % buckets

                event_buffer.extend(_event_rows(session, prefix, bucket=bucket))
                item_buffer.extend(_item_rows(session, prefix, bucket=bucket))
                rows = _label_rows(session, labels, bucket=bucket)
                label_buffer.extend(rows)
                example_buffer.append(
                    (
                        session,
                        len(raw_events),
                        cut_index,
                        len(prefix),
                        len(future),
                        bucket,
                    )
                )

                examples += 1
                progress["examples"] = examples
                click_labels += 1
                cart_labels += sum(1 for row in rows if row[1] == "carts")
                order_labels += sum(1 for row in rows if row[1] == "orders")

                if examples % flush_examples == 0:
                    flush_buffers()

                if max_examples is not None and examples >= max_examples:
                    break

        flush_buffers()
    finally:
        for writer in writers.values():
            writer.close()

    if examples <= 0 or event_rows <= 0 or item_rows <= 0 or label_rows <= 0:
        for path in temp_paths.values():
            path.unlink(missing_ok=True)
        raise RuntimeError("ranking training cache is empty")

    os.replace(temp_paths["events"], events_path)
    os.replace(temp_paths["items"], items_path)
    os.replace(temp_paths["labels"], labels_path)
    os.replace(temp_paths["examples"], examples_path)

    elapsed = round(time.perf_counter() - started, 3)
    manifest = RankingTrainingCacheManifest(
        source_manifest_id=source_id,
        input_id=input_id,
        config=config,
        sessions_seen=sessions_seen,
        examples=examples,
        event_rows=event_rows,
        item_rows=item_rows,
        label_rows=label_rows,
        click_labels=click_labels,
        cart_labels=cart_labels,
        order_labels=order_labels,
        events_sha256=sha256_file(events_path),
        items_sha256=sha256_file(items_path),
        labels_sha256=sha256_file(labels_path),
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
            "sessions": examples,
            "events": event_rows,
            "label_rows": label_rows,
            "elapsed_seconds": elapsed,
        },
    )
    return manifest
