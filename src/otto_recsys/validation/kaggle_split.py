from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson

from otto_recsys.runtime import Heartbeat

DAY_MILLIS = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class ValidationManifest:
    """Immutable description of one local OTTO validation benchmark."""

    manifest_id: str
    created_at_utc: str
    raw_manifest_id: str

    days: int
    seed: int
    max_ts: int
    split_ts: int

    train_sessions: int
    train_events: int
    known_items: int

    test_sessions: int
    click_labels: int
    cart_label_items: int
    order_label_items: int


def _read_raw_manifest_id(path: str | Path) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("raw manifest must contain a JSON object")

    manifest_id = payload.get("manifest_id")

    if not isinstance(manifest_id, str) or not manifest_id:
        raise ValueError("raw manifest does not contain manifest_id")

    return manifest_id


def _validation_id(payload: dict[str, int | str]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def _future_labels(
    future_events: list[dict[str, Any]],
) -> dict[str, int | list[int]]:
    """Build OTTO click/cart/order ground truth after a session prefix."""
    labels: dict[str, int | list[int]] = {}

    for event in future_events:
        if event["type"] == "clicks":
            labels["clicks"] = int(event["aid"])
            break

    carts = sorted(
        {
            int(event["aid"])
            for event in future_events
            if event["type"] == "carts"
        }
    )
    orders = sorted(
        {
            int(event["aid"])
            for event in future_events
            if event["type"] == "orders"
        }
    )

    if carts:
        labels["carts"] = carts

    if orders:
        labels["orders"] = orders

    return labels


def build_validation(
    source_path: str | Path,
    raw_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    max_ts: int,
    days: int,
    seed: int,
    logger: logging.Logger,
    heartbeat_seconds: float = 30.0,
) -> ValidationManifest:
    """Reproduce OTTO's time-based local test-generation semantics."""
    if days <= 0:
        raise ValueError("days must be positive")

    if max_ts <= 0:
        raise ValueError("max_ts must be positive")

    source = Path(source_path).resolve()
    destination = Path(output_dir).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    destination.mkdir(parents=True, exist_ok=True)

    raw_manifest_id = _read_raw_manifest_id(raw_manifest_path)
    split_ts = max_ts - (days * DAY_MILLIS)

    if split_ts <= 0:
        raise ValueError("computed split timestamp is invalid")

    train_output = destination / "train_sessions.jsonl"
    test_output = destination / "test_sessions.jsonl"
    labels_output = destination / "test_labels.jsonl"

    train_temp = destination / ".train_sessions.jsonl.tmp"
    test_temp = destination / ".test_sessions.jsonl.tmp"
    labels_temp = destination / ".test_labels.jsonl.tmp"

    for temporary in (train_temp, test_temp, labels_temp):
        temporary.unlink(missing_ok=True)

    known_items: set[int] = set()

    train_sessions = 0
    train_events = 0

    first_progress: dict[str, int] = {
        "sessions": 0,
        "events": 0,
        "train_sessions": 0,
    }
    first_started = time.perf_counter()

    def first_snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - first_started, 1e-9)
        return {
            **first_progress,
            "throughput": round(first_progress["events"] / elapsed, 1),
        }

    logger.info(
        "validation_training_start",
        extra={
            "event": "validation_training_start",
            "stage": "validation_training",
            "max_ts": max_ts,
            "split_ts": split_ts,
            "days": days,
            "seed": seed,
        },
    )

    try:
        with (
            Heartbeat(
                logger,
                stage="validation_training",
                interval_seconds=heartbeat_seconds,
                progress_provider=first_snapshot,
            ),
            source.open("rb") as source_handle,
            train_temp.open("wb") as train_handle,
        ):
            for line in source_handle:
                record = orjson.loads(line)

                session = int(record["session"])
                training_events: list[dict[str, Any]] = record["events"]

                first_progress["sessions"] += 1
                first_progress["events"] += len(training_events)

                if not training_events:
                    raise ValueError(
                        f"session {session} contains no events"
                    )

                if int(training_events[0]["ts"]) > split_ts:
                    continue

                trimmed_events = [
                    event
                    for event in training_events
                    if int(event["ts"]) < split_ts
                ]

                if len(trimmed_events) < 2:
                    continue

                train_sessions += 1
                train_events += len(trimmed_events)
                first_progress["train_sessions"] = train_sessions

                for event in trimmed_events:
                    known_items.add(int(event["aid"]))

                train_handle.write(
                    orjson.dumps(
                        {
                            "session": session,
                            "events": trimmed_events,
                        }
                    )
                    + b"\n"
                )

        os.replace(train_temp, train_output)

        logger.info(
            "validation_training_complete",
            extra={
                "event": "validation_training_complete",
                "stage": "validation_training",
                "train_sessions": train_sessions,
                "train_events": train_events,
                "known_items": len(known_items),
                "elapsed_seconds": round(
                    time.perf_counter() - first_started,
                    3,
                ),
            },
        )

        rng = random.Random(seed)

        test_sessions = 0
        click_labels = 0
        cart_label_items = 0
        order_label_items = 0

        second_progress: dict[str, int] = {
            "sessions": 0,
            "events": 0,
            "test_sessions": 0,
        }
        second_started = time.perf_counter()

        def second_snapshot() -> dict[str, int | float]:
            elapsed = max(time.perf_counter() - second_started, 1e-9)
            return {
                **second_progress,
                "throughput": round(
                    second_progress["events"] / elapsed,
                    1,
                ),
            }

        logger.info(
            "validation_test_start",
            extra={
                "event": "validation_test_start",
                "stage": "validation_test",
            },
        )

        with (
            Heartbeat(
                logger,
                stage="validation_test",
                interval_seconds=heartbeat_seconds,
                progress_provider=second_snapshot,
            ),
            source.open("rb") as source_handle,
            test_temp.open("wb") as test_handle,
            labels_temp.open("wb") as labels_handle,
        ):
            for line in source_handle:
                record = orjson.loads(line)

                session = int(record["session"])
                candidate_events: list[dict[str, Any]] = record["events"]

                second_progress["sessions"] += 1
                second_progress["events"] += len(candidate_events)

                if not candidate_events:
                    raise ValueError(
                        f"session {session} contains no events"
                    )

                if int(candidate_events[0]["ts"]) <= split_ts:
                    continue

                filtered_events = [
                    event
                    for event in candidate_events
                    if int(event["aid"]) in known_items
                ]

                if len(filtered_events) < 2:
                    continue

                # Equivalent to the organizer's split_events(): retain between
                # one and len(events)-1 observed events.
                split_index = rng.randint(1, len(filtered_events) - 1)

                observed_events = filtered_events[:split_index]
                future_events = filtered_events[split_index:]
                labels = _future_labels(future_events)

                test_handle.write(
                    orjson.dumps(
                        {
                            "session": session,
                            "events": observed_events,
                        }
                    )
                    + b"\n"
                )

                labels_handle.write(
                    orjson.dumps(
                        {
                            "session": session,
                            "labels": labels,
                        }
                    )
                    + b"\n"
                )

                test_sessions += 1
                second_progress["test_sessions"] = test_sessions

                if "clicks" in labels:
                    click_labels += 1

                carts = labels.get("carts")
                orders = labels.get("orders")

                if isinstance(carts, list):
                    cart_label_items += len(carts)

                if isinstance(orders, list):
                    order_label_items += len(orders)

        os.replace(test_temp, test_output)
        os.replace(labels_temp, labels_output)

    except BaseException:
        for temporary in (train_temp, test_temp, labels_temp):
            temporary.unlink(missing_ok=True)
        raise

    identity_payload: dict[str, int | str] = {
        "raw_manifest_id": raw_manifest_id,
        "days": days,
        "seed": seed,
        "max_ts": max_ts,
        "split_ts": split_ts,
        "train_sessions": train_sessions,
        "train_events": train_events,
        "known_items": len(known_items),
        "test_sessions": test_sessions,
        "click_labels": click_labels,
        "cart_label_items": cart_label_items,
        "order_label_items": order_label_items,
    }

    manifest = ValidationManifest(
        manifest_id=_validation_id(identity_payload),
        created_at_utc=datetime.now(UTC).isoformat(
            timespec="milliseconds"
        ),
        raw_manifest_id=raw_manifest_id,
        days=days,
        seed=seed,
        max_ts=max_ts,
        split_ts=split_ts,
        train_sessions=train_sessions,
        train_events=train_events,
        known_items=len(known_items),
        test_sessions=test_sessions,
        click_labels=click_labels,
        cart_label_items=cart_label_items,
        order_label_items=order_label_items,
    )

    manifest_path = destination / "manifest.json"
    manifest_temp = destination / ".manifest.json.tmp"

    manifest_temp.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temp, manifest_path)

    logger.info(
        "validation_complete",
        extra={
            "event": "validation_complete",
            "stage": "validation",
            "status": "passed",
            "test_sessions": test_sessions,
            "elapsed_seconds": round(
                time.perf_counter() - second_started,
                3,
            ),
            "manifest_id": manifest.manifest_id,
        },
    )

    return manifest
