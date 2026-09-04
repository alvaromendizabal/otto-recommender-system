from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

import orjson
import pyarrow as pa
import pyarrow.parquet as pq

from otto_recsys.data.schema import ACTION_TO_ID, EVENT_SCHEMA
from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class ConversionManifest:
    """Durable resume state for one raw-to-Parquet conversion."""

    schema_version: int
    source_name: str
    source_size_bytes: int
    source_sha256: str
    source_manifest_id: str

    input_offset: int
    parts_written: int
    sessions_processed: int
    events_processed: int

    status: str


def _write_manifest_atomic(
    manifest: ConversionManifest,
    path: Path,
) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_manifest(path: Path) -> ConversionManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("conversion manifest must contain an object")

    return ConversionManifest(**payload)


def _raw_source_identity(
    raw_manifest_path: str | Path,
    source_path: Path,
) -> tuple[str, str]:
    payload = json.loads(
        Path(raw_manifest_path).read_text(encoding="utf-8")
    )

    manifest_id = payload.get("manifest_id")
    files = payload.get("files")

    if not isinstance(manifest_id, str):
        raise ValueError("raw manifest has no manifest_id")

    if not isinstance(files, list):
        raise ValueError("raw manifest has no files list")

    for record in files:
        if (
            isinstance(record, dict)
            and record.get("name") == source_path.name
        ):
            source_sha256 = record.get("sha256")
            expected_size = record.get("size_bytes")

            if not isinstance(source_sha256, str):
                raise ValueError("source record has no SHA-256")

            if expected_size != source_path.stat().st_size:
                raise RuntimeError(
                    "raw source size does not match raw manifest"
                )

            return manifest_id, source_sha256

    raise ValueError(
        f"{source_path.name} is not present in the raw manifest"
    )


def _empty_columns() -> dict[str, list[int]]:
    return {
        "session": [],
        "aid": [],
        "ts": [],
        "event_type": [],
        "event_index": [],
    }


def _write_part(
    output_dir: Path,
    part_number: int,
    columns: dict[str, list[int]],
) -> tuple[Path, int]:
    rows = len(columns["session"])

    if rows == 0:
        raise ValueError("cannot write an empty Parquet part")

    table = pa.Table.from_pydict(
        columns,
        schema=EVENT_SCHEMA,
    )

    final_path = output_dir / f"part-{part_number:06d}.parquet"
    temp_path = output_dir / f".part-{part_number:06d}.parquet.tmp"

    pq.write_table(
        table,
        temp_path,
        compression="zstd",
        compression_level=3,
        use_dictionary=["event_type"],
        write_statistics=True,
        row_group_size=min(rows, 250_000),
    )

    os.replace(temp_path, final_path)

    return final_path, rows


def _reconcile_output(
    output_dir: Path,
    manifest: ConversionManifest,
) -> None:
    """Remove uncommitted remnants and verify committed parts."""
    for temp_path in output_dir.glob(".*.tmp"):
        temp_path.unlink()

    committed = {
        output_dir / f"part-{index:06d}.parquet"
        for index in range(manifest.parts_written)
    }

    missing = [
        path
        for path in sorted(committed)
        if not path.exists()
    ]

    if missing:
        raise RuntimeError(
            f"committed Parquet parts are missing: {missing[:5]}"
        )

    for part_path in output_dir.glob("part-*.parquet"):
        part_number = int(part_path.stem.split("-")[1])

        if part_number >= manifest.parts_written:
            part_path.unlink()


def convert_jsonl_to_parquet(
    input_path: str | Path,
    output_dir: str | Path,
    raw_manifest_path: str | Path,
    *,
    logger: logging.Logger,
    events_per_part: int = 1_000_000,
    heartbeat_seconds: float = 30.0,
    max_sessions: int | None = None,
) -> ConversionManifest:
    """Convert OTTO JSONL to typed Parquet with exact resume semantics."""
    if events_per_part <= 0:
        raise ValueError("events_per_part must be positive")

    if max_sessions is not None and max_sessions <= 0:
        raise ValueError("max_sessions must be positive")

    source = Path(input_path).resolve()
    destination = Path(output_dir).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    destination.mkdir(parents=True, exist_ok=True)

    manifest_path = destination / "manifest.json"

    raw_manifest_id, source_sha256 = _raw_source_identity(
        raw_manifest_path,
        source,
    )

    source_size = source.stat().st_size

    if manifest_path.exists():
        manifest = _read_manifest(manifest_path)

        expected_identity = (
            source.name,
            source_size,
            source_sha256,
            raw_manifest_id,
        )
        observed_identity = (
            manifest.source_name,
            manifest.source_size_bytes,
            manifest.source_sha256,
            manifest.source_manifest_id,
        )

        if expected_identity != observed_identity:
            raise RuntimeError(
                "conversion source identity differs from existing manifest"
            )
    else:
        if any(destination.glob("part-*.parquet")):
            raise RuntimeError(
                "Parquet parts exist without a conversion manifest"
            )

        manifest = ConversionManifest(
            schema_version=1,
            source_name=source.name,
            source_size_bytes=source_size,
            source_sha256=source_sha256,
            source_manifest_id=raw_manifest_id,
            input_offset=0,
            parts_written=0,
            sessions_processed=0,
            events_processed=0,
            status="running",
        )
        _write_manifest_atomic(manifest, manifest_path)

    _reconcile_output(destination, manifest)

    if manifest.status == "complete":
        logger.info(
            "conversion_already_complete",
            extra={
                "event": "conversion_already_complete",
                "stage": "conversion",
                "sessions": manifest.sessions_processed,
                "events": manifest.events_processed,
            },
        )
        return manifest

    progress = {
        "sessions": manifest.sessions_processed,
        "events": manifest.events_processed,
        "part": manifest.parts_written,
    }

    columns = _empty_columns()
    started = time.perf_counter()

    def progress_snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["events"] / elapsed, 1),
        }

    def commit_part(
        handle: BinaryIO,
        current_manifest: ConversionManifest,
    ) -> ConversionManifest:
        nonlocal columns

        if not columns["session"]:
            return current_manifest

        part_number = current_manifest.parts_written
        part_path, rows = _write_part(
            destination,
            part_number,
            columns,
        )

        updated = replace(
            current_manifest,
            input_offset=handle.tell(),
            parts_written=part_number + 1,
            sessions_processed=progress["sessions"],
            events_processed=progress["events"],
            status="running",
        )

        _write_manifest_atomic(updated, manifest_path)

        logger.info(
            "part_committed",
            extra={
                "event": "part_committed",
                "stage": "conversion",
                "part": part_number,
                "rows": rows,
                "path": str(part_path),
                "sessions": progress["sessions"],
                "events": progress["events"],
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    1,
                ),
            },
        )

        columns = _empty_columns()
        progress["part"] = updated.parts_written

        return updated

    logger.info(
        "conversion_start",
        extra={
            "event": "conversion_start",
            "stage": "conversion",
            "source": str(source),
            "input_offset": manifest.input_offset,
            "sessions": manifest.sessions_processed,
            "events": manifest.events_processed,
        },
    )

    with (
        Heartbeat(
            logger,
            stage="conversion",
            interval_seconds=heartbeat_seconds,
            progress_provider=progress_snapshot,
        ),
        source.open("rb") as handle,
    ):
        handle.seek(manifest.input_offset)

        while True:
            if (
                max_sessions is not None
                and progress["sessions"] >= max_sessions
            ):
                break

            line = handle.readline()

            if not line:
                break

            if not line.strip():
                raise ValueError(
                    f"empty source line near byte offset {handle.tell()}"
                )

            record: Any = orjson.loads(line)
            session = record["session"]
            events = record["events"]

            if not isinstance(session, int) or session < 0:
                raise ValueError("invalid session identifier")

            if not isinstance(events, list) or not events:
                raise ValueError(
                    f"session {session} has no events"
                )

            if len(events) > 65_535:
                raise ValueError(
                    f"session {session} exceeds uint16 event-index range"
                )

            for event_index, event in enumerate(events):
                aid = event["aid"]
                ts = event["ts"]
                action = event["type"]

                if not isinstance(aid, int) or aid < 0:
                    raise ValueError(
                        f"session {session}: invalid aid"
                    )

                if not isinstance(ts, int) or ts < 0:
                    raise ValueError(
                        f"session {session}: invalid timestamp"
                    )

                if action not in ACTION_TO_ID:
                    raise ValueError(
                        f"session {session}: invalid action {action!r}"
                    )

                columns["session"].append(session)
                columns["aid"].append(aid)
                columns["ts"].append(ts)
                columns["event_type"].append(
                    ACTION_TO_ID[action]
                )
                columns["event_index"].append(event_index)

            progress["sessions"] += 1
            progress["events"] += len(events)

            # Flush only after the full session has been captured.
            if len(columns["session"]) >= events_per_part:
                manifest = commit_part(handle, manifest)

        manifest = commit_part(handle, manifest)

        reached_eof = handle.tell() >= source_size

    final_status = "complete" if reached_eof else "partial"

    manifest = replace(
        manifest,
        sessions_processed=progress["sessions"],
        events_processed=progress["events"],
        status=final_status,
    )

    _write_manifest_atomic(manifest, manifest_path)

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "conversion_complete",
        extra={
            "event": "conversion_complete",
            "stage": "conversion",
            "status": final_status,
            "sessions": manifest.sessions_processed,
            "events": manifest.events_processed,
            "elapsed_seconds": elapsed,
        },
    )

    return manifest
