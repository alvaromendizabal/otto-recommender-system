from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class FileRecord:
    """Immutable identity for one source file."""

    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RawDatasetManifest:
    """Content-addressed identity for an immutable raw dataset."""

    manifest_id: str
    source: str
    created_at_utc: str
    files: list[FileRecord]


def hash_file(
    path: str | Path,
    *,
    logger: logging.Logger,
    heartbeat_seconds: float = 30.0,
    chunk_size: int = 16 * 1024 * 1024,
) -> FileRecord:
    """Hash a large file with bounded memory, progress logs, and heartbeat."""
    source = Path(path).resolve()

    if not source.is_file():
        raise FileNotFoundError(source)

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    total_bytes = source.stat().st_size
    progress = {
        "bytes_processed": 0,
        "total_bytes": total_bytes,
    }

    digest = hashlib.sha256()
    started = time.perf_counter()
    progress_interval_bytes = 512 * 1024 * 1024
    next_progress_log = progress_interval_bytes

    def progress_snapshot() -> dict[str, int | float]:
        elapsed = max(time.perf_counter() - started, 1e-9)
        return {
            **progress,
            "throughput": round(progress["bytes_processed"] / elapsed / (1024**2), 2),
        }

    logger.info(
        "hash_start",
        extra={
            "event": "hash_start",
            "stage": f"sha256:{source.name}",
            "file": source.name,
            "total_bytes": total_bytes,
        },
    )

    with (
        Heartbeat(
            logger,
            stage=f"sha256:{source.name}",
            interval_seconds=heartbeat_seconds,
            progress_provider=progress_snapshot,
        ),
        source.open("rb") as handle,
    ):
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            progress["bytes_processed"] += len(chunk)

            if progress["bytes_processed"] >= next_progress_log:
                elapsed = max(time.perf_counter() - started, 1e-9)
                throughput = progress["bytes_processed"] / elapsed / (1024**2)

                logger.info(
                    "hash_progress",
                    extra={
                        "event": "hash_progress",
                        "stage": f"sha256:{source.name}",
                        "elapsed_seconds": round(elapsed, 1),
                        "bytes_processed": progress["bytes_processed"],
                        "total_bytes": total_bytes,
                        "throughput": round(throughput, 2),
                    },
                )

                next_progress_log += progress_interval_bytes

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "hash_complete",
        extra={
            "event": "hash_complete",
            "stage": f"sha256:{source.name}",
            "elapsed_seconds": elapsed,
            "bytes_processed": total_bytes,
        },
    )

    return FileRecord(
        name=source.name,
        size_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def _content_manifest_id(
    source: str,
    records: Sequence[FileRecord],
) -> str:
    """Create a deterministic ID from source identity and file contents."""
    payload = {
        "source": source,
        "files": [
            asdict(record)
            for record in sorted(records, key=lambda item: item.name)
        ],
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def build_manifest(
    paths: Sequence[str | Path],
    *,
    source: str,
    logger: logging.Logger,
) -> RawDatasetManifest:
    """Build a deterministic dataset manifest from any path sequence."""
    if not paths:
        raise ValueError("paths must not be empty")

    records = [
        hash_file(path, logger=logger)
        for path in paths
    ]

    records.sort(key=lambda item: item.name)

    return RawDatasetManifest(
        manifest_id=_content_manifest_id(source, records),
        source=source,
        created_at_utc=datetime.now(UTC).isoformat(timespec="milliseconds"),
        files=records,
    )


def write_manifest(
    manifest: RawDatasetManifest,
    path: str | Path,
) -> None:
    """Atomically write a raw-dataset manifest."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(destination)
