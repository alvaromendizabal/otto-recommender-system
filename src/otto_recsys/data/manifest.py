from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from otto_recsys.runtime import Heartbeat


@dataclass(frozen=True)
class FileRecord:
    name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class RawDatasetManifest:
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
    """Hash a potentially huge file with bounded memory and heartbeat."""
    source = Path(path)
    total_bytes = source.stat().st_size
    progress: dict[str, int] = {
        "bytes_processed": 0,
        "total_bytes": total_bytes,
    }

    digest = hashlib.sha256()
    started = time.perf_counter()

    with Heartbeat(
        logger,
        stage=f"sha256:{source.name}",
        interval_seconds=heartbeat_seconds,
        progress_provider=lambda: dict(progress),
    ):
        with source.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
                progress["bytes_processed"] += len(chunk)

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "hash_complete",
        extra={
            "event": "hash_complete",
            "file": source.name,
            "bytes_processed": total_bytes,
            "elapsed_seconds": elapsed,
        },
    )

    return FileRecord(
        name=source.name,
        size_bytes=total_bytes,
        sha256=digest.hexdigest(),
    )


def build_manifest(
    paths: list[str | Path],
    *,
    source: str,
    logger: logging.Logger,
) -> RawDatasetManifest:
    records = [hash_file(path, logger=logger) for path in paths]

    return RawDatasetManifest(
        manifest_id=str(uuid.uuid4()),
        source=source,
        created_at_utc=datetime.now(UTC).isoformat(timespec="milliseconds"),
        files=records,
    )


def write_manifest(
    manifest: RawDatasetManifest,
    path: str | Path,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    temp = destination.with_suffix(destination.suffix + ".tmp")
    temp.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(destination)
