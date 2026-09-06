"""Receipt-committed S3 artifacts with recovery on a fresh worker."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .checkpoint import write_json_atomic
from .evaluation import read_json, sha256_file
from .logging_utils import utc_now_iso


class BenchmarkArtifacts:
    def __init__(
        self,
        root: Path,
        input_id: str,
        logger: logging.Logger,
        *,
        uri: str | None = None,
        region: str = "us-west-2",
        client: Any = None,
    ) -> None:
        self.root, self.input_id, self.logger = root, input_id, logger
        self.client, self.bucket, self.prefix = client, "", ""
        self.used: dict[str, dict[str, Any]] = {}
        root.mkdir(parents=True, exist_ok=True)
        if uri is not None:
            parsed = urlparse(uri)
            if (
                parsed.scheme != "s3"
                or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
                or parsed.query
                or parsed.fragment
                or not parsed.path.strip("/")
                or ".." in PurePosixPath(parsed.path).parts
            ):
                raise ValueError("invalid S3 checkpoint URI")
            self.bucket, self.prefix = parsed.netloc, parsed.path.strip("/") + "/"
            if self.client is None:
                import boto3
                from botocore.config import Config

                self.client = boto3.client(
                    "s3",
                    region_name=region,
                    config=Config(
                        connect_timeout=10,
                        read_timeout=60,
                        retries={"mode": "standard", "total_max_attempts": 4},
                        max_pool_connections=4,
                    ),
                )

    def _path(self, name: str) -> Path:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe artifact path")
        return self.root / relative

    def _remote_bytes(self, name: str) -> bytes | None:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.prefix + name)
        except self.client.exceptions.NoSuchKey:
            return None
        with response["Body"] as body:
            return bytes(body.read())

    def _receipt(self, value: dict[str, Any]) -> None:
        if value.get("input_id") != self.input_id:
            raise ValueError("artifact belongs to a different benchmark")
        if not re.fullmatch("[a-f0-9]{64}", str(value.get("sha256", ""))):
            raise ValueError("invalid artifact receipt hash")

    def _publish(self, name: str, path: Path, receipt: dict[str, Any]) -> None:
        if self.client is None:
            return
        from boto3.s3.transfer import TransferConfig

        self.client.upload_file(
            str(path),
            self.bucket,
            self.prefix + name,
            Config=TransferConfig(
                max_concurrency=4,
                multipart_threshold=16 * 1024**2,
                multipart_chunksize=16 * 1024**2,
            ),
        )
        self.client.put_object(
            Bucket=self.bucket,
            Key=self.prefix + name + ".json",
            Body=json.dumps(receipt, sort_keys=True).encode(),
        )
        self.logger.info("artifact_durable", extra={"file": name, "bytes": path.stat().st_size})
        self.upload_log(self.root / "logs/two_tower_ann.jsonl", "logs/progress.jsonl")

    def get(self, name: str) -> Path | None:
        path = self._path(name)
        receipt_path = path.with_suffix(path.suffix + ".json")
        local = None
        if path.is_file() and receipt_path.is_file():
            try:
                local = read_json(receipt_path)
            except (OSError, json.JSONDecodeError):
                local = None
            if local is not None:
                self._receipt(local)
                if sha256_file(path) != local["sha256"]:
                    local = None
        remote = None
        if self.client is not None:
            data = self._remote_bytes(name + ".json")
            if data is not None:
                remote = json.loads(data)
                self._receipt(remote)
        if local is not None and (remote is None or local["sha256"] == remote["sha256"]):
            if self.client is not None and remote is None:
                self._publish(name, path, local)
            self.used[name] = local
            self.logger.info("artifact_reused", extra={"file": name})
            return path
        if remote is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self.prefix + name)
        except self.client.exceptions.NoSuchKey:
            self.logger.warning("artifact_blob_missing", extra={"file": name})
            return None
        with response["Body"] as body, temporary.open("wb") as handle:
            for block in body.iter_chunks(chunk_size=8 * 1024**2):
                handle.write(block)
        if sha256_file(temporary) != remote["sha256"]:
            temporary.unlink()
            self.logger.warning("artifact_checksum_rejected", extra={"file": name})
            return None
        temporary.replace(path)
        write_json_atomic(remote, receipt_path)
        self.used[name] = remote
        self.logger.info("artifact_restored", extra={"file": name})
        return path

    def produce(self, name: str, writer: Callable[[Path], None]) -> Path:
        existing = self.get(name)
        if existing is not None:
            return existing
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        started = time.perf_counter()
        self.logger.info("artifact_start", extra={"file": name})
        writer(temporary)
        receipt = {
            "input_id": self.input_id,
            "sha256": sha256_file(temporary),
            "bytes": temporary.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "completed_at_utc": utc_now_iso(),
        }
        temporary.replace(path)
        write_json_atomic(receipt, path.with_suffix(path.suffix + ".json"))
        self._publish(name, path, receipt)
        self.used[name] = receipt
        self.logger.info(
            "artifact_complete", extra={"file": name, "elapsed_seconds": receipt["elapsed_seconds"]}
        )
        return path

    def json(self, name: str, producer: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        def write(path: Path) -> None:
            path.write_text(json.dumps(producer(), indent=2, sort_keys=True) + "\n")

        return read_json(self.produce(name, write))

    def upload_log(self, path: Path, name: str) -> None:
        if self.client is not None and path.is_file():
            # Read a stable snapshot while the heartbeat can append to the log.
            self.client.put_object(
                Bucket=self.bucket, Key=self.prefix + name, Body=path.read_bytes()
            )
