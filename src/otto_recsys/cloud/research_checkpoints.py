"""Content-verified research checkpoints through the standard AWS role chain."""

from __future__ import annotations

import importlib
import logging
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from otto_recsys.experiments.manifest import sha256_file

ARTIFACT_SUFFIXES = {".json", ".jsonl", ".parquet", ".txt", ".npz", ".csv", ".gz"}


class ResearchCheckpoints:
    """Single-writer namespace; callers publish data before the matching receipt."""

    def __init__(
        self,
        root: Path,
        uri: str,
        *,
        region: str,
        owner_account: str,
        logger: logging.Logger,
        client: Any = None,
    ) -> None:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not parsed.netloc
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
            or not re.fullmatch(r"\d{12}", owner_account)
            or any(p in {".", ".."} for p in parsed.path.split("/"))
        ):
            raise ValueError(
                "checkpoint destination requires an owned S3 bucket and project prefix"
            )
        self.root = root.resolve()
        self.bucket = parsed.netloc
        self.prefix = parsed.path.strip("/") + "/"
        self.owner = owner_account
        self.logger = logger
        self.client = client
        if self.client is None:
            sdk = importlib.import_module("boto3")
            configuration = importlib.import_module("botocore.config").Config(
                region_name=region,
                connect_timeout=10,
                read_timeout=120,
                retries={"mode": "standard", "total_max_attempts": 6},
            )
            self.client = sdk.client("s3", config=configuration)
        self.published: dict[str, str] = {}

    def publish(self, path: Path) -> None:
        relative = path.resolve().relative_to(self.root).as_posix()
        if path.suffix not in ARTIFACT_SUFFIXES:
            raise ValueError("only declared research artifact formats may be published")
        digest = sha256_file(path)
        if self.published.get(relative) == digest:
            return
        start = time.perf_counter()
        self.client.upload_file(
            str(path),
            self.bucket,
            self.prefix + relative,
            ExtraArgs={
                "Metadata": {"sha256": digest},
                "ChecksumAlgorithm": "SHA256",
                "ExpectedBucketOwner": self.owner,
            },
        )
        head = self.client.head_object(
            Bucket=self.bucket, Key=self.prefix + relative, ExpectedBucketOwner=self.owner
        )
        if (
            head["ContentLength"] != path.stat().st_size
            or head.get("Metadata", {}).get("sha256") != digest
        ):
            raise ValueError("remote research checkpoint does not match its uploaded receipt")
        self.published[relative] = digest
        self.logger.info(
            "research_artifact_durable",
            extra={
                "artifact": relative,
                "elapsed_seconds": time.perf_counter() - start,
                "bytes": path.stat().st_size,
            },
        )

    def restore(self) -> int:
        paginator = self.client.get_paginator("list_objects_v2")
        keys = [
            item["Key"]
            for page in paginator.paginate(
                Bucket=self.bucket, Prefix=self.prefix, ExpectedBucketOwner=self.owner
            )
            for item in page.get("Contents", [])
        ]
        # Receipts arrive after data. Component-specific readers recheck their full contracts.
        keys.sort(key=lambda key: (key.endswith(".json"), key))
        restored = 0
        for key in keys:
            relative = PurePosixPath(key[len(self.prefix) :])
            if relative.is_absolute() or ".." in relative.parts or not relative.parts:
                raise ValueError("remote checkpoint contains an invalid relative path")
            path = self.root.joinpath(*relative.parts)
            if path.suffix not in ARTIFACT_SUFFIXES:
                raise ValueError("remote checkpoint contains an undeclared artifact format")
            if path.resolve().is_relative_to(self.root) is False:
                raise ValueError("checkpoint restoration escapes the research workspace")
            head = self.client.head_object(
                Bucket=self.bucket, Key=key, ExpectedBucketOwner=self.owner
            )
            digest = head.get("Metadata", {}).get("sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("remote checkpoint has no SHA-256 provenance metadata")
            if path.is_file() and sha256_file(path) == digest:
                self.published[str(relative)] = digest
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            self.client.download_file(
                self.bucket, key, str(temporary), ExtraArgs={"ExpectedBucketOwner": self.owner}
            )
            if sha256_file(temporary) != digest:
                temporary.unlink()
                raise ValueError("downloaded checkpoint failed SHA-256 verification")
            temporary.replace(path)
            self.published[str(relative)] = digest
            restored += 1
        self.logger.info("research_artifacts_restored", extra={"restored_files": restored})
        return restored
