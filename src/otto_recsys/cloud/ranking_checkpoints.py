"""Durable ranking feature buckets through the Studio AWS CLI credential chain."""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

from otto_recsys.ranking.feature_cache import FAMILIES, valid_part


class S3FeatureCheckpoints:
    """Publish bucket data before receipts; restore one verified bucket at a time."""

    def __init__(self, uri: str, *, region: str, logger: logging.Logger) -> None:
        parsed = urlsplit(uri)
        if (
            parsed.scheme != "s3"
            or not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", parsed.netloc)
            or not parsed.path.strip("/")
            or parsed.query
            or parsed.fragment
            or any(part in {".", ".."} for part in parsed.path.split("/"))
        ):
            raise ValueError("checkpoint URI must be s3://bucket/project-prefix")
        self.root = uri.rstrip("/")
        self.region = region
        self.logger = logger
        self.uri: str | None = None
        self.remote_receipts: dict[int, dict] = {}

    def run(self, arguments: list[str], *, allow_missing: bool = False) -> bool:
        started = time.perf_counter()
        result = subprocess.run(
            ["aws", "--region", self.region, "s3", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        if result.returncode:
            if allow_missing and ("(404)" in result.stderr or "NoSuchKey" in result.stderr):
                return False
            raise RuntimeError(f"ranking checkpoint transfer failed: {result.stderr.strip()}")
        self.logger.info(
            "ranking_checkpoint_transfer",
            extra={
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "status": "passed",
            },
        )

        return True

    def upload(self, path: Path, relative: str) -> None:
        if self.uri is None:
            raise RuntimeError("initialize ranking checkpoint storage first")
        self.run(["cp", str(path), f"{self.uri}/{relative}", "--only-show-errors"])

    def restore(self, directory: Path, input_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", input_id):
            raise ValueError("invalid feature identity")
        self.uri = f"{self.root}/{input_id}"
        self.remote_receipts.clear()
        expected = json.loads((directory / "feature_contract.json").read_text())
        restored = 0
        with tempfile.TemporaryDirectory(dir=directory, prefix="staging-") as name:
            staging = Path(name)
            self.run(
                [
                    "sync",
                    self.uri + "/",
                    str(staging),
                    "--exclude",
                    "*",
                    "--include",
                    "feature_contract.json",
                    "--include",
                    "parts/part-*.json",
                    "--only-show-errors",
                ]
            )
            remote_contract = staging / "feature_contract.json"
            if remote_contract.exists() and json.loads(remote_contract.read_text()) != expected:
                raise ValueError("remote feature contract mismatch")
            for receipt_path in sorted((staging / "parts").glob("part-*.json")):
                if not re.fullmatch(r"part-\d{3}\.json", receipt_path.name):
                    continue
                if not remote_contract.exists():
                    raise ValueError("remote feature parts have no contract")
                bucket = int(receipt_path.stem.removeprefix("part-"))
                if bucket >= expected["buckets"]:
                    raise ValueError("remote feature bucket is outside contract")
                try:
                    receipt = json.loads(receipt_path.read_text())
                except ValueError:
                    continue
                local = valid_part(directory, bucket, input_id)
                if local is not None and local == receipt:
                    self.remote_receipts[bucket] = receipt
                    continue
                available = True
                for family in FAMILIES:
                    relative = f"parts/{receipt_path.stem}/{family}.parquet"
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not self.run(
                        ["cp", f"{self.uri}/{relative}", str(path), "--only-show-errors"],
                        allow_missing=True,
                    ):
                        available = False
                        break
                verified = valid_part(staging, bucket, input_id) if available else None
                if verified is None:
                    self.logger.info("ranking_remote_bucket_rejected", extra={"bucket": bucket})
                    shutil.rmtree(staging / "parts" / receipt_path.stem)
                    continue
                destination = directory / "parts"
                destination.mkdir(parents=True, exist_ok=True)
                source_dir = staging / "parts" / receipt_path.stem
                target_dir = destination / receipt_path.stem
                target_dir.mkdir(parents=True, exist_ok=True)
                for family in FAMILIES:
                    shutil.move(
                        str(source_dir / f"{family}.parquet"), target_dir / f"{family}.parquet"
                    )
                # A crash during file moves leaves no matching complete receipt.
                shutil.move(str(receipt_path), destination / receipt_path.name)
                self.remote_receipts[bucket] = verified
                restored += 1
        self.upload(directory / "feature_contract.json", "feature_contract.json")
        self.logger.info(
            "ranking_checkpoint_ready", extra={"restored_parts": restored, "uri": self.uri}
        )

    def publish_part(self, directory: Path, bucket: int, input_id: str) -> None:
        receipt = valid_part(directory, bucket, input_id)
        if receipt is None:
            raise ValueError("refusing to publish an invalid feature bucket")
        if self.remote_receipts.get(bucket) == receipt:
            return
        stem = f"part-{bucket:03d}"
        for family in FAMILIES:
            relative = f"parts/{stem}/{family}.parquet"
            self.upload(directory / relative, relative)
        self.upload(directory / "parts" / f"{stem}.json", f"parts/{stem}.json")
        self.remote_receipts[bucket] = receipt
        self.publish_logs(directory)
        self.logger.info("ranking_feature_bucket_durable", extra={"bucket": bucket})

    def publish_logs(self, directory: Path) -> None:
        path = directory / "logs/ranking_features.jsonl"
        if path.is_file() and self.uri is not None:
            with tempfile.TemporaryDirectory(dir=directory, prefix="log-") as temporary:
                snapshot = Path(temporary) / path.name
                shutil.copyfile(path, snapshot)
                self.upload(snapshot, "logs/" + path.name)

    def publish_summary(self, directory: Path) -> None:
        self.upload(directory / "manifest.json", "manifest.json")
        self.publish_logs(directory)
