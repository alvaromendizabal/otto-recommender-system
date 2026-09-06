"""Durable comparison parts, using the Studio AWS CLI credential chain."""

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

from otto_recsys.experiments.manifest import sha256_file


class S3ComparisonCheckpoints:
    """Store immutable-input namespaces; publish each data file before its receipt."""

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
        if not region:
            raise ValueError("AWS region is required")
        self.root_uri = uri.rstrip("/")
        self.region = region
        self.logger = logger
        self.uri: str | None = None

    def _run(self, arguments: list[str]) -> None:
        started = time.perf_counter()
        completed = subprocess.run(
            ["aws", "--region", self.region, *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
        status = "passed" if completed.returncode == 0 else "failed"
        self.logger.info(
            "checkpoint_transfer",
            extra={
                "stage": "s3_checkpoint",
                "status": status,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            },
        )
        if completed.returncode:
            raise RuntimeError(f"S3 checkpoint transfer failed: {completed.stderr.strip()}")

    def restore(self, directory: Path, input_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", input_id):
            raise ValueError("invalid comparison identity")
        self.uri = f"{self.root_uri}/{input_id}"
        contract = directory / "comparison_contract.json"
        expected_contract = json.loads(contract.read_text())
        restored = 0
        with tempfile.TemporaryDirectory(prefix="otto-comparison-") as temporary:
            staging = Path(temporary)
            self._run(
                [
                    "s3",
                    "sync",
                    self.uri + "/",
                    str(staging),
                    "--exclude",
                    "*",
                    "--include",
                    "comparison_contract.json",
                    "--include",
                    "parts/*",
                    "--only-show-errors",
                ]
            )
            remote_contract = staging / "comparison_contract.json"
            if (
                remote_contract.is_file()
                and json.loads(remote_contract.read_text()) != expected_contract
            ):
                raise ValueError("S3 comparison contract mismatch")
            for receipt_path in sorted((staging / "parts").glob("part-*.json")):
                if not re.fullmatch(r"part-\d{3}\.json", receipt_path.name):
                    continue
                if not remote_contract.is_file():
                    raise ValueError("S3 comparison parts lack their contract")
                data_path = receipt_path.with_suffix(".npz")
                try:
                    receipt = json.loads(receipt_path.read_text())
                    valid = (
                        isinstance(receipt, dict)
                        and receipt.get("input_id") == input_id
                        and data_path.is_file()
                        and receipt.get("sha256") == sha256_file(data_path)
                    )
                except (OSError, ValueError):
                    valid = False
                if not valid:
                    self.logger.warning(
                        "remote_comparison_part_invalid", extra={"part": receipt_path.stem}
                    )
                    continue
                local = directory / "parts" / data_path.name
                if local.is_file() and sha256_file(local) == receipt["sha256"]:
                    # Preserve good local data; refresh its receipt if needed.
                    target = directory / "parts" / receipt_path.name
                    temporary_path = target.with_suffix(".json.tmp")
                    temporary_path.write_bytes(receipt_path.read_bytes())
                    temporary_path.replace(target)
                    continue
                local.parent.mkdir(parents=True, exist_ok=True)
                for source in (data_path, receipt_path):
                    target = local.parent / source.name
                    temporary_path = target.with_suffix(target.suffix + ".tmp")
                    shutil.copyfile(source, temporary_path)
                    temporary_path.replace(target)
                restored += 1
        # Also verifies write access before loading the baseline or doing computation.
        self._upload(contract, "comparison_contract.json")
        self.logger.info(
            "comparison_checkpoint_ready", extra={"restored_parts": restored, "uri": self.uri}
        )

    def _upload(self, path: Path, relative: str) -> None:
        if self.uri is None:
            raise RuntimeError("initialize checkpoint storage before publishing")
        self._run(["s3", "cp", str(path), f"{self.uri}/{relative}", "--only-show-errors"])

    def publish_part(self, directory: Path, bucket: int, input_id: str) -> None:
        if bucket < 0:
            raise ValueError("invalid bucket")
        data = directory / "parts" / f"part-{bucket:03d}.npz"
        receipt_path = data.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("input_id") != input_id or receipt.get("sha256") != sha256_file(data):
            raise ValueError("refusing to publish an invalid comparison part")
        self._upload(data, f"parts/{data.name}")
        self._upload(receipt_path, f"parts/{receipt_path.name}")
        self.logger.info("comparison_part_durable", extra={"bucket": bucket})
        self.publish_logs(directory)

    def publish_metrics(self, directory: Path) -> None:
        self._upload(directory / "metrics.json", "metrics.json")

    def publish_logs(self, directory: Path) -> None:
        path = directory / "logs" / "two_tower_comparison.jsonl"
        if self.uri is not None and path.is_file():
            # A snapshot avoids racing the heartbeat while the CLI reads the log.
            with tempfile.TemporaryDirectory(prefix="otto-comparison-log-") as temporary:
                snapshot = Path(temporary) / path.name
                shutil.copyfile(path, snapshot)
                self._upload(snapshot, "logs/" + path.name)
