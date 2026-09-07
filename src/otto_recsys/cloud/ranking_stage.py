"""Receipt-last S3 recovery for candidate materialization and ranker fits.

Use one writer per remote run namespace. Workspace locks protect local writers;
this adapter deliberately does not pretend to implement a distributed lease.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

from otto_recsys.cloud.ranking_checkpoints import S3FeatureCheckpoints
from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.ranking.candidates import FAMILIES, valid_part


class S3CandidateCheckpoints(S3FeatureCheckpoints):
    """Restore only verified candidate buckets; never mutate observed features."""

    def restore(self, directory: Path, input_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", input_id):
            raise ValueError("invalid candidate identity")
        self.uri = f"{self.root}/{input_id}"
        self.remote_receipts.clear()
        expected = json.loads((directory / "candidate_contract.json").read_text())
        restored = 0
        with tempfile.TemporaryDirectory(dir=directory, prefix="staging-") as temporary:
            staging = Path(temporary)
            self.run(["sync", self.uri + "/", str(staging), "--exclude", "*",
                      "--include", "candidate_contract.json", "--include", "parts/part-*.json",
                      "--only-show-errors"])
            remote = staging / "candidate_contract.json"
            if remote.exists() and json.loads(remote.read_text()) != expected:
                raise ValueError("remote candidate contract mismatch")
            for receipt_path in sorted((staging / "parts").glob("part-*.json")):
                if not re.fullmatch(r"part-\d{3}\.json", receipt_path.name):
                    continue
                if not remote.exists():
                    raise ValueError("remote candidate receipts have no contract")
                bucket = int(receipt_path.stem.removeprefix("part-"))
                if not 0 <= bucket < expected["buckets"]:
                    raise ValueError("remote candidate bucket outside contract")
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
                    if not self.run(["cp", f"{self.uri}/{relative}", str(path),
                                     "--only-show-errors"], allow_missing=True):
                        available = False
                        break
                verified = valid_part(staging, bucket, input_id) if available else None
                if verified is None:
                    self.logger.info("candidate_remote_bucket_rejected", extra={"bucket": bucket})
                    continue
                destination = directory / "parts" / receipt_path.stem
                destination.mkdir(parents=True, exist_ok=True)
                for family in FAMILIES:
                    shutil.move(str(staging / "parts" / receipt_path.stem / f"{family}.parquet"),
                                destination / f"{family}.parquet")
                shutil.move(str(receipt_path), destination.with_suffix(".json"))
                self.remote_receipts[bucket] = verified
                restored += 1
        self.upload(directory / "candidate_contract.json", "candidate_contract.json")
        self.logger.info("candidate_checkpoint_ready", extra={"restored_parts": restored})

    def publish_part(self, directory: Path, bucket: int, input_id: str) -> None:
        receipt = valid_part(directory, bucket, input_id)
        if receipt is None:
            raise ValueError("refusing to publish an invalid candidate bucket")
        if self.remote_receipts.get(bucket) == receipt:
            return
        stem = f"part-{bucket:03d}"
        for family in FAMILIES:
            relative = f"parts/{stem}/{family}.parquet"
            self.upload(directory / relative, relative)
        self.upload(directory / "parts" / f"{stem}.json", f"parts/{stem}.json")
        self.remote_receipts[bucket] = receipt
        self.publish_logs(directory)
        self.logger.info("candidate_bucket_durable", extra={"bucket": bucket})

    def publish_logs(self, directory: Path) -> None:
        for path in sorted((directory / "logs").glob("*.jsonl")):
            with tempfile.TemporaryDirectory(dir=directory, prefix="log-") as temporary:
                snapshot = Path(temporary) / path.name
                shutil.copyfile(path, snapshot)
                self.upload(snapshot, "logs/" + path.name)


_MODEL_PATH = re.compile(
    r"fold-(\d+)/(clicks|carts|orders)/(contract\.json|model\.txt|"
    r"evaluation(?:_receipt)?\.json|checkpoints/\d{6}\.json)"
)


class S3ModelCheckpoints(S3FeatureCheckpoints):
    """Restore isolated run files and publish changed snapshots before receipts."""

    def __init__(self, uri: str, *, region: str, logger: logging.Logger) -> None:
        super().__init__(uri, region=region, logger=logger)
        self._uploaded: dict[str, str] = {}

    def restore(self, directory: Path, input_id: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", input_id):
            raise ValueError("invalid ranker run identity")
        self.uri = f"{self.root}/{input_id}"
        expected = json.loads((directory / "run_contract.json").read_text())
        with tempfile.TemporaryDirectory(dir=directory, prefix="staging-") as temporary:
            staging = Path(temporary)
            self.run(["sync", self.uri + "/", str(staging), "--exclude", "*",
                      "--include", "run_contract.json", "--include", "fold-*/*/contract.json",
                      "--include", "fold-*/*/checkpoints/*.json", "--include", "fold-*/*/model.txt",
                      "--include", "fold-*/*/evaluation*.json", "--only-show-errors"])
            remote = staging / "run_contract.json"
            paths = [p for p in staging.rglob("*") if p.is_file() and p != remote]
            if paths and not remote.exists():
                raise ValueError("remote ranker artifacts have no run contract")
            if remote.exists() and json.loads(remote.read_text()) != expected:
                raise ValueError("remote ranker run contract mismatch")
            for source in paths:
                relative = source.relative_to(staging).as_posix()
                match = _MODEL_PATH.fullmatch(relative)
                if match is None or int(match.group(1)) not in expected["outer_folds"]:
                    raise ValueError("unexpected path in remote ranker namespace")
                target = directory / relative
                # Existing local snapshots are verified by the ranker, not blindly overwritten.
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), target)
        self.upload(directory / "run_contract.json", "run_contract.json")
        self.logger.info("ranker_checkpoint_ready", extra={"uri": self.uri})

    def publish(self, directory: Path) -> None:
        def priority(path: Path) -> tuple[int, str]:
            name = path.name
            order = (0 if name in {"run_contract.json", "contract.json"} else
                     3 if name == "evaluation_receipt.json" else
                     4 if name == "metrics.json" else 2 if name == "evaluation.json" else 1)
            return order, path.as_posix()

        paths = [path for path in directory.rglob("*") if path.is_file()
                 and (path.relative_to(directory).as_posix()
                      in {"run_contract.json", "metrics.json"}
                      or _MODEL_PATH.fullmatch(path.relative_to(directory).as_posix()))]
        for path in sorted(paths, key=priority):
            relative = path.relative_to(directory).as_posix()
            digest = sha256_file(path)
            if self._uploaded.get(relative) != digest:
                self.upload(path, relative)
                self._uploaded[relative] = digest
        for path in sorted((directory / "logs").glob("*.jsonl")):
            with tempfile.TemporaryDirectory(dir=directory, prefix="log-") as temporary:
                snapshot = Path(temporary) / path.name
                shutil.copyfile(path, snapshot)
                self.upload(snapshot, "logs/" + path.name)

    def publish_summary(self, directory: Path) -> None:
        self.publish(directory)
