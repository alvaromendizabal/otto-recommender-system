from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from otto_recsys.cloud.comparison_checkpoints import S3ComparisonCheckpoints
from otto_recsys.experiments.manifest import sha256_file


class LocalS3:
    """Filesystem-backed CLI boundary; exercises the real restore/publish logic."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.writes: list[str] = []
        self.fail_receipt = False

    def run(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        assert command[:4] == ["aws", "--region", "us-west-2", "s3"]
        assert kwargs["timeout"] == 300
        operation, source, target = command[4:7]
        if operation == "sync":
            remote = self.directory / source.removeprefix("s3://")
            destination = Path(target)
            if remote.exists():
                shutil.copytree(remote, destination, dirs_exist_ok=True)
        else:
            assert operation == "cp"
            assert target.startswith("s3://")
            if self.fail_receipt and target.endswith("/parts/part-000.json"):
                return subprocess.CompletedProcess(command, 1, "", "simulated transfer failure")
            destination = self.directory / target.removeprefix("s3://")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            self.writes.append(target)
        return subprocess.CompletedProcess(command, 0, "", "")


@pytest.fixture
def remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalS3:
    fake = LocalS3(tmp_path / "remote")
    monkeypatch.setattr("otto_recsys.cloud.comparison_checkpoints.subprocess.run", fake.run)
    return fake


def store() -> S3ComparisonCheckpoints:
    return S3ComparisonCheckpoints(
        "s3://bucket/comparisons", region="us-west-2", logger=logging.getLogger("checkpoint-test")
    )


def contract(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "comparison_contract.json").write_text('{"version": 1}')


def part(directory: Path, input_id: str) -> None:
    data = directory / "parts/part-000.npz"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(b"verified count arrays")
    data.with_suffix(".json").write_text(
        json.dumps({"input_id": input_id, "sha256": sha256_file(data)})
    )


def test_fresh_workspace_recovers_committed_parts(tmp_path: Path, remote: LocalS3) -> None:
    original = tmp_path / "original"
    contract(original)
    checkpoint = store()
    checkpoint.restore(original, "a" * 64)
    part(original, "a" * 64)
    checkpoint.publish_part(original, 0, "a" * 64)
    assert remote.writes[-2].endswith("part-000.npz")
    assert remote.writes[-1].endswith("part-000.json")
    fresh = tmp_path / "fresh"
    contract(fresh)
    replacement = store()
    replacement.restore(fresh, "a" * 64)
    target = fresh / "parts/part-000.npz"
    assert target.read_bytes() == b"verified count arrays"
    modified = target.stat().st_mtime_ns
    replacement.restore(fresh, "a" * 64)
    assert target.stat().st_mtime_ns == modified
    target.write_bytes(b"corrupt local cache")
    replacement.restore(fresh, "a" * 64)
    assert target.read_bytes() == b"verified count arrays"


def test_interrupted_upload_has_no_committed_remote_part(tmp_path: Path, remote: LocalS3) -> None:
    original = tmp_path / "original"
    contract(original)
    checkpoint = store()
    checkpoint.restore(original, "b" * 64)
    part(original, "b" * 64)
    remote.fail_receipt = True
    with pytest.raises(RuntimeError, match="transfer failed"):
        checkpoint.publish_part(original, 0, "b" * 64)
    fresh = tmp_path / "fresh"
    contract(fresh)
    store().restore(fresh, "b" * 64)
    assert not (fresh / "parts/part-000.npz").exists()
    # The valid local part survives; a retry can upload it without recomputation.
    assert (original / "parts/part-000.npz").read_bytes() == b"verified count arrays"
    remote.fail_receipt = False
    checkpoint.publish_part(original, 0, "b" * 64)
    store().restore(fresh, "b" * 64)
    assert (fresh / "parts/part-000.npz").is_file()


def test_remote_corruption_is_not_restored(tmp_path: Path, remote: LocalS3) -> None:
    original = tmp_path / "original"
    contract(original)
    checkpoint = store()
    checkpoint.restore(original, "c" * 64)
    part(original, "c" * 64)
    checkpoint.publish_part(original, 0, "c" * 64)
    remote_part = remote.directory / "bucket/comparisons" / ("c" * 64) / "parts/part-000.npz"
    remote_part.write_bytes(b"truncated")
    fresh = tmp_path / "fresh"
    contract(fresh)
    store().restore(fresh, "c" * 64)
    assert not (fresh / "parts/part-000.npz").exists()


def test_remote_contract_mismatch_fails_closed(tmp_path: Path, remote: LocalS3) -> None:
    original = tmp_path / "original"
    contract(original)
    checkpoint = store()
    checkpoint.restore(original, "d" * 64)
    remote_contract = (
        remote.directory / "bucket/comparisons" / ("d" * 64) / "comparison_contract.json"
    )
    remote_contract.write_text('{"version": 999}')
    with pytest.raises(ValueError, match="contract mismatch"):
        store().restore(original, "d" * 64)


def test_aws_access_failure_is_not_treated_as_empty_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 1, "", "AccessDenied")

    monkeypatch.setattr("otto_recsys.cloud.comparison_checkpoints.subprocess.run", denied)
    contract(tmp_path)
    with pytest.raises(RuntimeError, match="AccessDenied"):
        store().restore(tmp_path, "e" * 64)


@pytest.mark.parametrize("uri", ["https://bucket/path", "s3://bucket", "s3://bucket/a/../b"])
def test_invalid_checkpoint_location_is_rejected(uri: str) -> None:
    with pytest.raises(ValueError, match="checkpoint URI"):
        S3ComparisonCheckpoints(uri, region="us-west-2", logger=logging.getLogger("test"))
