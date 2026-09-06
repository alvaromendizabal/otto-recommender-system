from __future__ import annotations

import base64
import hashlib
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from botocore.exceptions import FlexibleChecksumError, IncompleteReadError, ReadTimeoutError
from botocore.httpchecksum import Sha256Checksum, StreamingChecksumBody
from botocore.response import StreamingBody
from urllib3.exceptions import ReadTimeoutError as HTTPReadTimeoutError
from urllib3.response import HTTPResponse

from otto_two_tower.benchmark_artifacts import BenchmarkArtifacts


class MemoryS3:
    class NoSuchKey(Exception):
        pass

    def __init__(self) -> None:
        self.exceptions = SimpleNamespace(NoSuchKey=self.NoSuchKey)
        self.objects = {}
        self.fail_receipt = False
        self.denied = False
        self.streams = []
        self.damage_key = None
        self.damage = None

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if self.denied:
            raise PermissionError("AccessDenied")
        if Key not in self.objects:
            raise self.NoSuchKey(Key)
        payload = self.objects[Key]
        raw = HTTPResponse(body=io.BytesIO(payload), preload_content=False)
        length = len(payload)
        if Key == self.damage_key and self.damage == "length":
            length += 1
        if Key == self.damage_key and self.damage == "timeout":

            class Interrupted(io.BytesIO):
                def read(self, amount: int | None = None) -> bytes:
                    if self.tell() or amount is None:
                        raise HTTPReadTimeoutError(None, None, "interrupted stream")
                    return super().read(amount)

            raw = HTTPResponse(body=Interrupted(payload), preload_content=False)
        if Key == self.damage_key and self.damage == "checksum":
            expected = base64.b64encode(hashlib.sha256(b"different content").digest()).decode()
            body = StreamingChecksumBody(raw, length, Sha256Checksum(), expected)
        else:
            body = StreamingBody(raw, length)
        self.streams.append(raw)
        return {"Body": body}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> None:
        if self.fail_receipt:
            raise OSError("interrupted receipt upload")
        self.objects[Key] = Body

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs: object) -> None:
        self.objects[key] = Path(filename).read_bytes()


def store(root: Path, remote: MemoryS3, identity: str = "run") -> BenchmarkArtifacts:
    return BenchmarkArtifacts(
        root,
        identity,
        logging.getLogger("artifacts-test"),
        uri="s3://test-bucket/benchmark/run",
        client=remote,
    )


def test_fresh_workspace_restores_committed_artifact_without_recomputation(tmp_path: Path) -> None:
    remote = MemoryS3()
    first = store(tmp_path / "first", remote)
    first.produce("index.faiss", lambda p: p.write_bytes(b"verified-index"))
    second = store(tmp_path / "second", remote)

    def forbidden(path: Path) -> None:
        pytest.fail("valid remote work was recomputed")

    path = second.produce("index.faiss", forbidden)
    before = path.stat().st_mtime_ns
    assert path.read_bytes() == b"verified-index"
    second.produce("index.faiss", forbidden)
    assert path.stat().st_mtime_ns == before
    assert all(stream.closed for stream in remote.streams)


def test_receipt_interruption_resends_local_work_without_recomputing(tmp_path: Path) -> None:
    remote = MemoryS3()
    artifacts = store(tmp_path, remote)
    remote.fail_receipt = True
    with pytest.raises(OSError):
        artifacts.produce("query.npz", lambda p: p.write_bytes(b"complete"))
    assert "benchmark/run/query.npz.json" not in remote.objects
    remote.fail_receipt = False
    path = artifacts.produce("query.npz", lambda _: pytest.fail("should resend valid local work"))
    assert path.read_bytes() == b"complete"
    assert "benchmark/run/query.npz.json" in remote.objects


def test_corruption_is_recomputed_and_wrong_identity_fails_closed(tmp_path: Path) -> None:
    remote = MemoryS3()
    store(tmp_path / "one", remote).produce("part.npz", lambda p: p.write_bytes(b"good"))
    remote.objects["benchmark/run/part.npz"] = b"corrupt"
    fresh = store(tmp_path / "two", remote)
    assert (
        fresh.produce("part.npz", lambda p: p.write_bytes(b"regenerated")).read_bytes()
        == b"regenerated"
    )
    with pytest.raises(ValueError, match="different benchmark"):
        store(tmp_path / "other", remote, "different").get("part.npz")
    remote.denied = True
    with pytest.raises(PermissionError):
        fresh.get("part.npz")


@pytest.mark.parametrize("name", ["../secret", "/absolute", "a/../../secret"])
def test_artifact_paths_cannot_escape_root(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        store(tmp_path, MemoryS3()).get(name)


@pytest.mark.parametrize("part", ["receipt", "blob"])
@pytest.mark.parametrize("damage", ["length", "checksum", "timeout"])
def test_sdk_validation_and_stream_cleanup_survive_failed_transfers(
    tmp_path: Path, part: str, damage: str
) -> None:
    remote = MemoryS3()
    payload = b"x" * (8 * 1024**2 + 31)
    store(tmp_path / "first", remote).produce("index.faiss", lambda p: p.write_bytes(payload))
    remote.damage_key = "benchmark/run/index.faiss" + (".json" if part == "receipt" else "")
    remote.damage = damage
    second = store(tmp_path / "second", remote)
    with pytest.raises((IncompleteReadError, FlexibleChecksumError, ReadTimeoutError)):
        second.get("index.faiss")
    assert not (second.root / "index.faiss").exists()
    assert not (second.root / "index.faiss.json").exists()
    assert not (second.root / "index.faiss.tmp").exists()
    assert all(stream.closed for stream in remote.streams)
    remote.damage = None
    path = second.produce("index.faiss", lambda _: pytest.fail("valid remote work was recomputed"))
    assert path.read_bytes() == payload
    assert all(stream.closed for stream in remote.streams)
