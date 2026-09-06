from __future__ import annotations

import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from otto_two_tower.benchmark_artifacts import BenchmarkArtifacts


class Body(io.BytesIO):
    def iter_chunks(self, chunk_size: int):
        while chunk := self.read(chunk_size):
            yield chunk


class MemoryS3:
    class NoSuchKey(Exception):
        pass

    def __init__(self) -> None:
        self.exceptions = SimpleNamespace(NoSuchKey=self.NoSuchKey)
        self.objects = {}
        self.fail_receipt = False
        self.denied = False

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if self.denied:
            raise PermissionError("AccessDenied")
        if Key not in self.objects:
            raise self.NoSuchKey(Key)
        return {"Body": Body(self.objects[Key])}

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
