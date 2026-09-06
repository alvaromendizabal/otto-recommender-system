from __future__ import annotations

import logging
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from botocore.httpchecksum import StreamingChecksumBody
from urllib3.response import HTTPResponse

from otto_two_tower.benchmark_artifacts import BenchmarkArtifacts


def test_real_boto3_transport_restores_multiple_chunks_and_keeps_valid_work(
    tmp_path: Path, s3_http: Any
) -> None:
    logger = logging.getLogger("transport")
    options = {"uri": "s3://test-bucket/benchmark/run", "client": s3_http.client}
    first = BenchmarkArtifacts(tmp_path / "first", "run", logger, **options)
    payload = b"verified data\n" * 700_000  # More than one 8 MiB read chunk.
    first.produce("index.faiss", lambda path: path.write_bytes(payload))
    assert s3_http.writes[:2] == ["benchmark/run/index.faiss", "benchmark/run/index.faiss.json"]
    response = s3_http.client.get_object(Bucket="test-bucket", Key="benchmark/run/index.faiss")
    with closing(response["Body"]) as body:
        assert isinstance(body, StreamingChecksumBody)
        assert isinstance(body._raw_stream, HTTPResponse)
        assert body.read() == payload
    second = BenchmarkArtifacts(tmp_path / "fresh", "run", logger, **options)
    path = second.produce("index.faiss", lambda _: pytest.fail("valid remote work was recomputed"))
    assert path.read_bytes() == payload
    assert s3_http.writes == ["benchmark/run/index.faiss", "benchmark/run/index.faiss.json"]
    assert second.used["index.faiss"]["sha256"] == first.used["index.faiss"]["sha256"]
    assert second.get("missing.npz") is None


def test_real_boto3_transport_rejects_wrong_receipt_identity(tmp_path: Path, s3_http: Any) -> None:
    options = {"uri": "s3://test-bucket/benchmark/run", "client": s3_http.client}
    logger = logging.getLogger("transport")
    first = BenchmarkArtifacts(tmp_path / "first", "run", logger, **options)
    first.produce("part.npz", lambda path: path.write_bytes(b"part"))
    other = BenchmarkArtifacts(tmp_path / "other", "different", logger, **options)
    with pytest.raises(ValueError, match="different benchmark"):
        other.get("part.npz")
    assert not (other.root / "part.npz").exists()
