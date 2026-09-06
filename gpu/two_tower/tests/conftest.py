from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def toolkit_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Callable[[dict[str, str]], list[str]]:
    # Import lazily: dependency-free launch tests also load this conftest.
    from sagemaker_training import environment, mapping, params

    path = tmp_path / "hyperparameters.json"
    monkeypatch.setattr(environment, "hyperparameters_file_dir", str(path))

    def load(parameters: dict[str, str]) -> list[str]:
        path.write_text(json.dumps(parameters))
        decoded = environment.read_hyperparameters()
        user = mapping.split_by_criteria(
            decoded, keys=params.SAGEMAKER_HYPERPARAMETERS, prefix=params.SAGEMAKER_PREFIX
        ).excluded
        return list(mapping.to_cmd_args(user))

    return load


@pytest.fixture
def s3_http() -> Any:
    """A local S3 protocol fixture with the real boto3 HTTP/streaming stack."""
    # Keep SDK imports lazy so the parser-only preflight remains dependency-free.
    import base64
    import hashlib
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from types import SimpleNamespace
    from urllib.parse import unquote, urlsplit

    import boto3
    from botocore.config import Config

    objects: dict[str, bytes] = {}
    reads, writes = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def key(self) -> str:
            return unquote(urlsplit(self.path).path).removeprefix("/test-bucket/")

        def do_GET(self) -> None:
            key = self.key()
            reads.append(key)
            if key not in objects:
                payload = b"<Error><Code>NoSuchKey</Code><Message>Missing key</Message></Error>"
                self.send_response(404)
                self.send_header("Content-Type", "application/xml")
            else:
                payload = objects[key]
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                checksum = base64.b64encode(hashlib.sha256(payload).digest()).decode()
                self.send_header("x-amz-checksum-sha256", checksum)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_PUT(self) -> None:
            # Plain HTTP uses fixed-length uploads for these small fixture files.
            length = int(self.headers["Content-Length"])
            payload = self.rfile.read(length)
            assert len(payload) == length
            key = self.key()
            objects[key] = payload
            writes.append(key)
            self.send_response(200)
            self.send_header("ETag", '"' + hashlib.sha256(payload).hexdigest() + '"')
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = boto3.client(
        "s3",
        endpoint_url=f"http://127.0.0.1:{server.server_port}",
        region_name="us-west-2",
        aws_access_key_id="local-test",
        aws_secret_access_key="local-test",
        config=Config(
            connect_timeout=2,
            read_timeout=5,
            retries={"mode": "standard", "total_max_attempts": 1},
            s3={"addressing_style": "path"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_supported",
        ),
    )
    try:
        yield SimpleNamespace(client=client, objects=objects, reads=reads, writes=writes)
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
