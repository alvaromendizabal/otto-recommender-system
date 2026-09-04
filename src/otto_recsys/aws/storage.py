from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

from otto_recsys.runtime import Heartbeat


def require_bucket() -> str:
    """Return the canonical OTTO bucket from the environment."""
    bucket = os.environ.get("OTTO_BUCKET")
    if not bucket:
        raise RuntimeError("OTTO_BUCKET environment variable is required")
    return bucket


def s3_uri(key: str, *, bucket: str | None = None) -> str:
    """Build a canonical OTTO S3 URI."""
    resolved_bucket = bucket or require_bucket()
    clean_key = key.strip("/")
    return f"s3://{resolved_bucket}/{clean_key}"


def run_streaming_command(
    command: list[str],
    *,
    logger: logging.Logger,
    stage: str,
) -> None:
    """Execute a long command with visible output and heartbeat telemetry."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        with Heartbeat(
            logger,
            stage=stage,
            interval_seconds=30.0,
            pid_provider=lambda: process.pid,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)

            return_code = process.wait()
    finally:
        if process.stdout is not None:
            process.stdout.close()

    if return_code != 0:
        raise RuntimeError(
            f"{stage} failed with exit code {return_code}: {' '.join(command)}"
        )


def upload_file(
    local_path: str | Path,
    key: str,
    *,
    logger: logging.Logger,
    bucket: str | None = None,
) -> str:
    """Upload one file using the configured AWS CLI."""
    source = Path(local_path).resolve()
    destination = s3_uri(key, bucket=bucket)

    run_streaming_command(
        ["aws", "s3", "cp", str(source), destination],
        logger=logger,
        stage="s3_upload",
    )
    return destination


def download_file(
    key: str,
    local_path: str | Path,
    *,
    logger: logging.Logger,
    bucket: str | None = None,
) -> Path:
    """Download one S3 object."""
    source = s3_uri(key, bucket=bucket)
    destination = Path(local_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    run_streaming_command(
        ["aws", "s3", "cp", source, str(destination)],
        logger=logger,
        stage="s3_download",
    )
    return destination
