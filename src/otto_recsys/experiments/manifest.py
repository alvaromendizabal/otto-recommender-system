from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    """Return an offset-aware UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def canonical_json_sha256(value: Any) -> str:
    """Hash JSON-compatible configuration deterministically."""
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return SHA-256 for a file using bounded memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


@dataclass(frozen=True)
class RunManifest:
    """Immutable description of a reproducible project run."""

    run_id: str
    stage: str
    status: str
    started_at_utc: str
    finished_at_utc: str | None
    elapsed_seconds: float | None

    git_commit: str
    git_dirty: bool
    python_version: str
    platform: str
    uv_lock_sha256: str

    config_sha256: str
    seed: int

    dataset_manifest_id: str | None
    validation_manifest_id: str | None

    metrics: dict[str, float]
    artifacts: dict[str, str]

    @classmethod
    def start(
        cls,
        stage: str,
        *,
        config: Any,
        seed: int,
        project_root: str | Path = ".",
        dataset_manifest_id: str | None = None,
        validation_manifest_id: str | None = None,
    ) -> RunManifest:
        root = Path(project_root).resolve()
        lock_path = root / "uv.lock"

        return cls(
            run_id=str(uuid.uuid4()),
            stage=stage,
            status="running",
            started_at_utc=utc_now_iso(),
            finished_at_utc=None,
            elapsed_seconds=None,
            git_commit=_git_output(root, "rev-parse", "HEAD"),
            git_dirty=bool(_git_output(root, "status", "--porcelain")),
            python_version=platform.python_version(),
            platform=platform.platform(),
            uv_lock_sha256=sha256_file(lock_path),
            config_sha256=canonical_json_sha256(config),
            seed=seed,
            dataset_manifest_id=dataset_manifest_id,
            validation_manifest_id=validation_manifest_id,
            metrics={},
            artifacts={},
        )

    def finish(
        self,
        *,
        status: str,
        elapsed_seconds: float,
        metrics: dict[str, float] | None = None,
        artifacts: dict[str, str] | None = None,
    ) -> RunManifest:
        if status not in {"completed", "failed"}:
            raise ValueError("status must be 'completed' or 'failed'")

        return replace(
            self,
            status=status,
            finished_at_utc=utc_now_iso(),
            elapsed_seconds=float(elapsed_seconds),
            metrics=metrics or {},
            artifacts=artifacts or {},
        )

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        temp_path = destination.with_suffix(destination.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(destination)
