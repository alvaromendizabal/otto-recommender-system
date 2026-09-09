"""Prove competition query provenance before any recommendation is generated."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from otto_recsys.experiments.manifest import sha256_file


def check_competition_input(directory: Path, expected: dict[str, Any]) -> dict[str, Any]:
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError("competition input requires a verified conversion manifest")
    converted = json.loads(path.read_text())
    if converted != expected["conversion"]:
        raise ValueError(
            "competition input differs from the official truncated test file; "
            "the post-competition full test release contains future events"
        )
    parts = sorted(directory.glob("part-*.parquet"))
    observed = {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in parts}
    if not parts or observed != expected["files"]:
        raise ValueError("competition test partition bytes differ from their source attestation")
    return {"source": expected["source"], "raw_sha256": expected["raw_sha256"],
            "sessions": converted["sessions_processed"], "events": converted["events_processed"]}


def require_competition_input(directory: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    expected = json.loads((root / "reports/submissions/competition_input.json").read_text())
    return check_competition_input(directory, expected)
