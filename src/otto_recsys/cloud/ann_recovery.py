"""Provenance-checked reuse of reference metrics across ANN code revisions."""

from __future__ import annotations

import ast
import json
import math
import re
import tarfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from otto_recsys.cloud.sagemaker_pipeline import canonical_sha256, source_s3_uri
from otto_recsys.cloud.source_preflight import utc_now
from otto_recsys.experiments.manifest import sha256_file


@dataclass(frozen=True)
class ReferencePart:
    name: str
    path: Path
    receipt: dict[str, Any]


def _reference_function(source: str) -> str:
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_reference_counts"
    ]
    if len(functions) != 1:
        raise ValueError("reference metric derivation function is missing or ambiguous")
    return ast.dump(functions[0], include_attributes=False)


def prepare_reference_reuse(
    *,
    bucket: str,
    fold: int,
    previous_run_id: str,
    previous_contract: dict[str, Any],
    contract: dict[str, Any],
    reference: dict[str, Any],
    source_root: Path,
    workspace: Path,
    keys: set[str],
    download: Callable[[str, Path], None],
) -> list[ReferencePart]:
    """Validate everything locally; this function never writes to remote state.

    Only reference count parts are eligible. Equality of the complete derivation
    function and metrics module is deliberately stricter than a version label.
    ANN indexes, query embeddings, timings and selections cannot cross revisions.
    """
    started = time.perf_counter()
    if not re.fullmatch(r"[a-f0-9]{64}", previous_run_id):
        raise ValueError("invalid previous ANN run ID")
    if canonical_sha256(previous_contract) != previous_run_id:
        raise ValueError("previous ANN run contract identity mismatch")
    for field in (
        "training_run_id",
        "reference_run_id",
        "reference_input_id",
        "reference_manifest_sha256",
    ):
        if previous_contract[field] != contract[field]:
            raise ValueError(f"reference recovery provenance mismatch: {field}")
    if (
        reference["validation_fold"] != fold
        or reference["input_id"] != contract["reference_input_id"]
    ):
        raise ValueError("reference recovery fold or input mismatch")
    workspace.mkdir(parents=True, exist_ok=True)
    archive = workspace / "source.tar.gz"
    download(
        source_s3_uri(bucket, previous_contract["code_commit"], previous_contract["source_sha256"]),
        archive,
    )
    if sha256_file(archive) != previous_contract["source_sha256"]:
        raise ValueError("previous source archive checksum mismatch")
    with tarfile.open(archive, "r:gz") as source:
        for name in ("ranking_metrics.py", "ann_benchmark.py"):
            relative = "otto_two_tower/" + name
            handle = source.extractfile(relative)
            if handle is None:
                raise ValueError("previous metric source missing")
            with handle:
                old = handle.read().decode()
            new = (source_root / relative).read_text()
            if name == "ann_benchmark.py":
                old, new = _reference_function(old), _reference_function(new)
            if old != new:
                raise ValueError("reference metric implementation changed; recomputation required")
    prefix = f"retrieval/two-tower/ann/fold-{fold}/{previous_run_id}/checkpoints/"
    buckets = reference["ranking_manifest"]["config"]["buckets"]
    parts = []
    for objective in ("clicks", "carts", "orders"):
        for part_id in range(buckets):
            name = f"reference/{objective}/part-{part_id:03d}.npz"
            key = prefix + name
            if key + ".json" not in keys:
                continue  # Uncommitted blobs are never reused.
            path = workspace / name
            path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path = path.with_suffix(".npz.json")
            download(f"s3://{bucket}/{key}.json", receipt_path)
            receipt = json.loads(receipt_path.read_text())
            if receipt["input_id"] != previous_run_id:
                raise ValueError("reference receipt belongs to a different run")
            download(f"s3://{bucket}/{key}", path)
            elapsed = float(receipt["elapsed_seconds"])
            if (
                sha256_file(path) != receipt["sha256"]
                or path.stat().st_size != receipt["bytes"]
                or not math.isfinite(elapsed)
                or elapsed < 0
            ):
                raise ValueError("reference receipt checksum, size or timing mismatch")
            with np.load(path, allow_pickle=False) as payload:
                sessions, counts = payload["sessions"], payload["counts"]
                if (
                    sessions.ndim != 1
                    or not np.issubdtype(sessions.dtype, np.integer)
                    or np.any(sessions < 0)
                    or np.any(np.diff(sessions) <= 0)
                    or np.any(sessions % buckets != part_id)
                    or counts.shape != (len(sessions), 7)
                    or not np.isfinite(counts).all()
                    or np.any(counts < 0)
                    or np.any(counts[:, 0] > 20)
                    or np.any(counts[:, 1] > counts[:, 0])
                    or np.any(counts[:, 2:] > 1 + 1e-12)
                ):
                    raise ValueError("reference metric checkpoint schema or coverage mismatch")
            parts.append(ReferencePart(name, path, receipt))
            print(
                f"[{utc_now()}] reference_reuse_verified file={name} parts={len(parts)} "
                f"elapsed_seconds={time.perf_counter() - started:.3f}",
                flush=True,
            )
    if not parts:
        raise ValueError("previous run has no committed reference count parts")
    return parts


def publish_reference_reuse(
    *,
    parts: list[ReferencePart],
    run_id: str,
    checkpoint_uri: str,
    existing_keys: set[str],
    download: Callable[[str, Path], None],
    upload: Callable[[Path, str], None],
) -> dict[str, Any]:
    """Commit data before receipts; retry safely after any interrupted transfer."""
    started = time.perf_counter()
    published, retained = [], []
    for part in parts:
        uri = checkpoint_uri.rstrip("/") + "/" + part.name
        key = uri.split("/", 3)[3]
        if key + ".json" in existing_keys:
            # Preserve existing work; reject conflicts instead of overwriting it.
            target_receipt = part.path.with_suffix(".target.json")
            target_blob = part.path.with_suffix(".target.npz")
            download(uri + ".json", target_receipt)
            receipt = json.loads(target_receipt.read_text())
            download(uri, target_blob)
            if receipt["input_id"] != run_id or sha256_file(target_blob) != receipt["sha256"]:
                raise ValueError("existing destination reference artifact is invalid")
            with (
                np.load(target_blob, allow_pickle=False) as actual,
                np.load(part.path, allow_pickle=False) as expected,
            ):
                if any(not np.array_equal(actual[k], expected[k]) for k in ("sessions", "counts")):
                    raise ValueError("existing destination reference metrics disagree")
            retained.append(part.name)
            continue
        receipt = {
            **part.receipt,
            "input_id": run_id,
            "reused_from_run_id": part.receipt["input_id"],
            "reused_from_sha256": part.receipt["sha256"],
            "reused_at_utc": utc_now(),
        }
        local_receipt = part.path.with_suffix(".destination.json")
        local_receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n")
        upload(part.path, uri)
        upload(local_receipt, uri + ".json")
        published.append(part.name)
        print(
            f"[{utc_now()}] reference_reuse_durable file={part.name} "
            f"elapsed_seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
    return {
        "status": "passed",
        "run_id": run_id,
        "published": published,
        "retained": retained,
        "source_run_ids": sorted({part.receipt["input_id"] for part in parts}),
        "reused_compute_seconds": sum(float(part.receipt["elapsed_seconds"]) for part in parts),
        "completed_at_utc": utc_now(),
        "elapsed_seconds": time.perf_counter() - started,
    }
