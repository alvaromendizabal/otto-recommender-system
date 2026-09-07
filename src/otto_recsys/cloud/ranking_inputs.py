"""Restore audited ranking prerequisites before any candidate computation.

Published feature bytes are immutable inputs, not a request to rebuild features.
Only authenticated S3 reads are used here; downstream output publication is separate.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import polars as pl

from otto_recsys.cloud.ranking_checkpoints import S3FeatureCheckpoints
from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import utc_now_iso
from otto_recsys.ranking.feature_cache import FAMILIES, valid_part, workspace_lock
from otto_recsys.runtime import Heartbeat


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read ranking input evidence: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _copy_atomic(source: Path, target: Path) -> None:
    """Install verified bytes, preserving their published checksum."""
    if target.is_file() and sha256_file(target) == sha256_file(source):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(source.read_bytes())
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(target)


def _publication(evidence: Path) -> tuple[dict[str, Any], dict[str, Any], str]:
    contract_path = evidence / "ranking_feature_contract.json"
    contract = _json(contract_path)
    report = _json(evidence / "ranking_features.json")
    audit = _json(evidence / "ranking_features_audit.json")
    publication = _json(evidence / "ranking_features_publication.json")
    identity = canonical_json_sha256(contract)
    digest = sha256_file(contract_path)
    buckets = contract.get("buckets")
    mismatches = audit.get("mismatches", {})
    if (
        contract.get("schema_version") != 1
        or isinstance(buckets, bool) or not isinstance(buckets, int) or not 1 <= buckets <= 999
        or any(part.get("status") != "passed" or part.get("input_id") != identity
               for part in (report, audit, publication))
        or report.get("contract_sha256") != digest
        or audit.get("feature_contract_sha256") != digest
        or not re.fullmatch(r"[0-9a-f]{64}", str(report.get("parts_sha256", "")))
        or audit.get("parts_sha256") != report["parts_sha256"]
        or report.get("completed_buckets") != buckets or audit.get("verified_buckets") != buckets
        or set(mismatches) != {"sessions", "items", "queries", "duplicate_keys"}
        or any(value != 0 for value in mismatches.values())
    ):
        raise ValueError("Published feature contract, summary and independent audit disagree")
    uri = publication.get("uri", "").rstrip("/")
    if not uri.endswith("/" + identity):
        raise ValueError("Published feature URI must end with its exact content identity")
    return contract, report, uri


def _verify_receipts(receipts: list[dict[str, Any]], report: dict[str, Any]) -> None:
    for bucket, receipt in enumerate(receipts):
        files = receipt.get("files", {})
        elapsed = receipt.get("compute_seconds")
        if (receipt.get("input_id") != report["input_id"] or receipt.get("bucket") != bucket
                or set(files) != set(FAMILIES) or isinstance(elapsed, bool)
                or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                or elapsed < 0):
            raise ValueError(f"Invalid published feature receipt for bucket {bucket}")
        for part in files.values():
            if (not re.fullmatch(r"[0-9a-f]{64}", str(part.get("sha256", "")))
                    or any(isinstance(part.get(name), bool)
                           or not isinstance(part.get(name), int) or part[name] < 0
                           for name in ("rows", "bytes"))):
                raise ValueError(f"Invalid published feature file metadata in bucket {bucket}")
    if (len(receipts) != report["completed_buckets"]
            or canonical_json_sha256([r["files"] for r in receipts]) != report["parts_sha256"]
            or any(sum(r["files"][name]["rows"] for r in receipts) != report["rows"][name]
                   for name in FAMILIES)):
        raise ValueError("Feature receipts do not match the independently audited publication")


def _matches(path: Path, expected: dict[str, Any]) -> bool:
    try:
        return (
            path.stat().st_size == expected["bytes"]
            and sha256_file(path) == expected["sha256"]
            and pl.scan_parquet(path).select(pl.len()).collect().item() == expected["rows"]
        )
    except (OSError, ValueError, pl.exceptions.PolarsError):
        return False


def restore_observed_features(
    directory: Path, evidence: Path, *, region: str, logger: logging.Logger,
) -> dict[str, Any]:
    """Recover an absent/partial cache; retain every matching file and its mtime.

    The committed independent audit pins the entire receipt inventory. Download
    into staging, verify SHA-256, byte/row counts, then commit each bucket's
    receipt last. No feature builder is invoked and nothing is uploaded to S3.
    """
    started = time.perf_counter()
    contract, report, uri = _publication(evidence)
    identity = report["input_id"]
    transport = S3FeatureCheckpoints(uri, region=region, logger=logger)
    progress = {"bucket": 0, "buckets": contract["buckets"], "downloaded_files": 0}
    restored = downloaded_bytes = 0
    with workspace_lock(directory), Heartbeat(
        logger, stage="ranking_input_restore", interval_seconds=15,
        progress_provider=progress.copy,
    ):
        logger.info("ranking_input_restore_start", extra={"input_id": identity, "uri": uri})
        local_contract = directory / "feature_contract.json"
        if local_contract.exists():
            try:
                current = _json(local_contract)
            except ValueError:
                current = None  # Damaged metadata can be restored from pinned evidence.
            if current is not None and current != contract:
                raise ValueError("Local feature contract is incompatible; existing work preserved")
        local = [valid_part(directory, bucket, identity) for bucket in range(contract["buckets"])]
        complete = all(part is not None for part in local)
        if complete:
            try:
                _verify_receipts([part for part in local if part is not None], report)
            except ValueError:
                complete = False
        if not complete:
            with tempfile.TemporaryDirectory(dir=directory, prefix="staging-") as temporary:
                staging = Path(temporary)

                def download(relative: str) -> Path:
                    path = staging / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    transport.run(["cp", f"{uri}/{relative}", str(path), "--only-show-errors"])
                    return path

                remote_contract = download("feature_contract.json")
                if sha256_file(remote_contract) != report["contract_sha256"]:
                    raise ValueError(
                        "Remote feature contract checksum differs from published audit"
                    )
                receipts = [_json(download(f"parts/part-{bucket:03d}.json"))
                            for bucket in range(contract["buckets"])]
                _verify_receipts(receipts, report)
                _copy_atomic(remote_contract, local_contract)
                for bucket, receipt in enumerate(receipts):
                    progress["bucket"] = bucket
                    prior = local[bucket]
                    if prior is not None and prior["files"] == receipt["files"]:
                        logger.info("ranking_input_bucket_reused", extra={"bucket": bucket})
                        continue
                    for family in FAMILIES:
                        relative = f"parts/part-{bucket:03d}/{family}.parquet"
                        target = directory / relative
                        expected = receipt["files"][family]
                        if _matches(target, expected):
                            continue
                        incoming = download(relative)
                        if not _matches(incoming, expected):
                            raise ValueError(
                                f"Restored feature file failed checksum/row checks: {relative}"
                            )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        incoming.replace(target)
                        progress["downloaded_files"] += 1
                        downloaded_bytes += expected["bytes"]
                        logger.info("ranking_input_file_restored", extra={
                            "bucket": bucket, "file": relative,
                            "elapsed_seconds": round(time.perf_counter() - started, 3),
                        })
                    relative_receipt = f"parts/part-{bucket:03d}.json"
                    _copy_atomic(staging / relative_receipt, directory / relative_receipt)
                    if valid_part(directory, bucket, identity) is None:
                        raise ValueError(
                            f"Restored feature bucket {bucket} failed final verification"
                        )
                    restored += 1
                    logger.info("ranking_input_bucket_restored", extra={"bucket": bucket})
        # Missing metadata alone never requires rebuilding or downloading valid feature data.
        _copy_atomic(evidence / "ranking_feature_contract.json", local_contract)
        _copy_atomic(evidence / "ranking_features.json", directory / "manifest.json")
        result = {
            "status": "passed", "input_id": identity, "source_uri": uri,
            "completed_at_utc": utc_now_iso(), "restored_buckets": restored,
            "reused_buckets": contract["buckets"] - restored,
            "verified_buckets": contract["buckets"], "verified_data_files": 3 * contract["buckets"],
            "downloaded_data_files": progress["downloaded_files"],
            "downloaded_data_bytes": downloaded_bytes, "rows": report["rows"],
            "parts_sha256": report["parts_sha256"],
            "source_contract_sha256": report["contract_sha256"],
            "attempt_elapsed_seconds": time.perf_counter() - started,
            "feature_computation_performed": False,
        }
        logger.info("ranking_input_restore_complete", extra={
            "restored_parts": restored, "reused_parts": result["reused_buckets"],
            "elapsed_seconds": round(result["attempt_elapsed_seconds"], 3),
        })
        return result


def prepare_ranking_inputs(
    cache: Path, observed: Path, covisit: Path, vectors: Path, index: Path,
    *, evidence: Path, region: str, logger: logging.Logger,
) -> dict[str, Any]:
    """Report all absent upstream files together, then restore the saved features."""
    started = time.perf_counter()
    logger.info("ranking_input_preflight_start")
    required = [cache / name for name in
                ("manifest.json", "items.parquet", "examples.parquet", "labels.parquet")]
    required += [covisit / f"{family}.{suffix}" for family in ("time", "type", "buy")
                 for suffix in ("parquet", "json")]
    required += [vectors, vectors.parent / "manifest.json", index, index.parent / "manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(
            "Frozen ranking inputs are missing: " + "; ".join(missing)
            + ". Restore these existing artifacts or supply their explicit CLI paths. "
            "No retrieval training or feature recomputation was started."
        )
    contract, _, _ = _publication(evidence)
    ranking = _json(cache / "manifest.json")
    if (ranking.get("input_id") != contract["training_cache_input_id"]
            or ranking.get("validation_manifest_id") != contract["validation_manifest_id"]):
        raise ValueError("Frozen ranking cache does not match the published feature identity")
    with Heartbeat(logger, stage="ranking_input_checksums", interval_seconds=15):
        for name in ("examples", "labels"):
            digest = sha256_file(cache / f"{name}.parquet")
            if (digest != contract["input_sha256"][name]
                    or digest != ranking.get(f"{name}_sha256")):
                raise ValueError(f"Frozen {name} checksum does not match published features")
        if sha256_file(cache / "items.parquet") != ranking.get("items_sha256"):
            raise ValueError("Frozen ranking items checksum mismatch")
    result = restore_observed_features(observed, evidence, region=region, logger=logger)
    result["preflight_elapsed_seconds"] = time.perf_counter() - started
    logger.info("ranking_input_preflight_complete", extra={
        "status": "passed", "elapsed_seconds": round(result["preflight_elapsed_seconds"], 3)
    })
    return result
