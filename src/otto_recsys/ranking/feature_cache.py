"""Content-addressed, bucket-resumable preparation of observed ranking features."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import utc_now_iso
from otto_recsys.ranking.features import FEATURE_SPEC, OBJECTIVES, observed_features, query_ledger
from otto_recsys.ranking.splits import inner_partition, split_role
from otto_recsys.ranking.training_cache import fold_for_session
from otto_recsys.runtime import Heartbeat

FAMILIES = ("sessions", "items", "queries")


class FeatureCheckpoints(Protocol):
    def restore(self, directory: Path, input_id: str) -> None: ...
    def publish_part(self, directory: Path, bucket: int, input_id: str) -> None: ...
    def publish_summary(self, directory: Path) -> None: ...


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def workspace_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("ranking feature cache already has an active writer") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def valid_part(directory: Path, bucket: int, input_id: str) -> dict[str, Any] | None:
    """Only a matching receipt plus every verified file can commit a bucket."""
    root = directory / "parts" / f"part-{bucket:03d}"
    try:
        receipt = json.loads(root.with_suffix(".json").read_text())
        if (
            not isinstance(receipt, dict)
            or receipt.get("input_id") != input_id
            or receipt.get("bucket") != bucket
            or set(receipt["files"]) != set(FAMILIES)
        ):
            return None
        for family in FAMILIES:
            path = root / f"{family}.parquet"
            evidence = receipt["files"][family]
            if evidence["sha256"] != sha256_file(path) or evidence["bytes"] != path.stat().st_size:
                return None
            if evidence["rows"] != pl.scan_parquet(path).select(pl.len()).collect().item():
                return None
        return receipt
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError):
        return None


def make_contract(cache: Path, *, inner_seed: int) -> dict[str, Any]:
    manifest = json.loads((cache / "manifest.json").read_text())
    if not 1 <= manifest["config"]["buckets"] <= 999:
        raise ValueError("feature buckets must be between 1 and 999")
    checksums = {}
    for name in ("events", "examples", "labels"):
        checksums[name] = sha256_file(cache / f"{name}.parquet")
        if checksums[name] != manifest[f"{name}_sha256"]:
            raise ValueError(f"frozen {name} checksum mismatch")
    code = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("features.py", "feature_cache.py", "splits.py", "training_cache.py")
    }
    code["ranking_checkpoints.py"] = sha256_file(
        Path(__file__).parents[1] / "cloud/ranking_checkpoints.py"
    )
    return {
        "schema_version": 1,
        "validation_manifest_id": manifest["validation_manifest_id"],
        "training_cache_input_id": manifest["input_id"],
        "input_sha256": checksums,
        "fold_seed": manifest["config"]["fold_seed"],
        "folds": manifest["config"]["folds"],
        "buckets": manifest["config"]["buckets"],
        "sessions": manifest["sessions"],
        "inner_seed": inner_seed,
        "inner_partitions": 5,
        "inner_selection_partition": 0,
        "feature_spec": FEATURE_SPEC,
        "code_sha256": code,
        "runtime": {"python": platform.python_version(), "polars": version("polars")},
        "validation_scope": "exploratory nested session splits within the existing time window",
        "outer_role": "fold == outer_fold",
        "inner_role": "fold != outer_fold and inner_partition == 0",
        "fit_role": "fold != outer_fold and inner_partition != 0",
        "label_window_timestamps": "not retained in the frozen labels cache",
        "untouched_temporal_holdout": False,
        "retriever_fit_provenance_certified": False,
        "neural_candidates_materialized": False,
    }


def validate_examples(examples: pl.DataFrame, contract: dict[str, Any]) -> pl.DataFrame:
    if examples.height != contract["sessions"] or examples["session"].n_unique() != examples.height:
        raise ValueError("session coverage or uniqueness mismatch")
    if examples["session"].null_count() or examples.filter(pl.col("session") < 0).height:
        raise ValueError("invalid session ID")
    expected_folds = [
        fold_for_session(int(s), seed=contract["fold_seed"], folds=contract["folds"])
        for s in examples["session"]
    ]
    if expected_folds != examples["fold"].to_list():
        raise ValueError("frozen outer fold assignment mismatch")
    if examples.filter(pl.col("bucket") != pl.col("session") % contract["buckets"]).height:
        raise ValueError("frozen bucket assignment mismatch")
    inner = [inner_partition(int(s), seed=contract["inner_seed"]) for s in examples["session"]]
    return examples.with_columns(pl.Series("inner_partition", inner, dtype=pl.UInt8))


def build_feature_cache(
    cache: Path,
    output: Path,
    *,
    logger: logging.Logger,
    inner_seed: int = 20260907,
    heartbeat_seconds: float = 15,
    checkpoints: FeatureCheckpoints | None = None,
) -> dict[str, Any]:
    """Resume valid parts; recompute only incomplete or corrupt buckets."""
    started = time.perf_counter()
    progress = {"bucket": 0, "buckets": 0, "sessions": 0}
    with (
        workspace_lock(output),
        Heartbeat(
            logger,
            stage="ranking_features",
            interval_seconds=heartbeat_seconds,
            progress_provider=progress.copy,
        ),
    ):
        logger.info("ranking_features_start", extra={"stage": "input_verification"})
        contract = make_contract(cache, inner_seed=inner_seed)
        input_id = canonical_json_sha256(contract)
        contract_path = output / "feature_contract.json"
        if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
            raise ValueError("feature contract mismatch; select a new output directory")
        write_json(contract_path, contract)
        if checkpoints is not None:
            checkpoints.restore(output, input_id)
        examples = validate_examples(pl.read_parquet(cache / "examples.parquet"), contract)
        expected_rows = {
            "events": examples["observed_events"].sum(),
            "labels": sum(
                examples[name].sum() for name in ("click_labels", "cart_labels", "order_labels")
            ),
        }
        for family, expected_rows_count in expected_rows.items():
            actual_rows = (
                pl.scan_parquet(cache / f"{family}.parquet").select(pl.len()).collect().item()
            )
            if actual_rows != expected_rows_count:
                raise ValueError(f"frozen {family} row coverage mismatch")
        progress["buckets"] = int(contract["buckets"])
        receipts = []
        reused = 0
        for bucket in range(contract["buckets"]):
            progress["bucket"] = bucket
            receipt = valid_part(output, bucket, input_id)
            if receipt is not None:
                reused += 1
                logger.info("ranking_feature_bucket_reused", extra={"bucket": bucket})
            else:
                part_started = time.perf_counter()
                subset = examples.filter(pl.col("bucket") == bucket)
                events = (
                    pl.scan_parquet(cache / "events.parquet")
                    .filter(
                        pl.col("bucket") == bucket,
                    )
                    .collect()
                )
                labels = (
                    pl.scan_parquet(cache / "labels.parquet")
                    .filter(
                        pl.col("bucket") == bucket,
                    )
                    .collect()
                )
                sessions, items = observed_features(events, subset)
                queries = query_ledger(subset, labels)
                # Label counts are verified independently of candidate membership.
                for objective, column in zip(
                    OBJECTIVES, ("click_labels", "cart_labels", "order_labels"), strict=True
                ):
                    actual = queries.filter(pl.col("objective") == objective).sort("session")[
                        "true_items"
                    ]
                    if actual.to_list() != subset.sort("session")[column].to_list():
                        raise ValueError("frozen label count mismatch")
                root = output / "parts" / f"part-{bucket:03d}"
                root.mkdir(parents=True, exist_ok=True)
                files = {}
                for family, frame in zip(FAMILIES, (sessions, items, queries), strict=True):
                    path = root / f"{family}.parquet"
                    temporary = path.with_suffix(".parquet.tmp")
                    frame.write_parquet(temporary, compression="zstd")
                    temporary.replace(path)
                    files[family] = {
                        "sha256": sha256_file(path),
                        "rows": frame.height,
                        "bytes": path.stat().st_size,
                    }
                receipt = {
                    "input_id": input_id,
                    "bucket": bucket,
                    "files": files,
                    "completed_at_utc": utc_now_iso(),
                    "compute_seconds": time.perf_counter() - part_started,
                }
                write_json(root.with_suffix(".json"), receipt)
                logger.info(
                    "ranking_feature_bucket_complete",
                    extra={
                        "bucket": bucket,
                        "sessions": sessions.height,
                        "elapsed_seconds": round(receipt["compute_seconds"], 3),
                    },
                )
            # Publish even a reused local receipt: an earlier upload may have failed.
            if checkpoints is not None:
                checkpoints.publish_part(output, bucket, input_id)
            receipts.append(receipt)
            progress["sessions"] += receipt["files"]["sessions"]["rows"]
        roles = []
        for outer in range(contract["folds"]):
            counts = {"fit": 0, "inner": 0, "outer": 0}
            for fold, inner in examples.select("fold", "inner_partition").iter_rows():
                counts[split_role(fold, inner, outer_fold=outer, folds=contract["folds"])] += 1
            roles.append({"outer_fold": outer, **counts})
        ledger = (
            pl.scan_parquet(output / "parts/part-*/queries.parquet")
            .group_by("objective")
            .agg(
                pl.col("true_items").sum(),
                pl.col("recall_denominator").sum(),
                (pl.col("true_items") > 0).sum().alias("labeled_queries"),
            )
            .collect()
            .sort("objective")
        )
        summary = {
            "status": "passed",
            "input_id": input_id,
            "completed_at_utc": utc_now_iso(),
            "contract_sha256": sha256_file(contract_path),
            "completed_buckets": len(receipts),
            "reused_buckets": reused,
            "attempt_elapsed_seconds": time.perf_counter() - started,
            "retained_compute_seconds": sum(r["compute_seconds"] for r in receipts),
            "rows": {
                family: sum(r["files"][family]["rows"] for r in receipts) for family in FAMILIES
            },
            "bytes": sum(f["bytes"] for r in receipts for f in r["files"].values()),
            "split_counts": roles,
            "official_denominators": ledger.to_dicts(),
            "all_roles_nonempty": all(
                all(row[key] > 0 for key in ("fit", "inner", "outer")) for row in roles
            ),
            "parts_sha256": canonical_json_sha256([r["files"] for r in receipts]),
            "untouched_temporal_holdout": False,
            "ranking_model_trained": False,
        }
        write_json(output / "manifest.json", summary)
        logger.info(
            "ranking_features_complete",
            extra={
                "status": "passed",
                "sessions": progress["sessions"],
                "elapsed_seconds": round(summary["attempt_elapsed_seconds"], 3),
            },
        )
        if checkpoints is not None:
            checkpoints.publish_summary(output)
        return summary
