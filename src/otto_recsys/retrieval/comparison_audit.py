"""Independently audit saved comparison counts without rebuilding retrieval."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.logging_utils import utc_now_iso

OBJECTIVES = ("clicks", "carts", "orders")
WEIGHTS = np.array([0.1, 0.3, 0.6])


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected object: {path.name}")
    return payload


def _match(actual: Any, expected: Any, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=False):
        raise ValueError(f"metric mismatch: {name}")


def _validate_counts(counts: np.ndarray, depths: list[int]) -> None:
    if counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("counts must be nonnegative integers")
    denom, base = counts[:, :, :1], counts[:, :, 1:2]
    neural, union, exclusive = counts[:, :, 2::3], counts[:, :, 3::3], counts[:, :, 4::3]
    if (
        np.any(denom > 20)
        or np.any(base > denom)
        or np.any(neural > denom)
        or np.any(union > denom)
        or np.any(union < base)
        or np.any(union < neural)
        or np.any(union > base + neural)
        or np.any(exclusive > np.asarray(depths))
        or np.any((denom == 0) & (exclusive != 0))
        or np.any(np.minimum(exclusive, 20) > neural)
        or np.any(union - base > exclusive)
        or np.any((denom < 20) & (exclusive > neural))
    ):
        raise ValueError("invalid top-20 count invariants")
    for values in (neural, union, exclusive):
        if np.any(values[:, :, 1:] < values[:, :, :-1]):
            raise ValueError("counts decrease with candidate depth")


def audit_comparison(
    directory: Path, *, expected_input_id: str, buckets: int, logger: logging.Logger
) -> dict[str, Any]:
    """Verify identities, all parts, count invariants, estimates and paired CIs.

    The audit uses multinomial sample multiplicities rather than the evaluator's
    row-gather aggregation. It never imports or calls the evaluator's summary.
    It verifies saved counts, not the original predictions or ground truth.
    """
    started = time.perf_counter()
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    report = _read(directory / "metrics.json")
    contract = _read(directory / "comparison_contract.json")
    if (
        report.get("status") != "passed"
        or report.get("schema_version") != 1
        or contract.get("schema_version") != 1
        or report.get("input_id") != expected_input_id
        or canonical_json_sha256(contract) != expected_input_id
        or report.get("contract") != contract
        or report.get("prediction_input_id") != contract["prediction_input_id"]
    ):
        raise ValueError("comparison identity mismatch")
    depths = contract["depths"]
    if not depths or depths != sorted(set(depths)) or depths[0] < 1:
        raise ValueError("invalid depths")
    expected_files = {
        f"part-{bucket:03d}.{suffix}" for bucket in range(buckets) for suffix in ("json", "npz")
    }
    actual_files = {path.name for path in (directory / "parts").iterdir() if path.is_file()}
    if actual_files != expected_files:
        raise ValueError("incomplete or unexpected comparison parts")
    session_parts, count_parts, receipts = [], [], []
    for bucket in range(buckets):
        path = directory / "parts" / f"part-{bucket:03d}.npz"
        receipt = _read(path.with_suffix(".json"))
        digest = sha256_file(path)
        if receipt["input_id"] != expected_input_id or receipt["sha256"] != digest:
            raise ValueError(f"invalid receipt: bucket {bucket}")
        elapsed = float(receipt["elapsed_seconds"])
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("invalid bucket timing")
        with np.load(path, allow_pickle=False) as part:
            sessions, counts = part["sessions"], part["counts"]
        if (
            sessions.ndim != 1
            or sessions.dtype.kind not in "iu"
            or np.any(sessions < 0)
            or np.any(sessions % buckets != bucket)
            or np.any(sessions[1:] <= sessions[:-1])
            or counts.shape != (len(sessions), 3, 2 + 3 * len(depths))
        ):
            raise ValueError(f"invalid sessions or count shape: bucket {bucket}")
        _validate_counts(counts, depths)
        session_parts.append(sessions)
        count_parts.append(counts)
        receipts.append({"bucket": bucket, "sha256": digest, "elapsed_seconds": elapsed})
        logger.info("audit_part_verified", extra={"bucket": bucket, "buckets": buckets})
    sessions = np.concatenate(session_parts)
    if len(sessions) != report["sessions"] or len(np.unique(sessions)) != len(sessions):
        raise ValueError("session coverage mismatch")
    counts = np.concatenate(count_parts)[np.argsort(sessions)].astype(np.int64)
    totals = counts.sum(axis=0)
    denominator = totals[:, 0]
    if np.any(denominator <= 0):
        raise ValueError("each objective requires labels")
    bootstrap = report["bootstrap"]
    iterations = contract["bootstrap_iterations"]
    if iterations < 2 or bootstrap != {
        "iterations": iterations,
        "seed": contract["bootstrap_seed"],
        "method": "paired percentile",
        "unit": "session",
        "confidence": 0.95,
    }:
        raise ValueError("bootstrap contract mismatch")
    # Only denominator and incremental hits are needed for this paired bootstrap.
    values = np.concatenate(
        (counts[:, :, :1], counts[:, :, 3::3] - counts[:, :, 1:2]), axis=2
    ).astype(np.float64)
    flat = values.reshape(len(sessions), -1)
    rng = np.random.default_rng(bootstrap["seed"])
    draws = np.empty((iterations, 3, len(depths)))
    for iteration in range(iterations):
        chosen = rng.choice(len(sessions), size=len(sessions), replace=True)
        multiplicity = np.bincount(chosen, minlength=len(sessions))
        summed = (multiplicity @ flat).reshape(3, -1)
        if np.any(summed[:, 0] == 0):
            raise ValueError("bootstrap sample lacks objective labels")
        draws[iteration] = summed[:, 1:] / summed[:, :1]
        if (iteration + 1) % 100 == 0 or iteration + 1 == iterations:
            logger.info(
                "audit_bootstrap_progress",
                extra={
                    "stage": f"bootstrap {iteration + 1}/{iterations}",
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
            )
    if [row["neural_k"] for row in report["points"]] != depths:
        raise ValueError("reported depths mismatch")
    for j, row in enumerate(report["points"]):
        base = totals[:, 1] / denominator
        neural = totals[:, 2 + 3 * j] / denominator
        union = totals[:, 3 + 3 * j] / denominator
        delta = union - base
        for i, objective in enumerate(OBJECTIVES):
            expected = {
                "denominator": denominator[i],
                "base_ceiling": base[i],
                "neural_ceiling": neural[i],
                "union_ceiling": union[i],
                "incremental_ceiling": delta[i],
                "neural_only_positive_hits": totals[i, 4 + 3 * j],
                "incremental_ci95": np.quantile(draws[:, i, j], [0.025, 0.975]),
            }
            for key, value in expected.items():
                _match(row["objectives"][objective][key], value, f"{objective}/{j}/{key}")
        for name, values_at_depth in (
            ("base", base),
            ("neural", neural),
            ("union", union),
            ("incremental", delta),
        ):
            _match(row[f"weighted_{name}_ceiling"], values_at_depth @ WEIGHTS, name)
        _match(
            row["weighted_incremental_ci95"],
            np.quantile(draws[:, :, j] @ WEIGHTS, [0.025, 0.975]),
            "weighted confidence interval",
        )
    _match(
        report["completed_bucket_compute_seconds"],
        sum(row["elapsed_seconds"] for row in receipts),
        "bucket timing",
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "input_id": expected_input_id,
        "metrics_sha256": sha256_file(directory / "metrics.json"),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "verified_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "numpy_version": np.__version__,
        "verified_parts": buckets,
        "sessions": len(sessions),
        "objective_order": list(OBJECTIVES),
        "aggregate_counts": totals.tolist(),
        "bootstrap_iterations_verified": iterations,
        "bootstrap_method": "independent sample-multiplicity aggregation",
        "absolute_tolerance": 1e-12,
        "receipts": receipts,
        "scope": "Saved count integrity and aggregation; not a replay of raw-label retrieval.",
    }
