"""Independent audit of committed ANN counts and reported search measurements."""

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
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _match(actual: Any, expected: Any, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=False):
        raise ValueError(f"metric mismatch: {name}")


def _verified(root: Path, name: str, run_id: str, receipts: list[dict[str, Any]]) -> Path:
    path = root / name
    receipt = _read(Path(str(path) + ".json"))
    elapsed = float(receipt["elapsed_seconds"])
    if (
        receipt["input_id"] != run_id
        or receipt["bytes"] != path.stat().st_size
        or receipt["sha256"] != sha256_file(path)
        or not math.isfinite(elapsed)
        or elapsed < 0
    ):
        raise ValueError(f"invalid artifact receipt: {name}")
    receipts.append({"path": name, "sha256": receipt["sha256"], "bytes": receipt["bytes"]})
    return path


def _validate_counts(counts: np.ndarray) -> None:
    if (
        counts.ndim != 2
        or counts.shape[1] != 7
        or not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts[:, 0] > 20)
        or np.any(counts[:, 1] > counts[:, 0])
        or np.any(counts[:, 2:] > 1 + 1e-12)
        or np.any(counts[:, [0, 1, 2, 5]] != np.floor(counts[:, [0, 1, 2, 5]]))
        or np.any(counts[:, 2] != (counts[:, 0] > 0))
        or np.any(counts[:, 5] != (counts[:, 1] > 0))
        or np.any((counts[:, 3:5] > 0) != (counts[:, 1:2] > 0))
    ):
        raise ValueError("invalid ranking count invariants")
    _match(counts[:, 6], counts[:, 1] / 20, "precision count")


def _load_counts(
    root: Path,
    family: str,
    buckets: int,
    run_id: str,
    logger: logging.Logger,
    receipts: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    all_sessions, objectives = None, []
    for objective in OBJECTIVES:
        expected = {
            f"part-{i:03d}{suffix}" for i in range(buckets) for suffix in (".npz", ".npz.json")
        }
        if {p.name for p in (root / family / objective).iterdir() if p.is_file()} != expected:
            raise ValueError("incomplete or unexpected count parts")
        session_parts, count_parts = [], []
        for bucket in range(buckets):
            name = f"{family}/{objective}/part-{bucket:03d}.npz"
            path = _verified(root, name, run_id, receipts)
            with np.load(path, allow_pickle=False) as part:
                sessions, counts = part["sessions"], part["counts"]
            if (
                sessions.ndim != 1
                or sessions.dtype.kind not in "iu"
                or np.any(sessions < 0)
                or np.any(sessions % buckets != bucket)
                or np.any(sessions[1:] <= sessions[:-1])
                or counts.shape != (len(sessions), 7)
            ):
                raise ValueError("invalid session coverage or count shape")
            _validate_counts(counts)
            session_parts.append(sessions)
            count_parts.append(counts)
            logger.info("ann_audit_part_verified", extra={"file": name, "bucket": bucket})
        sessions = np.concatenate(session_parts)
        order = np.argsort(sessions)
        sessions = sessions[order]
        if all_sessions is not None and not np.array_equal(sessions, all_sessions):
            raise ValueError("objective session coverage mismatch")
        all_sessions = sessions
        objectives.append(np.concatenate(count_parts)[order])
    if all_sessions is None:
        raise ValueError("no count sessions")
    return all_sessions, np.stack(objectives, axis=1)


def _audit_ranking(counts: np.ndarray, reported: dict[str, Any]) -> list[list[float]]:
    # fsum is independent of the worker's NumPy row-sum implementation.
    totals = np.array([[math.fsum(counts[:, i, j]) for j in range(7)] for i in range(3)])
    if np.any(totals[:, [0, 2]] <= 0):
        raise ValueError("each objective requires labeled sessions")
    _match(reported["sessions"], len(counts), "session count")
    _match(
        reported["weighted_recall_at_20"],
        WEIGHTS @ (totals[:, 1] / totals[:, 0]),
        "official recall",
    )
    for objective, row in zip(OBJECTIVES, totals, strict=True):
        expected = {
            "capped_denominator": row[0],
            "hits_at_20": row[1],
            "labeled_sessions": row[2],
            "recall_at_20": row[1] / row[0],
            "ndcg_at_20": row[3] / row[2],
            "mrr_at_20": row[4] / row[2],
            "hit_rate_at_20": row[5] / row[2],
            "precision_at_20": row[6] / row[2],
        }
        for name, value in expected.items():
            _match(reported["objectives"][objective][name], value, f"{objective}/{name}")
    return totals.tolist()


def _audit_latency(row: dict[str, Any], settings: dict[str, Any]) -> None:
    values = np.asarray(row["observations_ms"], dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
        or len(values) != row["samples"]
        or row["samples"] != row["unique_queries"] * row["repeats"]
        or row["repeats"] != settings["latency_repeats"]
        or row["unique_queries"] != settings["latency_queries"]
        or row["warmup_calls"] != settings["warmup_queries"]
    ):
        raise ValueError("invalid latency observations or sampling contract")
    for percentile in (50, 95, 99):
        _match(row[f"p{percentile}_ms"], np.percentile(values, percentile), "latency percentile")


def audit_ann(directory: Path, *, expected_run_id: str, logger: logging.Logger) -> dict[str, Any]:
    """Verify stored evidence without running an encoder, index, or retrieval.

    Full-fold counts and their paired interval are independently aggregated.
    Fidelity selection and latency summaries are checked against recorded
    observations; neighbor sets and raw-label retrieval are not replayed.
    """
    started = time.perf_counter()
    receipts: list[dict[str, Any]] = []
    records = {
        name: _read(_verified(directory, name, expected_run_id, receipts))
        for name in (
            "metrics.json",
            "contract.json",
            "cohort.json",
            "selection.json",
            "prediction_export/prediction_manifest.json",
        )
    }
    report, contract = records["metrics.json"], records["contract.json"]
    cohort, selection = records["cohort.json"], records["selection.json"]
    prediction = records["prediction_export/prediction_manifest.json"]
    settings = contract["settings"]
    if (
        report["status"] != "passed"
        or report["schema_version"] != 1
        or report["input_id"] != expected_run_id
        or report["contract"] != contract
        or settings["run_id"] != expected_run_id
        or canonical_json_sha256(cohort) != contract["cohort_sha256"]
        or prediction["input_id"] != expected_run_id
        or prediction["status"] != "passed"
        or report["reference_input_id"] != prediction["reference_input_id"]
        or report["validation_fold"] != prediction["validation_fold"]
        or report["code_commit"] != prediction["code_commit"]
    ):
        raise ValueError("ANN report provenance mismatch")
    tune, confirm = cohort["tuning"], cohort["confirmation"]
    if (
        len(tune) != report["tuning_sessions"]
        or len(confirm) != report["confirmation_sessions"]
        or len(tune) + len(confirm) != settings["sample_sessions"]
        or len(set(tune + confirm)) != len(tune) + len(confirm)
    ):
        raise ValueError("invalid tuning and confirmation cohorts")
    probes = settings["probes"]
    if sorted(map(int, report["tuning"])) != sorted(set(probes)):
        raise ValueError("probe sweep mismatch")
    depth, target = str(settings["candidate_depth"]), settings["target_overlap"]
    eligible = [
        p
        for p in probes
        if all(
            report["tuning"][str(p)]["search"][o]["fidelity"][depth] >= target for o in OBJECTIVES
        )
    ]
    chosen = min(eligible) if eligible else None
    if (
        chosen is None
        or chosen != report["selected_nprobe"]
        or selection["selected_nprobe"] != chosen
        or selection["confirmation_used_for_selection"]
        or selection["target_overlap"] != target
        or not report["confirmation_fidelity_passed"]
        or not all(
            report["confirmation"]["search"][o]["fidelity"][depth] >= target for o in OBJECTIVES
        )
        or prediction["search"]["nprobe"] != chosen
    ):
        raise ValueError("selection or confirmation gate mismatch")
    latency_groups = 0
    for stage, data in [*report["tuning"].items(), ("confirmation", report["confirmation"])]:
        cohort_size = len(confirm) if stage == "confirmation" else len(tune)
        for objective in OBJECTIVES:
            search = data["search"][objective]
            for fidelity in search["fidelity"].values():
                if not math.isfinite(fidelity) or not 0 <= fidelity <= 1:
                    raise ValueError("invalid neighbor fidelity")
            _audit_latency(search["latency"], settings)
            _match(
                search["batch_throughput_queries_per_second"],
                cohort_size / search["batch_search_seconds"],
                "throughput",
            )
            latency_groups += 1
    for row in report["exact_cpu_latency_on_tuning_queries"].values():
        _audit_latency(row, settings)
        latency_groups += 1
    buckets = prediction["ranking_manifest"]["config"]["buckets"]
    if buckets <= 0:
        raise ValueError("invalid bucket count")
    sessions, exact = _load_counts(
        directory, "reference", buckets, expected_run_id, logger, receipts
    )
    ann_sessions, approximate = _load_counts(
        directory, "prediction_export/counts", buckets, expected_run_id, logger, receipts
    )
    if (
        not np.array_equal(sessions, ann_sessions)
        or len(sessions) != prediction["sessions"]
        or not set(tune + confirm).issubset(sessions)
        or not np.array_equal(exact[:, :, [0, 2]], approximate[:, :, [0, 2]])
    ):
        raise ValueError("paired session or label coverage mismatch")
    totals = {
        "exact": _audit_ranking(exact, report["full_reference_ranking"]),
        "ann": _audit_ranking(approximate, report["full_ann_ranking"]),
    }
    bootstrap = report["full_ann_paired_uncertainty"]
    if (
        bootstrap["iterations"] != 500
        or bootstrap["seed"] != settings["seed"]
        or bootstrap["unit"] != "paired session"
    ):
        raise ValueError("bootstrap contract mismatch")
    flat = np.concatenate((exact[:, :, 0], approximate[:, :, 1] - exact[:, :, 1]), axis=1)
    rng = np.random.default_rng(bootstrap["seed"])
    draws = []
    for i in range(bootstrap["iterations"]):
        multiplicity = np.bincount(
            rng.integers(0, len(sessions), size=len(sessions)), minlength=len(sessions)
        )
        summed = multiplicity @ flat
        if np.any(summed[:3] <= 0):
            raise ValueError("bootstrap sample lacks objective labels")
        draws.append(float(WEIGHTS @ (summed[3:] / summed[:3])))
        if (i + 1) % 100 == 0:
            logger.info(
                "ann_audit_bootstrap_progress",
                extra={"step": i + 1, "elapsed_seconds": round(time.perf_counter() - started, 3)},
            )
    interval = np.quantile(draws, [0.025, 0.975]).tolist()
    _match(bootstrap["weighted_recall_at_20_delta_ci95"], interval, "paired confidence interval")
    return {
        "schema_version": 1,
        "status": "passed",
        "input_id": expected_run_id,
        "metrics_sha256": sha256_file(directory / "metrics.json"),
        "auditor_source_sha256": sha256_file(Path(__file__)),
        "verified_at_utc": utc_now_iso(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "sessions": len(sessions),
        "verified_count_parts": 6 * buckets,
        "aggregate_counts": totals,
        "selected_nprobe": chosen,
        "latency_groups_verified": latency_groups,
        "bootstrap_iterations_verified": 500,
        "weighted_recall_at_20_delta_ci95": interval,
        "bootstrap_method": "independent sample-multiplicity aggregation",
        "numpy_version": np.__version__,
        "absolute_tolerance": 1e-12,
        "receipts": receipts,
        "scope": (
            "Saved count integrity, full-fold ranking aggregation and paired interval, "
            "selection logic, recorded latency percentiles and throughput. Does not replay "
            "raw labels, neighbor sets, or GPU execution."
        ),
    }
