from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.retrieval.ann_audit import OBJECTIVES, _audit_ranking, audit_ann

RUN = "a" * 64


def receipt(path: Path) -> None:
    Path(str(path) + ".json").write_text(
        json.dumps(
            {
                "input_id": RUN,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "elapsed_seconds": 0.1,
            }
        )
    )


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    receipt(path)


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    cohort = {"tuning": [0, 1], "confirmation": [2, 3]}
    settings = {
        "run_id": RUN,
        "sample_sessions": 4,
        "probes": [1, 2],
        "candidate_depth": 800,
        "target_overlap": 0.98,
        "latency_queries": 2,
        "latency_repeats": 2,
        "warmup_queries": 1,
        "seed": 20260906,
    }
    contract = {"settings": settings, "cohort_sha256": canonical_json_sha256(cohort)}
    latency = {
        "observations_ms": [1, 2, 3, 4],
        "samples": 4,
        "unique_queries": 2,
        "repeats": 2,
        "warmup_calls": 1,
        "p50_ms": 2.5,
        "p95_ms": 3.85,
        "p99_ms": 3.97,
    }

    def search(overlap: float) -> dict:
        return {
            "search": {
                o: {
                    "fidelity": {"800": overlap},
                    "latency": copy.deepcopy(latency),
                    "batch_search_seconds": 2.0,
                    "batch_throughput_queries_per_second": 1.0,
                }
                for o in OBJECTIVES
            }
        }

    rankings = {}
    for family, hit in (("reference", 1), ("prediction_export/counts", 0)):
        for objective in OBJECTIVES:
            for bucket in range(2):
                path = tmp_path / family / objective / f"part-{bucket:03d}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                sessions = np.arange(bucket, 8, 2)
                counts = np.tile([1, hit, 1, hit, hit, hit, hit / 20], (4, 1))
                np.savez(path, sessions=sessions, counts=counts)
                receipt(path)
        rankings[family] = {
            "sessions": 8,
            "weighted_recall_at_20": float(hit),
            "objectives": {
                o: {
                    "capped_denominator": 8,
                    "hits_at_20": 8 * hit,
                    "labeled_sessions": 8,
                    "recall_at_20": hit,
                    "ndcg_at_20": hit,
                    "mrr_at_20": hit,
                    "hit_rate_at_20": hit,
                    "precision_at_20": hit / 20,
                }
                for o in OBJECTIVES
            },
        }
    report = {
        "status": "passed",
        "schema_version": 1,
        "input_id": RUN,
        "contract": contract,
        "reference_input_id": "reference",
        "validation_fold": 0,
        "code_commit": "commit",
        "tuning_sessions": 2,
        "confirmation_sessions": 2,
        "tuning": {"1": search(0.9), "2": search(0.99)},
        "confirmation": search(0.99),
        "selected_nprobe": 2,
        "confirmation_fidelity_passed": True,
        "exact_cpu_latency_on_tuning_queries": {o: copy.deepcopy(latency) for o in OBJECTIVES},
        "full_reference_ranking": rankings["reference"],
        "full_ann_ranking": rankings["prediction_export/counts"],
        "full_ann_paired_uncertainty": {
            "iterations": 500,
            "seed": 20260906,
            "unit": "paired session",
            "weighted_recall_at_20_delta_ci95": [-1.0, -1.0],
        },
    }
    write(tmp_path / "metrics.json", report)
    write(tmp_path / "contract.json", contract)
    write(tmp_path / "cohort.json", cohort)
    write(
        tmp_path / "selection.json",
        {"selected_nprobe": 2, "target_overlap": 0.98, "confirmation_used_for_selection": False},
    )
    write(
        tmp_path / "prediction_export/prediction_manifest.json",
        {
            "status": "passed",
            "input_id": RUN,
            "reference_input_id": "reference",
            "validation_fold": 0,
            "code_commit": "commit",
            "search": {"nprobe": 2},
            "ranking_manifest": {"config": {"buckets": 2}},
            "sessions": 8,
        },
    )
    return tmp_path


def run(path: Path) -> dict:
    return audit_ann(path, expected_run_id=RUN, logger=logging.getLogger("ann-audit-test"))


def test_audit_reproduces_known_complete_loss_and_is_repeatable(evidence: Path) -> None:
    first, second = run(evidence), run(evidence)
    assert first["weighted_recall_at_20_delta_ci95"] == [-1.0, -1.0]
    assert first["verified_count_parts"] == 12
    assert first["latency_groups_verified"] == 12
    assert first["aggregate_counts"] == second["aggregate_counts"]
    assert first["receipts"] == second["receipts"]


@pytest.mark.parametrize("damage", ["metric", "interval", "latency", "selection", "fidelity"])
def test_semantic_tampering_fails_even_with_a_valid_report_receipt(
    evidence: Path, damage: str
) -> None:
    path = evidence / "metrics.json"
    report = json.loads(path.read_text())
    if damage == "metric":
        report["full_ann_ranking"]["weighted_recall_at_20"] = 0.1
    elif damage == "interval":
        report["full_ann_paired_uncertainty"]["weighted_recall_at_20_delta_ci95"] = [-0.1, 0.1]
    elif damage == "latency":
        report["confirmation"]["search"]["clicks"]["latency"]["p95_ms"] = 0.1
    elif damage == "selection":
        report["selected_nprobe"] = 1
    else:
        report["confirmation"]["search"]["orders"]["fidelity"]["800"] = 0.9
    write(path, report)
    with pytest.raises(ValueError, match="mismatch"):
        run(evidence)


@pytest.mark.parametrize("damage", ["missing", "hash", "sessions", "counts", "paired_labels"])
def test_bad_count_evidence_is_rejected(evidence: Path, damage: str) -> None:
    path = evidence / "prediction_export/counts/clicks/part-000.npz"
    with np.load(path) as part:
        sessions, counts = part["sessions"].copy(), part["counts"].copy()
    if damage == "missing":
        path.unlink()
    elif damage == "hash":
        path.write_bytes(b"corrupt")
    else:
        if damage == "sessions":
            sessions[1] = sessions[0]
        elif damage == "counts":
            counts[0, 1] = 2
        else:
            counts[0, 0] = 2
        np.savez(path, sessions=sessions, counts=counts)
        receipt(path)
    with pytest.raises(ValueError):
        run(evidence)


def test_overlapping_cohorts_fail_before_count_aggregation(evidence: Path) -> None:
    cohort = {"tuning": [0, 1], "confirmation": [1, 2]}
    write(evidence / "cohort.json", cohort)
    contract = json.loads((evidence / "contract.json").read_text())
    contract["cohort_sha256"] = canonical_json_sha256(cohort)
    write(evidence / "contract.json", contract)
    report = json.loads((evidence / "metrics.json").read_text())
    report["contract"] = contract
    write(evidence / "metrics.json", report)
    with pytest.raises(ValueError, match="cohorts"):
        run(evidence)


def test_official_metric_uses_summed_capped_denominators() -> None:
    counts = np.zeros((2, 3, 7))
    counts[:, :, 0] = np.array([[1], [20]])
    counts[:, :, 1:] = [1, 1, 0.5, 1, 1, 0.05]
    objective = {
        "capped_denominator": 21,
        "hits_at_20": 2,
        "labeled_sessions": 2,
        "recall_at_20": 2 / 21,
        "ndcg_at_20": 0.5,
        "mrr_at_20": 1,
        "hit_rate_at_20": 1,
        "precision_at_20": 0.05,
    }
    reported = {
        "sessions": 2,
        "weighted_recall_at_20": 2 / 21,
        "objectives": {o: objective.copy() for o in OBJECTIVES},
    }
    _audit_ranking(counts, reported)
    reported["weighted_recall_at_20"] = (1 + 1 / 20) / 2
    with pytest.raises(ValueError, match="official recall"):
        _audit_ranking(counts, reported)
