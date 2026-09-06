from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from otto_recsys.experiments.manifest import canonical_json_sha256, sha256_file
from otto_recsys.retrieval.comparison_audit import audit_comparison
from otto_recsys.retrieval.neural_evaluation import session_counts, summarize_counts


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


@pytest.fixture
def comparison(tmp_path: Path) -> tuple[Path, str]:
    contract = {
        "schema_version": 1,
        "depths": [20, 40],
        "bootstrap_iterations": 20,
        "bootstrap_seed": 17,
        "prediction_input_id": "saved-prediction-identity",
    }
    input_id = canonical_json_sha256(contract)
    sessions = np.arange(12, dtype=np.int64)
    counts = np.array(
        [
            [
                session_counts(set(range(25)), set(range(session % 7)), list(range(40)), (20, 40))
                for _ in range(3)
            ]
            for session in sessions
        ],
        dtype=np.int32,
    )
    (tmp_path / "parts").mkdir()
    for bucket in range(2):
        path = tmp_path / "parts" / f"part-{bucket:03d}.npz"
        subset = sessions % 2 == bucket
        np.savez_compressed(path, sessions=sessions[subset], counts=counts[subset])
        write_json(
            path.with_suffix(".json"),
            {
                "input_id": input_id,
                "sha256": sha256_file(path),
                "elapsed_seconds": 0.25,
            },
        )
    report = {
        "schema_version": 1,
        "status": "passed",
        "contract": contract,
        "input_id": input_id,
        "prediction_input_id": contract["prediction_input_id"],
        "sessions": len(sessions),
        "completed_bucket_compute_seconds": 0.5,
        **summarize_counts(counts, depths=(20, 40), iterations=20, seed=17),
    }
    write_json(tmp_path / "metrics.json", report)
    write_json(tmp_path / "comparison_contract.json", contract)
    return tmp_path, input_id


def run_audit(comparison: tuple[Path, str]) -> dict:
    directory, identity = comparison
    return audit_comparison(
        directory, expected_input_id=identity, buckets=2, logger=logging.getLogger("audit-test")
    )


@pytest.mark.parametrize("valid", [True, False])
def test_audit_cli_reports_success_and_failure_with_utc_totals(
    comparison: tuple[Path, str],
    valid: bool,
) -> None:
    directory, identity = comparison
    destination = directory / "audit.json"
    process = subprocess.run(
        [
            sys.executable,
            "scripts/audit_two_tower_comparison.py",
            "--comparison-dir",
            str(directory),
            "--expected-input-id",
            identity if valid else "wrong",
            "--buckets",
            "2",
            "--report",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert (process.returncode == 0) is valid
    assert destination.exists() is valid
    assert ("OTTO_TWO_TOWER_COMPARISON_AUDIT_PASSED" in process.stdout) is valid
    rows = [
        json.loads(line)
        for line in (directory / "logs/two_tower_comparison_audit.jsonl").read_text().splitlines()
    ]
    assert rows[-1]["status"] == ("passed" if valid else "failed")
    assert rows[-1]["elapsed_seconds"] >= 0
    assert all(row["timestamp"].endswith("+00:00") for row in rows)


def test_independent_audit_reproduces_capped_metrics_and_paired_intervals(
    comparison: tuple[Path, str],
) -> None:
    result = run_audit(comparison)
    assert result["status"] == "passed"
    assert result["verified_parts"] == 2
    assert result["sessions"] == 12
    assert result["bootstrap_iterations_verified"] == 20
    assert result["aggregate_counts"][0][0] == 240
    # Raw exclusive positives can exceed capped incremental hits.
    assert result["aggregate_counts"][0][-1] > (
        result["aggregate_counts"][0][-2] - result["aggregate_counts"][0][1]
    )


@pytest.mark.parametrize("mutation", ["estimate", "interval", "identity", "coverage", "timing"])
def test_audit_rejects_misreported_results(comparison: tuple[Path, str], mutation: str) -> None:
    path = comparison[0] / "metrics.json"
    report = json.loads(path.read_text())
    if mutation == "estimate":
        report["points"][0]["weighted_union_ceiling"] += 0.01
    elif mutation == "interval":
        report["points"][0]["objectives"]["carts"]["incremental_ci95"][0] += 0.01
    elif mutation == "identity":
        report["prediction_input_id"] = "another-run"
    elif mutation == "coverage":
        report["sessions"] += 1
    else:
        report["completed_bucket_compute_seconds"] += 1
    write_json(path, report)
    with pytest.raises(ValueError):
        run_audit(comparison)


@pytest.mark.parametrize(
    "mutation",
    ["hash", "missing", "negative", "duplicate", "bucket", "cap", "decreasing", "nonfinite_time"],
)
def test_audit_rejects_invalid_parts_even_with_matching_receipts(
    comparison: tuple[Path, str],
    mutation: str,
) -> None:
    path = comparison[0] / "parts/part-000.npz"
    receipt_path = path.with_suffix(".json")
    receipt = json.loads(receipt_path.read_text())
    if mutation == "missing":
        receipt_path.unlink()
    elif mutation == "hash":
        path.write_bytes(b"interrupted-upload")
    elif mutation == "nonfinite_time":
        receipt["elapsed_seconds"] = float("nan")
        write_json(receipt_path, receipt)
    else:
        with np.load(path, allow_pickle=False) as part:
            sessions, counts = part["sessions"], part["counts"]
        if mutation == "negative":
            counts[0, 0, 1] = -1
        elif mutation == "duplicate":
            sessions[1] = sessions[0]
        elif mutation == "bucket":
            sessions[0] += 1
        elif mutation == "cap":
            counts[0, 0, 3] = 21
        else:
            counts[0, 0, -1] = 19
        np.savez_compressed(path, sessions=sessions, counts=counts)
        receipt["sha256"] = sha256_file(path)
        write_json(receipt_path, receipt)
    with pytest.raises(ValueError):
        run_audit(comparison)
