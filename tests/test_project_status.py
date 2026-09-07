"""Status must remain useful in a fresh checkout and reject mismatched evidence."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def run_status(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/project_status.py", "--root", str(root), "--json"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_status_uses_published_evidence_without_local_data(tmp_path: Path) -> None:
    shutil.copytree("reports/metrics", tmp_path / "reports/metrics")
    result = run_status(tmp_path)
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout)
    assert status["ann_comparison"] == "passed"
    assert status["sessions"] == 103468
    assert "nested validation" in status["next_task"]
    assert status["paid_compute_started"] is False
    assert status["timestamp"].endswith("+00:00")
    assert status["elapsed_seconds"] >= 0


def test_status_with_no_published_evidence_stays_pending(tmp_path: Path) -> None:
    result = run_status(tmp_path)
    assert result.returncode == 0
    assert json.loads(result.stdout)["ann_comparison"] == "pending"


@pytest.mark.parametrize("change", ["metric", "identity", "missing_audit"])
def test_status_cannot_certify_altered_or_missing_audit(tmp_path: Path, change: str) -> None:
    target = tmp_path / "reports/metrics"
    shutil.copytree("reports/metrics", target)
    report = target / "two_tower_fold0_ann_comparison.json"
    value = json.loads(report.read_text())
    if change == "metric":
        value["points"][-1]["weighted_union_ceiling"] = 1.0
    elif change == "identity":
        value["prediction_input_id"] = "another-run"
    else:
        (target / "two_tower_fold0_ann_comparison_audit.json").unlink()
    report.write_text(json.dumps(value))
    result = run_status(tmp_path)
    assert result.returncode != 0
    assert "does not match its audit" in result.stderr
