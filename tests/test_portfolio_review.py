"""Public review rejects stale identities and contradictory metric/release claims."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from otto_recsys.portfolio_review import (
    AUDIT,
    EVALUATION,
    FRONTIER,
    MANIFEST,
    PROMOTION,
    RELEASE,
    review_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path: Path) -> Path:
    for name in (AUDIT, EVALUATION, FRONTIER, MANIFEST, PROMOTION, RELEASE):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return tmp_path


def test_review_recomputes_counts_and_separates_evaluation_scopes() -> None:
    snapshot = review_snapshot(ROOT)
    assert snapshot["verified_release"]["private_score"] == 0.57586
    assert snapshot["controlled_study"]["weighted_recall_at_20"]["selected"] == pytest.approx(
        0.5843921411460516
    )
    assert snapshot["candidate_diagnostic"]["outside_pool_misses"] == 1954
    assert snapshot["research_status"] == "rejected_after_corrected_comparator"


@pytest.mark.parametrize("file,key,value,message", [
    (RELEASE, "private_score", 0.99, "release score"),
    (RELEASE, "public_score", float("nan"), "Invalid number"),
    (RELEASE, "sessions", True, "Invalid count"),
    (RELEASE, "rows", 5015408, "coverage"),
    (RELEASE, "submission_ref", 1, "submission reference"),
    (RELEASE, "sha256", "a" * 64, "prediction identity"),
    (RELEASE, "evaluation_setting", "official competition winner", "post-competition"),
    (EVALUATION, "sessions", 1, "checksum mismatch"),
])
def test_review_rejects_contradictory_release_or_modified_evidence(
    evidence: Path, file: str, key: str, value: object, message: str
) -> None:
    path = evidence / file
    report = json.loads(path.read_text())
    report[key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match=message):
        review_snapshot(evidence)


@pytest.mark.parametrize("field,value", [
    ("within_pool_misses", 1),
    ("outside_pool_misses", -1),
    ("strongest_ranked_cart_hits", 3000),
    ("cart_denominator", True),
])
def test_review_rejects_false_candidate_diagnosis(evidence: Path, field: str, value: int) -> None:
    path = evidence / FRONTIER
    report = json.loads(path.read_text())
    report["research_frontier"]["time_controlled_diagnostics"][field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        review_snapshot(evidence)


@pytest.mark.parametrize("optimized", [False, True])
def test_dependency_free_command_from_another_directory(tmp_path: Path, optimized: bool) -> None:
    command = [sys.executable, "-S"] + (["-O"] if optimized else [])
    command += [str(ROOT / "scripts/review_portfolio.py"), "--json"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 0, result.stderr
    snapshot = json.loads(result.stdout)
    assert snapshot["status"] == "passed"
    assert snapshot["verified_release"]["rows"] == 5015409
    assert list(tmp_path.iterdir()) == []
