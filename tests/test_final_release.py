"""Retained-release arithmetic, checkpoint locks and publication boundaries."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from otto_recsys.checkpoints import workspace_lock
from otto_recsys.portfolio_review import RELEASE, review_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_final_evidence_has_separate_matched_and_external_scopes() -> None:
    result = review_snapshot(ROOT)
    assert result["verified_release"]["private_score"] == 0.57586
    assert result["verified_release"]["rows"] == 5015409
    assert result["controlled_study"]["weighted_recall_at_20"]["selected"] == pytest.approx(
        0.5843921411460516
    )
    assert result["controlled_study"]["sessions"] == 432492


@pytest.mark.parametrize("key,value", [("private_score", 0.99), ("rows", 1), ("sessions", True)])
def test_modified_receipt_is_rejected(tmp_path: Path, key: str, value: object) -> None:
    shutil.copytree(ROOT / "reports", tmp_path / "reports")
    p = tmp_path / RELEASE
    result = json.loads(p.read_text())
    result[key] = value
    p.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="checksum"):
        review_snapshot(tmp_path)


def test_checkpoint_lock_excludes_second_writer_and_releases(tmp_path: Path) -> None:
    with (
        workspace_lock(tmp_path),
        pytest.raises(RuntimeError, match="active writer"),
        workspace_lock(tmp_path),
    ):
        raise AssertionError("second writer entered")
    with workspace_lock(tmp_path):
        assert (tmp_path / ".lock").is_file()


def test_public_docs_link_only_to_retained_files() -> None:
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            assert (path.parent / target.split("#", 1)[0]).exists(), (path, target)


def test_retired_gpu_project_is_not_a_public_dependency() -> None:
    assert not (ROOT / "gpu").exists()
    assert not (ROOT / "research").exists()
    for p in (ROOT / "src").rglob("*.py"):
        assert "two_tower" not in p.read_text(), p
