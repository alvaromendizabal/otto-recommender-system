"""Publication safeguards and scope contracts, independent of private model state."""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location("public_boundary", ROOT / "tools/check_public_boundary.py")
assert SPEC and SPEC.loader
BOUNDARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOUNDARY)


def test_publication_scanner_detects_identifiers_without_echoing_them() -> None:
    account = "987654" + "321098"
    key = "AK" + "IA" + "A" * 16
    text = "arn:aws:iam::" + account + ":role/Example " + key
    assert set(BOUNDARY.violations(text)) == {"deployment-account-id", "aws-access-key"}
    assert BOUNDARY.violations("arn:aws:iam::000000000000:role/Example") == []


def test_closeout_keeps_measurement_and_backup_limits() -> None:
    report = json.loads((ROOT / "reports/closeout/summary.json").read_text())
    assert report["existing_submission_sha256_verified"] is True
    assert report["new_submission"] is False
    assert report["complete_private_state_backup_verified"] is False
    assert report["release"]["private_score"] == 0.57586
    assert report["last_ranking_comparison"]["primary_hits"] < report["last_ranking_comparison"]["incumbent_hits"]


def test_reproduction_docs_do_not_promise_the_private_neural_release() -> None:
    guide = (ROOT / "docs/QUICKSTART.md").read_text()
    assert "synthetic" in guide.lower()
    assert "--execute" in guide
    assert "not" in guide and "neural" in guide
    assert "does not stop a cloud instance" in guide


def test_new_markdown_local_links_exist() -> None:
    for relative in ("README.md", "docs/QUICKSTART.md", "docs/DECISIONS.md", "docs/RESEARCH_STATUS.md"):
        path = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            target = target.split("#", 1)[0]
            assert (path.parent / target).exists(), (relative, target)
