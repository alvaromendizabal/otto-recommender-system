"""Public figures must reject changed evidence and inconsistent scientific claims."""

from __future__ import annotations

import hashlib
import io
import json
import runpy
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PORTFOLIO = runpy.run_path(str(ROOT / "scripts/build_portfolio_figures.py"))


def replace_report(directory: Path, name: str, value: dict[str, Any]) -> None:
    """Represent deliberate republication, rather than just damaged file bytes."""
    path = directory / name
    path.write_text(json.dumps(value))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


class TestPortfolioEvidence(unittest.TestCase):
    def test_download_archive_contains_one_portable_html(self) -> None:
        html = "<!doctype html><html><body>OTTO · offline report</body></html>"
        packaged = PORTFOLIO["report_archive"](html)
        self.assertEqual(packaged, PORTFOLIO["report_archive"](html))
        with zipfile.ZipFile(io.BytesIO(packaged)) as archive:
            self.assertEqual(archive.namelist(), ["otto-research-report.html"])
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read("otto-research-report.html").decode(), html)

    def test_committed_research_is_valid_for_public_figures(self) -> None:
        data = PORTFOLIO["read_evidence"](ROOT)
        self.assertEqual(data["evaluation"]["sessions"], data["audit"]["statistics"]["sessions"])

    def test_changed_evidence_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "reports/research"
            shutil.copytree(ROOT / "reports/research", directory)
            (directory / "evaluation.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                PORTFOLIO["read_evidence"](root)

    def test_inconsistent_republished_claims_are_rejected(self) -> None:
        for change in ("lineage", "weighted_score", "paired_gain", "label_access"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                directory = root / "reports/research"
                shutil.copytree(ROOT / "reports/research", directory)
                name = "evaluation_seal.json" if change == "label_access" else "evaluation.json"
                report = json.loads((directory / name).read_text())
                if change == "lineage":
                    report["seal_id"] = "different-model-lineage"
                elif change == "weighted_score":
                    report["scores"]["selected"]["weighted_recall_at_20"] = 0.99
                elif change == "paired_gain":
                    report["paired_intervals"]["core"]["absolute_gain"] = 0.25
                else:
                    report["evaluation_labels_consulted"] = True
                replace_report(directory, name, report)
                with self.assertRaises(ValueError):
                    PORTFOLIO["read_evidence"](root)
