"""Keep the offline results explorer aligned with its generator and live review links."""

import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import unittest
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/portfolio"
REPO_PATH = "/alvaromendizabal/otto-recommender-system/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
        self.ids = set()
        self.chart_regions = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "chart" in attrs.get("class", "").split():
            self.chart_regions.append(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        if tag == "a" and "href" in attrs:
            self.hrefs.append(attrs["href"])


def broken_links(html, root):
    """Inspect actual anchors, not URL-like strings inside the embedded Plotly library."""
    parsed = Links()
    parsed.feed(html)
    errors = []
    for href in parsed.hrefs:
        url = urlsplit(href)
        if not url.path and url.fragment and unquote(url.fragment) not in parsed.ids:
            errors.append(href)
        elif url.hostname == "github.com" and url.path.startswith(REPO_PATH):
            parts = url.path[len(REPO_PATH):].split("/", 2)
            if len(parts) == 3 and parts[0] in {"blob", "tree"} and parts[1] == "main":
                path = (root / unquote(parts[2])).resolve()
                if not path.is_relative_to(root.resolve()) or not path.exists():
                    errors.append(href)
    return errors


class ReportArtifactsTests(unittest.TestCase):
    def test_report_contract_matches_maintained_inputs(self):
        receipt = json.loads((REPORT / "manifest.json").read_text())
        sources = {
            "generator_sha256": "scripts/build_portfolio_figures.py",
            "analysis_lock_sha256": "notebooks/requirements.txt",
            "research_manifest_sha256": "reports/research/manifest.json",
            "robustness_comparison_sha256": "reports/robustness/comparison.json",
        }
        for field, name in sources.items():
            with self.subTest(source=name):
                self.assertEqual(
                    receipt["contract"][field],
                    hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                    "Regenerate the derived explorer after changing its inputs",
                )
        self.assertIn("index.html", receipt["files"])
        self.assertIn("otto-research-report.zip", receipt["files"])
        for name, expected in receipt["files"].items():
            path = (REPORT / name).resolve()
            self.assertTrue(path.is_relative_to(REPORT.resolve()))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_saved_report_has_live_review_links(self):
        self.assertEqual(broken_links((REPORT / "index.html").read_text(), ROOT), [])

    def test_wide_charts_are_keyboard_accessible_and_explained(self):
        parsed = Links()
        parsed.feed((REPORT / "index.html").read_text())
        self.assertEqual(len(parsed.chart_regions), 8)
        for region in parsed.chart_regions:
            self.assertEqual(region.get("tabindex"), "0")
            self.assertEqual(region.get("role"), "region")
            self.assertTrue(region.get("aria-label"))
            self.assertIn(region.get("aria-describedby"), parsed.ids)

    def test_download_is_the_same_single_offline_report(self):
        with zipfile.ZipFile(REPORT / "otto-research-report.zip") as archive:
            self.assertEqual(archive.namelist(), ["otto-research-report.html"])
            self.assertEqual(archive.read("otto-research-report.html"), (REPORT / "index.html").read_bytes())
            self.assertIsNone(archive.testzip())

    def test_link_checker_distinguishes_deleted_targets_from_javascript_text(self):
        prefix = "https://github.com" + REPO_PATH + "blob/main/"
        dead = prefix + "docs/absent-review-target.md"
        good = prefix + "docs/MODEL_CARD.md"
        html = f'<script>const text="href=missing";</script><a href="{dead}">bad</a>'
        html += f'<a href="{good}">model</a><a href="#kept">jump</a><h2 id="kept">Kept</h2>'
        self.assertEqual(broken_links(html, ROOT), [dead])
        self.assertEqual(broken_links('<a href="#absent">bad jump</a>', ROOT), ["#absent"])


if __name__ == "__main__":
    unittest.main()
