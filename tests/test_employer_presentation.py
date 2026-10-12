"""Keep the public review surface accessible, navigable and evidence-scoped."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_overview_svg_is_accessible_and_self_contained() -> None:
    path = ROOT / "docs/assets/overview.svg"
    tree = ET.fromstring(path.read_text())
    ns = {"svg": "http://www.w3.org/2000/svg"}
    assert tree.attrib["role"] == "img"
    assert tree.find("svg:title", ns) is not None
    assert tree.find("svg:desc", ns) is not None
    assert tree.attrib["viewBox"] == "0 0 1200 368"
    for element in tree.iter():
        assert element.tag.rsplit("}", 1)[-1] not in {"script", "foreignObject", "image"}
        for name in element.attrib:
            assert not name.lower().startswith("on")
            assert not name.endswith("href")
    assert path.stat().st_size < 16_384


def test_overview_metrics_have_a_readable_scope() -> None:
    tree = ET.parse(ROOT / "docs/assets/overview.svg")
    text = " ".join(tree.getroot().itertext())
    for phrase in ("216.7M", "1.67M", "5.02M", "Historical offline", "synthetic data"):
        assert phrase in text


def test_employer_route_retains_evidence_and_privacy_links() -> None:
    text = (ROOT / "docs/EVIDENCE_MAP.md").read_text()
    assert "0.584392" in text and "0.57586" in text
    assert "not competition-performance measurements" in text
    assert "PUBLICATION_SCOPE.md" in text
    assert "RESEARCH_STATUS.md" in text
    links = re.findall(r"\]\(([^)]+)\)", text)
    assert len(links) >= 20
    for link in links:
        if "://" not in link and not link.startswith("#"):
            assert (ROOT / "docs" / link.split("#", 1)[0]).exists(), link


def test_readme_keeps_results_separate_and_routes_to_working_software() -> None:
    text = (ROOT / "README.md").read_text()
    for phrase in (
        "docs/assets/overview.svg", "docs/EVIDENCE_MAP.md", "reproduce.py review",
        "0.584392", "0.57586", "different populations", "synthetic",
        "corrected click/cart challenger is **rejected**",
    ):
        assert phrase in text
    for phrase in ("9.9/10", "chasing the", "beat the top score"):
        assert phrase not in text.lower()
