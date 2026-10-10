"""Verify and optionally stage the single-file synthetic public report.

The report remains byte-identical to the current authenticated renderer output.
This publication boundary complements behavioral tests; it is not a JavaScript
security sandbox or evidence about the historical research model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otto_recsys.public_demo import verify_output  # noqa: E402
from otto_recsys.public_demo_report import render_report  # noqa: E402

SOURCE_FILES = {"report.html", "predictions.json", "result.json", "manifest.json"}
MAX_BYTES = 2_000_000
SECRET = re.compile(
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"github_pat_[A-Za-z0-9_]{30,}|KGAT_[A-Za-z0-9_-]{25,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
    r"[?&](?:X-Amz-Signature|X-Amz-Credential|X-Goog-Signature)="
)
PRIVATE = re.compile(r"s3://|arn:aws(?:-[a-z]+)?:|/(?:home|workspace|mnt)/", re.IGNORECASE)
NETWORK = re.compile(
    r"\b(?:fetch|XMLHttpRequest|WebSocket|EventSource|importScripts)\s*\("
    r"|\bsendBeacon\s*\("
)
IMPORT = re.compile(r"\bimport\s*(?:\(|[\"'{*])|\b(?:import|export)\s+[^;\n]+\s+from\s")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


class ReportBoundary(HTMLParser):
    """Permit inline report content and explicit navigation, never runtime assets."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.visible: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        require(tag not in {"iframe", "object", "embed", "base", "form"}, "Active container")
        require(tag != "foreignobject", "Active SVG content")
        require(not any(key.startswith("on") for key in attributes), "Inline event handler")
        require(not any(key in attributes for key in ("src", "srcset", "poster")), "Runtime asset")
        if tag in {"script", "style"}:
            self.hidden_depth += 1
        if tag == "meta":
            require(attributes.get("http-equiv", "").lower() != "refresh", "Automatic redirect")
        for key in ("href", "xlink:href"):
            if key not in attributes:
                continue
            reference = attributes[key]
            if reference.startswith("#"):
                continue
            target = urlsplit(reference)
            navigation = tag == "a" or (
                tag == "link" and attributes.get("rel", "").lower() == "canonical"
            )
            require(
                navigation and target.scheme == "https" and bool(target.netloc),
                "Unhosted reference or runtime asset",
            )
            require(not target.username and not target.password, "Credential-bearing URL")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self.hidden_depth = max(0, self.hidden_depth - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.visible.append(data)


def verify_html(raw: bytes) -> None:
    require(0 < len(raw) <= MAX_BYTES, "Missing or oversized public report")
    text = raw.decode("utf-8")
    require(not SECRET.search(text), "Credential-like public material")
    require(not PRIVATE.search(text), "Private operational locator")
    require(not NETWORK.search(text), "Runtime network access")
    require(not IMPORT.search(text), "Runtime module import")
    require(not re.search(r"@import\b|\burl\s*\(", text, re.IGNORECASE), "CSS runtime asset")
    parser = ReportBoundary()
    parser.feed(text)
    parser.close()
    require("synthetic" in " ".join(parser.visible).lower(), "Missing visible synthetic scope")


def verified_report(source: Path) -> bytes:
    require(source.is_dir() and not source.is_symlink(), "Missing or symlinked source directory")
    require({path.name for path in source.iterdir()} == SOURCE_FILES, "Unexpected source files")
    for name in SOURCE_FILES:
        path = source / name
        require(path.is_file() and not path.is_symlink(), "Unsafe source artifact")
        require(0 < path.stat().st_size <= MAX_BYTES, "Oversized source artifact")
    result = verify_output(source)
    require(result.get("synthetic") is True, "Result must be explicitly synthetic")
    require(
        result.get("provenance", {}).get("targets_used_in_fit_or_inference") is False,
        "Missing target isolation contract",
    )
    raw = (source / "report.html").read_bytes()
    require(raw == render_report(result).encode("utf-8"), "Report differs from current renderer")
    verify_html(raw)
    return raw


def verify(source: Path, dist: Path | None = None, *, stage: bool = False) -> dict[str, Any]:
    raw = verified_report(source)
    if stage:
        require(dist is not None, "Staging requires a destination")
    if dist is not None:
        require(not dist.is_symlink(), "Symlinked public directory")
        if stage:
            dist.mkdir(parents=True, exist_ok=True)
            require(not any(dist.iterdir()), "Staging destination must be empty")
            (dist / "index.html").write_bytes(raw)
        require(dist.is_dir(), "Missing public directory")
        require({path.name for path in dist.iterdir()} == {"index.html"}, "Unexpected hosted files")
        page = dist / "index.html"
        require(page.is_file() and not page.is_symlink(), "Unsafe hosted entry point")
        require(page.stat().st_size == len(raw), "Hosted report size differs")
        require(page.read_bytes() == raw, "Hosted report is not byte-identical")
    return {
        "status": "PASS",
        "evidence_type": "SYNTHETIC_ONLY",
        "scope": "single_html_publication_boundary",
        "source_verification": "current_implementation_manifest_and_exact_renderer",
        "source_path": "report.html",
        "hosted_path": "index.html",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "hosted_files": ["index.html"] if dist is not None else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/public_demo")
    parser.add_argument("--dist", type=Path)
    parser.add_argument("--stage", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.source, args.dist, stage=args.stage), indent=2))


if __name__ == "__main__":
    main()
