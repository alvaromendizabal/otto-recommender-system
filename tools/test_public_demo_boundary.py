"""Regression checks for the generated report's exact publication boundary."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

from check_public_demo import MAX_BYTES, verify, verify_html

from otto_recsys.public_demo import run_demo


class PublicationBoundaryTests(unittest.TestCase):
    fixture: ClassVar[tempfile.TemporaryDirectory[str]]
    source: ClassVar[Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = tempfile.TemporaryDirectory()
        cls.source = Path(cls.fixture.name) / "source"
        run_demo(cls.source)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.fixture.cleanup()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.generated = self.root / "generated"
        shutil.copytree(self.source, self.generated)
        self.dist = self.root / "dist"

    def test_source_and_staged_report_are_verified(self) -> None:
        report = verify(self.generated, self.dist, stage=True)
        self.assertEqual(report["hosted_files"], ["index.html"])
        self.assertEqual(report["evidence_type"], "SYNTHETIC_ONLY")
        self.assertEqual(verify(self.generated, self.dist), report)
        self.assertEqual(
            (self.dist / "index.html").read_bytes(),
            (self.generated / "report.html").read_bytes(),
        )

    def test_source_only_check_writes_nothing(self) -> None:
        before = {p.name: p.read_bytes() for p in self.generated.iterdir()}
        self.assertEqual(verify(self.generated)["hosted_files"], [])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.generated.iterdir()})

    def test_staging_requires_empty_directory(self) -> None:
        verify(self.generated, self.dist, stage=True)
        with self.assertRaisesRegex(ValueError, "empty"):
            verify(self.generated, self.dist, stage=True)

    def test_extra_hosted_json_rejected(self) -> None:
        verify(self.generated, self.dist, stage=True)
        (self.dist / "result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "hosted files"):
            verify(self.generated, self.dist)

    def test_hosted_nested_directory_rejected(self) -> None:
        verify(self.generated, self.dist, stage=True)
        (self.dist / "assets").mkdir()
        with self.assertRaisesRegex(ValueError, "hosted files"):
            verify(self.generated, self.dist)

    def test_modified_hosted_bytes_rejected(self) -> None:
        verify(self.generated, self.dist, stage=True)
        page = self.dist / "index.html"
        raw = page.read_bytes()
        page.write_bytes(b"X" + raw[1:])
        with self.assertRaisesRegex(ValueError, "byte-identical"):
            verify(self.generated, self.dist)

    def test_source_checksum_change_rejected(self) -> None:
        with (self.generated / "report.html").open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify(self.generated)

    def test_resealed_report_still_requires_exact_renderer(self) -> None:
        path = self.generated / "report.html"
        raw = path.read_bytes() + b" "
        path.write_bytes(raw)
        manifest_path = self.generated / "manifest.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["files"]["report.html"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "renderer"):
            verify(self.generated)

    def test_extra_source_rejected(self) -> None:
        (self.generated / "weights.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "source files"):
            verify(self.generated)

    def test_symlinked_source_rejected(self) -> None:
        link = self.root / "link"
        link.symlink_to(self.generated, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlinked"):
            verify(link)

    def test_symlinked_host_file_rejected(self) -> None:
        self.dist.mkdir()
        (self.dist / "index.html").symlink_to(self.generated / "report.html")
        with self.assertRaisesRegex(ValueError, "Unsafe hosted"):
            verify(self.generated, self.dist)

    def test_oversized_report_rejected_before_read(self) -> None:
        path = self.generated / "report.html"
        with path.open("wb") as stream:
            stream.truncate(MAX_BYTES + 1)
        with self.assertRaisesRegex(ValueError, "Oversized"):
            verify(self.generated)

    def test_https_navigation_and_fragments_allowed(self) -> None:
        verify_html(
            b'<p>Synthetic only</p><a href="https://github.com/example/repo">Source</a>'
            b'<a href="#viewer">Viewer</a><script>const x = 2;</script>'
        )

    def test_hidden_scope_is_insufficient(self) -> None:
        with self.assertRaisesRegex(ValueError, "visible synthetic"):
            verify_html(b'<script>const evidence = "synthetic";</script><p>Report</p>')

    def test_external_runtime_assets_rejected(self) -> None:
        for attack in (
            '<script src="https://example.com/a.js"></script>',
            '<img src="https://example.com/pixel">',
            '<link rel="stylesheet" href="https://example.com/a.css">',
            '<style>@import "https://example.com/a.css";</style>',
            "<style>a{background:url(https://example.com/pixel)}</style>",
        ):
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                verify_html(("<p>Synthetic</p>" + attack).encode())

    def test_network_and_imports_rejected(self) -> None:
        for attack in (
            'fetch("https://example.com")',
            'new WebSocket("wss://example.com")',
            'navigator.sendBeacon("https://example.com", "x")',
            'import("./missing.js")',
            'import {x} from "./missing.js";',
        ):
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                verify_html(("<p>Synthetic</p><script>" + attack + "</script>").encode())

    def test_active_document_and_event_handlers_rejected(self) -> None:
        for attack in (
            "<iframe></iframe>",
            "<object></object>",
            "<form></form>",
            '<meta http-equiv="refresh" content="0;url=https://example.com">',
            '<button onclick="go()">Go</button>',
            "<svg><foreignObject/></svg>",
        ):
            with self.subTest(attack=attack), self.assertRaises(ValueError):
                verify_html(("<p>Synthetic</p>" + attack).encode())

    def test_unhosted_and_active_navigation_rejected(self) -> None:
        for href in (
            "predictions.json",
            "../secret",
            "javascript:alert(1)",
            "//example.com",
            "data:text/html,x",
            "https://user:password@example.com",
        ):
            with self.subTest(href=href), self.assertRaises(ValueError):
                verify_html(f'<p>Synthetic</p><a href="{href}">Link</a>'.encode())

    def test_credentials_and_private_locators_rejected(self) -> None:
        for value in (
            "AKIA" + "A" * 16,
            "ghp_" + "a" * 36,
            "s3://private/data",
            "/workspace/private/data",
            "-----BEGIN PRIVATE KEY-----",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                verify_html(f"<p>Synthetic {value}</p>".encode())


if __name__ == "__main__":
    unittest.main()
