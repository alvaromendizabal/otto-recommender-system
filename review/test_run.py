"""Failure-path tests for the public review runner; no optional dependencies."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("employer_review", HERE / "run.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Cannot load review runner")
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


class ReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "source"
        (self.root / "review").mkdir(parents=True)
        for name in ("run.py", "test_run.py", "template.html"):
            shutil.copyfile(HERE / name, self.root / "review" / name)
        (self.root / "proof.py").write_text('print("PUBLIC_CHECK_PASSED")\n')
        (self.root / "evidence.json").write_text('{"kind":"synthetic"}\n')
        (self.root / "README.md").write_text("# Public fixture\n")
        (self.root / "review/README.md").write_text("Public instructions\n")
        self.project = {
            "schema_version": 1,
            "repository": "alvaromendizabal/review-fixture",
            "reviewed_base_commit": "a" * 40,
            "title": "Fixture",
            "discipline": "Test fixture",
            "summary": "Synthetic test project, not research evidence.",
            "scope": "Synthetic test only.",
            "private_boundary": "No private artifacts.",
            "signals": [{"value": "1", "label": "synthetic check"}],
            "read_more": [{"path": "README.md", "label": "Overview"}],
            "cards": [
                {
                    "audience": "ML engineering",
                    "title": "Safe fixture",
                    "decision": "Execute the actual fixture.",
                    "boundary": "Synthetic only.",
                    "implementation": "proof.py",
                    "tests": "review/test_run.py",
                    "evidence": "evidence.json",
                }
            ],
            "commands": [{"id": "fixture", "runtime": "python", "entry": "proof.py"}],
        }
        self.output = self.base / "receipt"
        self.save()

    def save(self) -> None:
        (self.root / "review/project.json").write_text(json.dumps(self.project))

    def run_review(self) -> dict:
        return RUN.execute(self.root, RUN.load_project(self.root), self.output, False, 5)

    def test_valid_run_and_reuse(self) -> None:
        self.assertFalse(self.run_review()["reused"])
        self.assertTrue(self.run_review()["reused"])
        self.assertEqual(
            RUN.verify(self.output, self.root, self.project, False)["status"], "passed"
        )

    def test_nested_manifests_are_verified(self) -> None:
        self.project["commands"][0]["args"] = ["{output}/demo"]
        self.save()
        (self.root / "proof.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "p = Path(sys.argv[1]); p.mkdir()\n"
            "(p / 'manifest.json').write_text('{}')\n"
        )
        self.run_review()
        self.assertTrue(self.run_review()["reused"])
        (self.output / "demo/manifest.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Changed review artifact"):
            self.run_review()

    def test_real_command_timeout_never_publishes_success(self) -> None:
        (self.root / "proof.py").write_text("import time\ntime.sleep(5)\n")
        with self.assertRaises(RUN.subprocess.TimeoutExpired):
            RUN.execute(self.root, self.project, self.output, False, 1)
        self.assertFalse(self.output.exists())
        self.assertEqual(len(list(self.base.glob("receipt-failed-*/failure.json"))), 1)

    def test_changed_source_rejected(self) -> None:
        self.run_review()
        (self.root / "evidence.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_artifact_tampering_rejected(self) -> None:
        self.run_review()
        (self.output / "index.html").write_text("modified")
        with self.assertRaisesRegex(ValueError, "Changed review artifact"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_unlisted_artifact_rejected(self) -> None:
        self.run_review()
        (self.output / "extra.txt").write_text("unexpected")
        with self.assertRaisesRegex(ValueError, "Unexpected or missing"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_failed_command_never_publishes_success(self) -> None:
        (self.root / "proof.py").write_text("raise SystemExit(7)\n")
        with self.assertRaisesRegex(ValueError, "Check failed"):
            self.run_review()
        self.assertFalse(self.output.exists())
        failures = list(self.base.glob("receipt-failed-*/failure.json"))
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0].read_text())["status"], "failed")
        self.assertTrue((failures[0].parent / "fixture.log").is_file())

    def test_path_traversal_rejected(self) -> None:
        for path in ("../outside.txt", "/etc/passwd", "review/../proof.py", "review\\run.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                RUN.local_file(self.root, path)

    def test_symlink_rejected(self) -> None:
        if sys.platform == "win32":
            self.skipTest("Symlink permission is platform-specific")
        (self.root / "link.py").symlink_to(self.root / "proof.py")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            RUN.local_file(self.root, "link.py")

    def test_missing_evidence_rejected(self) -> None:
        (self.root / "evidence.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing reviewed file"):
            RUN.load_project(self.root)

    def test_duplicate_command_rejected(self) -> None:
        self.project["commands"] *= 2
        self.save()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            RUN.load_project(self.root)

    def test_unsupported_runtime_rejected(self) -> None:
        self.project["commands"][0]["runtime"] = "shell"
        self.save()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            RUN.load_project(self.root)

    def test_inside_checkout_output_rejected(self) -> None:
        self.output = self.root / "generated"
        with self.assertRaisesRegex(ValueError, "outside the checkout"):
            self.run_review()

    def test_check_mode_mismatch_rejected(self) -> None:
        self.run_review()
        with self.assertRaisesRegex(ValueError, "Different check mode"):
            RUN.verify(self.output, self.root, self.project, True)

    def test_missing_node_rejected_before_execution(self) -> None:
        with (
            patch.object(RUN.shutil, "which", return_value=None),
            self.assertRaisesRegex(ValueError, "Node.js"),
        ):
            RUN.execute(self.root, self.project, self.output, True, 5)
        self.assertFalse(self.output.exists())

    def test_timeout_rejected_before_execution(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            RUN.execute(self.root, self.project, self.output, False, 0)

    def test_html_escapes_untrusted_text(self) -> None:
        self.project["title"] = "<script>alert(1)</script>"
        page = RUN.page(self.project, root=self.root)
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)
        self.assertIn("No checks were executed", page)
        self.assertNotIn("Selected public checks passed", page)

    def test_render_is_deterministic(self) -> None:
        self.assertEqual(
            RUN.page(self.project, root=self.root), RUN.page(self.project, root=self.root)
        )

    def test_assertions_not_used_as_runtime_gates(self) -> None:
        import ast

        tree = ast.parse((HERE / "run.py").read_text())
        self.assertFalse(any(isinstance(node, ast.Assert) for node in ast.walk(tree)))

    def add_scope(self) -> None:
        (self.root / "engine").mkdir()
        (self.root / "engine/imported.py").write_text("VALUE = 1\n")
        self.project["source_scopes"] = [{"directory": "engine", "suffixes": [".py"]}]
        self.save()

    def test_imported_source_drift_rejects_reuse(self) -> None:
        self.add_scope()
        self.run_review()
        (self.root / "engine/imported.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_added_source_rejects_reuse(self) -> None:
        self.add_scope()
        self.run_review()
        (self.root / "engine/new.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_removed_source_rejects_reuse(self) -> None:
        self.add_scope()
        (self.root / "engine/other.py").write_text("VALUE = 2\n")
        self.run_review()
        (self.root / "engine/other.py").unlink()
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_review()

    def test_runtime_caches_are_not_sources(self) -> None:
        self.add_scope()
        self.run_review()
        (self.root / "engine/__pycache__").mkdir()
        (self.root / "engine/__pycache__/imported.pyc").write_bytes(b"cache")
        self.assertTrue(self.run_review()["reused"])

    def test_scope_path_and_suffix_contract(self) -> None:
        for directory, suffixes in [
            ("..", [".py"]),
            ("/etc", [".py"]),
            ("review/../engine", [".py"]),
            ("review", []),
            ("review", ["py"]),
            ("review", "*.py"),
        ]:
            self.project["source_scopes"] = [{"directory": directory, "suffixes": suffixes}]
            self.save()
            with (
                self.subTest(directory=directory, suffixes=suffixes),
                self.assertRaises(ValueError),
            ):
                RUN.load_project(self.root)

    def test_empty_source_scope_fails_closed(self) -> None:
        self.add_scope()
        (self.root / "engine/imported.py").unlink()
        with self.assertRaisesRegex(ValueError, "Empty source scope"):
            RUN.load_project(self.root)

    @unittest.skipIf(sys.platform == "win32", "Symlink permission is platform-specific")
    def test_scope_rejects_symlinked_files_and_directories(self) -> None:
        self.add_scope()
        for name, target in [
            ("linked.py", self.root / "proof.py"),
            ("linked", self.root / "review"),
        ]:
            link = self.root / "engine" / name
            link.symlink_to(target, target_is_directory=target.is_dir())
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Symlink"):
                RUN.load_project(self.root)
            link.unlink()

    def test_receipt_command_metadata_is_verified(self) -> None:
        self.run_review()
        receipt = json.loads((self.output / "receipt.json").read_text())
        receipt["checks"][0]["args"] = ["different"]
        (self.output / "receipt.json").write_text(json.dumps(receipt))
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["files"]["receipt.json"] = RUN.digest(self.output / "receipt.json")
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "Command identity"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_rehashed_html_still_must_match_receipt(self) -> None:
        self.run_review()
        (self.output / "index.html").write_text("misleading page")
        manifest = json.loads((self.output / "manifest.json").read_text())
        manifest["files"]["index.html"] = RUN.digest(self.output / "index.html")
        (self.output / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "page does not match"):
            RUN.verify(self.output, self.root, self.project, False)

    def test_immutable_links_require_clean_checkout(self) -> None:
        self.run_review()
        receipt = json.loads((self.output / "receipt.json").read_text())
        receipt["checkout"] = {"commit": "a" * 40, "clean": True}
        self.assertIn("/blob/" + "a" * 40 + "/", RUN.page(self.project, receipt, self.root))
        receipt["checkout"]["clean"] = False
        self.assertNotIn("/blob/" + "a" * 40 + "/", RUN.page(self.project, receipt, self.root))

    def test_source_archive_needs_no_git_metadata(self) -> None:
        self.run_review()
        receipt = json.loads((self.output / "receipt.json").read_text())
        self.assertEqual(receipt["checkout"], {"commit": None, "clean": False})
        self.assertIsNone(receipt["node"])


if __name__ == "__main__":
    unittest.main()
