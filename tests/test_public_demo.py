"""The synthetic demo is runnable and verifiable without project dependencies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from html.parser import HTMLParser
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from otto_recsys import public_demo as demo  # noqa: E402
from otto_recsys.public_demo_report import render_report  # noqa: E402

ACTIONS = ("clicks", "carts", "orders")


class ReportMarkup(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.elements: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.elements.append((tag, dict(attrs)))


class PublicDemoTests(unittest.TestCase):
    """Exercise contracts with synthetic data, including deliberately invalid inputs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = demo.generate_synthetic(seed=42)
        cls.cutoff = min(query.query_ts for query in cls.data.queries)
        cls.model = demo.fit_history(cls.data.history, cls.cutoff)

    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def _seal(self, payload: dict) -> tuple[Path, str]:
        path = self.directory / "predictions.json"
        digest = demo.seal_predictions(payload, path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        return path, digest

    def _snapshot(self, output: Path) -> dict[str, bytes]:
        return {
            str(path.relative_to(output)): path.read_bytes()
            for path in output.rglob("*")
            if path.is_file()
        }

    def test_synthetic_generation_is_seeded_and_temporally_partitioned(self) -> None:
        self.assertEqual(self.data, demo.generate_synthetic(seed=42))
        self.assertNotEqual(self.data, demo.generate_synthetic(seed=43))
        self.assertTrue(self.data.history)
        self.assertTrue(self.data.queries)
        self.assertTrue(self.data.targets)
        self.assertTrue(all(event.ts < self.cutoff for event in self.data.history))
        queries = {query.session: query for query in self.data.queries}
        self.assertEqual(len(queries), len(self.data.queries))
        for query in self.data.queries:
            self.assertTrue(query.prefix)
            self.assertTrue(all(event.session == query.session for event in query.prefix))
            self.assertTrue(all(event.ts < query.query_ts for event in query.prefix))
        for event in self.data.targets:
            self.assertGreater(event.ts, queries[event.session].query_ts)

    def test_fitting_rejects_events_at_and_after_the_history_cutoff(self) -> None:
        for timestamp in (self.cutoff, self.cutoff + 1):
            with self.subTest(timestamp=timestamp):
                future = replace(self.data.history[0], ts=timestamp)
                with self.assertRaises(ValueError):
                    demo.fit_history((*self.data.history, future), self.cutoff)

    def test_fitting_accepts_the_last_event_before_the_cutoff(self) -> None:
        last_allowed = replace(self.data.history[0], ts=self.cutoff - 1)
        model = demo.fit_history((*self.data.history, last_allowed), self.cutoff)
        self.assertEqual(model.cutoff, self.cutoff)

    def test_prediction_rejects_prefix_events_at_or_after_query_time(self) -> None:
        query = self.data.queries[0]
        for timestamp in (query.query_ts, query.query_ts + 1):
            with self.subTest(timestamp=timestamp):
                future = replace(query.prefix[-1], ts=timestamp)
                invalid = replace(query, prefix=(*query.prefix, future))
                with self.assertRaises(ValueError):
                    demo.predict(self.model, (invalid,))

    def test_prediction_rejects_a_query_before_model_availability(self) -> None:
        query = replace(self.data.queries[0], query_ts=self.cutoff - 1)
        with self.assertRaises(ValueError):
            demo.predict(self.model, (query,))

    def test_prediction_covers_every_session_and_objective_with_unique_items(self) -> None:
        predictions = demo.predict(self.model, self.data.queries)
        rows = predictions["rows"]
        self.assertEqual(len(rows), len(self.data.queries))
        self.assertEqual(
            {row["session"] for row in rows}, {query.session for query in self.data.queries}
        )
        for row in rows:
            self.assertEqual(set(row["objectives"]), set(ACTIONS))
            for objective in row["objectives"].values():
                ranked = objective["recommendations"]
                candidates = objective["candidates"]
                self.assertEqual(len(ranked), 20)
                self.assertEqual(len(ranked), len(set(ranked)))
                self.assertEqual(len(candidates), len(set(candidates)))
                self.assertTrue(set(ranked).issubset(candidates))
                self.assertEqual(
                    [item["item"] for item in objective["explanations"]], ranked
                )

    def test_prediction_rejects_duplicate_session_rows(self) -> None:
        query = self.data.queries[0]
        with self.assertRaises(ValueError):
            demo.predict(self.model, (query, query))

    def test_equal_scores_use_item_id_order(self) -> None:
        catalog = tuple(range(1, 31))
        model = demo.DemoModel(
            cutoff=100,
            catalog=catalog,
            popularity={action: {} for action in ACTIONS},
            cooccurrence={},
        )
        query = demo.Query(7, 101, (demo.Event(7, 30, 100, "clicks"),))
        row = demo.predict(model, (query,))["rows"][0]
        for objective in row["objectives"].values():
            explanations = objective["explanations"]
            for left, right in pairwise(explanations):
                if left["score"] == right["score"]:
                    self.assertLess(left["item"], right["item"])
            tied = [item for item in objective["recommendations"] if item != 30]
            self.assertGreater(len(tied), 1)
            self.assertEqual(tied, sorted(tied))

    def test_prediction_replay_is_byte_deterministic(self) -> None:
        first = demo.predict(self.model, self.data.queries)
        second = demo.predict(
            demo.fit_history(self.data.history, self.cutoff), self.data.queries
        )
        self.assertEqual(demo.canonical_json(first), demo.canonical_json(second))

    def test_scoring_rejects_modified_sealed_predictions(self) -> None:
        predictions = demo.predict(self.model, self.data.queries)
        path, digest = self._seal(predictions)
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            demo.evaluate_predictions(path, self.data.targets, digest)

    def test_pooled_metric_separates_retrieval_coverage_from_ranked_hits(self) -> None:
        predictions = {
            "schema_version": 1,
            "k": 20,
            "candidate_limit": 21,
            "rows": [{
                "session": 7,
                "query_ts": 100,
                "objectives": {
                    action: {
                        "recommendations": list(range(1, 21)),
                        "candidates": list(range(1, 22)),
                    }
                    for action in ACTIONS
                },
            }],
        }
        targets = (
            demo.Event(7, 21, 101, "clicks"),
            demo.Event(7, 22, 101, "carts"),
            demo.Event(7, 1, 101, "orders"),
        )
        path, digest = self._seal(predictions)
        metrics = demo.evaluate_predictions(path, targets, digest)
        self.assertAlmostEqual(metrics["weighted_recall_at_20"], 0.6)
        self.assertAlmostEqual(metrics["weighted_candidate_oracle"], 0.7)
        for action, expected_hits, expected_pool in (
            ("clicks", 0, 1), ("carts", 0, 0), ("orders", 1, 1)
        ):
            objective = metrics["objectives"][action]
            self.assertEqual(objective["hits"], expected_hits)
            self.assertEqual(objective["candidate_hits"], expected_pool)
            self.assertEqual(objective["denominator"], 1)

    def test_evaluation_rejects_future_targets_at_query_boundary(self) -> None:
        predictions = demo.predict(self.model, self.data.queries)
        path, digest = self._seal(predictions)
        queries = {query.session: query for query in self.data.queries}
        target = self.data.targets[0]
        invalid = replace(target, ts=queries[target.session].query_ts)
        with self.assertRaises(ValueError):
            demo.evaluate_predictions(path, (invalid, *self.data.targets[1:]), digest)

    def test_evaluation_rejects_missing_prediction_session_coverage(self) -> None:
        predictions = demo.predict(self.model, self.data.queries)
        missing_session = self.data.targets[0].session
        predictions["rows"] = [
            row for row in predictions["rows"] if row["session"] != missing_session
        ]
        path, digest = self._seal(predictions)
        with self.assertRaises(ValueError):
            demo.evaluate_predictions(path, self.data.targets, digest)

    def test_completed_replay_preserves_every_artifact_byte(self) -> None:
        output = self.directory / "demo"
        first = demo.run_demo(output, seed=42)
        initial_bytes = self._snapshot(output)
        self.assertTrue(
            {"predictions.json", "result.json", "report.html", "manifest.json"}
            <= initial_bytes.keys()
        )
        second = demo.run_demo(output, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(initial_bytes, self._snapshot(output))
        self.assertEqual(first, demo.verify_output(output, seed=42))

    def test_interrupted_run_resumes_from_the_existing_prediction_seal(self) -> None:
        output = self.directory / "interrupted"
        with (
            patch.object(demo, "evaluate_predictions", side_effect=RuntimeError("interrupted")),
            self.assertRaises(RuntimeError),
        ):
            demo.run_demo(output)
        predictions = (output / "predictions.json").read_bytes()
        manifest = json.loads((output / "manifest.json").read_bytes())
        self.assertEqual(manifest["status"], "started")
        resumed = demo.run_demo(output)
        self.assertEqual((output / "predictions.json").read_bytes(), predictions)
        self.assertEqual(resumed, demo.run_demo(self.directory / "fresh"))
        self.assertEqual(resumed, demo.verify_output(output))

    def test_replay_rejects_a_changed_renderer_identity(self) -> None:
        output = self.directory / "demo"
        demo.run_demo(output)
        path = output / "manifest.json"
        manifest = json.loads(path.read_bytes())
        manifest["renderer_sha256"] = "0" * 64
        path.write_bytes(demo.canonical_json(manifest))
        modified_bytes = self._snapshot(output)
        with self.assertRaisesRegex(ValueError, "renderer_sha256"):
            demo.verify_output(output)
        with self.assertRaises(ValueError):
            demo.run_demo(output)
        self.assertEqual(modified_bytes, self._snapshot(output))

    def test_new_output_directory_reproduces_result_predictions_and_report(self) -> None:
        first = self.directory / "first"
        second = self.directory / "second"
        self.assertEqual(demo.run_demo(first, seed=42), demo.run_demo(second, seed=42))
        for name in ("predictions.json", "result.json", "report.html"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_resume_rejects_a_different_seed_without_modifying_artifacts(self) -> None:
        output = self.directory / "demo"
        demo.run_demo(output, seed=42)
        initial_bytes = self._snapshot(output)
        with self.assertRaises(ValueError):
            demo.run_demo(output, seed=43)
        self.assertEqual(initial_bytes, self._snapshot(output))

    def test_replay_rejects_modified_outputs_without_overwriting_them(self) -> None:
        for name in ("predictions.json", "result.json", "report.html"):
            with self.subTest(name=name):
                output = self.directory / name.replace(".", "_")
                demo.run_demo(output, seed=42)
                path = output / name
                path.write_bytes(path.read_bytes() + b"tampered")
                modified_bytes = self._snapshot(output)
                with self.assertRaises(ValueError):
                    demo.verify_output(output)
                with self.assertRaises(ValueError):
                    demo.run_demo(output, seed=42)
                self.assertEqual(modified_bytes, self._snapshot(output))

    def test_foreign_directory_is_not_overwritten(self) -> None:
        note = self.directory / "my_file.txt"
        note.write_text("Keep this unrelated work.", encoding="utf-8")
        with self.assertRaises(ValueError):
            demo.run_demo(self.directory)
        self.assertEqual(note.read_text(encoding="utf-8"), "Keep this unrelated work.")
        self.assertEqual(list(self.directory.iterdir()), [note])

    def test_report_is_offline_and_keeps_future_targets_in_closed_details(self) -> None:
        output = self.directory / "demo"
        result = demo.run_demo(output)
        report = (output / "report.html").read_text(encoding="utf-8")
        self.assertIn("All item IDs, sessions and results on this page are synthetic", report)
        self.assertIn("published temporal study and scored release", report)
        self.assertIn(f'{result["metrics"]["weighted_recall_at_20"]:.4f}', report)
        markup = ReportMarkup()
        markup.feed(report)
        panels = [attrs for _, attrs in markup.elements if "data-demo-panel" in attrs]
        self.assertEqual(len(panels), len(result["examples"]) * len(ACTIONS))
        details = [
            attrs for tag, attrs in markup.elements
            if tag == "details" and attrs.get("class") == "evaluation"
        ]
        self.assertEqual(len(details), len(panels))
        self.assertTrue(all("open" not in attrs for attrs in details))
        for _, attrs in markup.elements:
            for key in ("src", "href"):
                value = attrs.get(key) or ""
                self.assertFalse(value.startswith(("http://", "https://", "//")), value)

    def test_report_escapes_prediction_evidence_and_scope(self) -> None:
        result = demo.run_demo(self.directory / "demo")
        injected = '<img src=x onerror="alert(1)">'
        result["scope"] = injected
        explanation = result["examples"][0]["explanations"]["clicks"][0]
        explanation["sources"] = [injected]
        explanation["reason"] = injected
        explanation["features"] = {injected: 1.0}
        rendered = render_report(result)
        self.assertNotIn(injected, rendered)
        self.assertIn("&lt;img src=x onerror=&quot;alert(1)&quot;&gt;", rendered)

    def test_report_rejects_real_data_claims_and_nonfinite_metrics(self) -> None:
        result = demo.run_demo(self.directory / "demo")
        for field, value in (("synthetic", False), ("schema_version", 2)):
            modified = {**result, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                render_report(modified)
        result["metrics"]["weighted_recall_at_20"] = float("nan")
        with self.assertRaises(ValueError):
            render_report(result)

    def test_cli_works_without_site_packages_from_another_directory(self) -> None:
        output = self.directory / "demo"
        command = [
            sys.executable,
            "-S",
            "-O",
            str(ROOT / "scripts/run_public_demo.py"),
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        receipts = []
        for extra in ([], [], ["--check"]):
            process = subprocess.run(
                [*command, *extra],
                cwd=self.directory,
                env=environment,
                check=False,
                text=True,
                capture_output=True,
                timeout=15,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            receipts.append(json.loads(process.stdout))
        self.assertFalse(receipts[0]["reused"])
        self.assertTrue(receipts[1]["reused"])
        self.assertTrue(receipts[2]["reused"])
        self.assertEqual(receipts[2]["status"], "passed")
        self.assertTrue(receipts[0]["synthetic"])


if __name__ == "__main__":
    unittest.main()
