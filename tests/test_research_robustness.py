"""Prevent data/seed confounding and accidental reuse across temporal studies."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from otto_recsys.research.robustness import cells, load_plan, seed_launch, verify_seed_launch

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
ARCHIVE = "b" * 64


class TestRobustnessProtocol(unittest.TestCase):
    def launch(self, seed: int = 20260909) -> dict:
        return seed_launch(
            ROOT,
            f"reference_seed_{seed}",
            source_commit=COMMIT,
            source_sha256=ARCHIVE,
        )

    def test_all_nine_cells_are_predeclared_and_reference_inputs_are_frozen(self) -> None:
        plan, reference = load_plan(ROOT)
        matrix = cells(plan)
        self.assertEqual(len(matrix), 9)
        self.assertEqual(len({c["cell_id"] for c in matrix}), 9)
        self.assertEqual({c["cohort_seed"] for c in matrix}, {20260908})
        self.assertEqual(len(plan["variants"]), 8)
        self.assertEqual(reference["training"]["seed"], 20260908)

    def test_published_progress_has_current_lineage_and_no_invented_pending_scores(self) -> None:
        from otto_recsys.research.robustness import fingerprint

        plan, _ = load_plan(ROOT)
        progress = json.loads((ROOT / "reports/robustness/progress.json").read_text())
        self.assertEqual(progress["protocol_id"], fingerprint(plan))
        self.assertEqual(
            {c["cell_id"] for c in progress["cells"]}, {c["cell_id"] for c in cells(plan)}
        )
        self.assertEqual(len(progress["cells"]), len(cells(plan)))
        for cell in progress["cells"]:
            if cell["status"] in {"planned", "running"}:
                self.assertIsNone(cell["weighted_recall_at_20"])
        runs = ROOT / "reports/robustness/runs"
        for path in runs.glob("*.launch.json"):
            launch = json.loads(path.read_text())
            if launch.get("task") == "window_study":
                from otto_recsys.research.temporal_robustness import verify_window_launch

                verify_window_launch(ROOT, launch)
            else:
                verify_seed_launch(ROOT, launch)
            receipt = json.loads(path.with_name(path.name.replace(".launch", "")).read_text())
            self.assertEqual(
                receipt["launch_sha256"], hashlib.sha256(path.read_bytes()).hexdigest()
            )
            request = path.with_name(path.name.replace(".launch", ".request"))
            self.assertEqual(
                receipt["request_sha256"], hashlib.sha256(request.read_bytes()).hexdigest()
            )

    def test_model_seed_changes_neither_cohorts_nor_negatives_nor_bootstrap(self) -> None:
        first, second = self.launch(), self.launch(20260910)
        for key in (
            "corpus",
            "retrieval_manifest_sha256",
            "model_inputs_sha256",
            "evaluation_seed",
            "bootstrap_replicates",
            "resources",
            "owner_account",
            "region",
        ):
            self.assertEqual(first[key], second[key], key)
        self.assertNotEqual(first["training"]["seed"], second["training"]["seed"])
        self.assertEqual(
            {k: v for k, v in first["training"].items() if k != "seed"},
            {k: v for k, v in second["training"].items() if k != "seed"},
        )
        self.assertNotEqual(first["checkpoint_uri"], second["checkpoint_uri"])
        self.assertNotIn("/runs/", first["checkpoint_uri"])
        self.assertEqual(verify_seed_launch(ROOT, first), first["robustness"])

    def test_changed_inputs_settings_bootstrap_or_namespace_are_rejected(self) -> None:
        good = self.launch()
        mutations = [
            ("training", "seed", 20260910),
            ("training", "rounds", 401),
            ("resources", "maximum_runtime_seconds", 14400),
            ("robustness", "cohort_seed", 20260910),
        ]
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                changed = copy.deepcopy(good)
                changed[group][key] = value
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_seed_launch(ROOT, changed)
        for key, value in (
            ("evaluation_seed", 20260910),
            ("checkpoint_uri", "s3://bucket/old-study"),
            ("model_inputs_sha256", "c" * 64),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(good)
                changed[key] = value
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_seed_launch(ROOT, changed)
        changed = copy.deepcopy(good)
        changed["corpus"][0]["sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_seed_launch(ROOT, changed)

    def test_verification_preserves_training_identity_and_cannot_change_inputs(self) -> None:
        from otto_recsys.research.robustness import verification_launch, verify_verification_launch

        training = self.launch()
        audit = verification_launch(ROOT, training, source_commit="c" * 40, source_sha256="d" * 64)
        self.assertEqual(audit["training_launch"], training)
        self.assertEqual(audit["checkpoint_uri"], training["checkpoint_uri"])
        self.assertEqual(audit["resources"]["maximum_runtime_seconds"], 1800)
        self.assertEqual(verify_verification_launch(ROOT, audit), training["robustness"])
        for mutate in ("namespace", "corpus", "training", "runtime"):
            with self.subTest(mutate=mutate):
                bad = copy.deepcopy(audit)
                if mutate == "namespace":
                    bad["checkpoint_uri"] += "/different"
                elif mutate == "corpus":
                    bad["corpus"][0]["sha256"] = "e" * 64
                elif mutate == "training":
                    bad["training_launch"]["training"]["rounds"] += 1
                else:
                    bad["resources"]["maximum_runtime_seconds"] = 7200
                with self.assertRaisesRegex(ValueError, "differs"):
                    verify_verification_launch(ROOT, bad)

    def test_reference_caches_cannot_be_used_for_earlier_windows(self) -> None:
        for cell in ("early_seed_20260909", "middle_seed_20260910", "reference_seed_20260908"):
            with self.subTest(cell=cell), self.assertRaisesRegex(ValueError, "own window inputs"):
                seed_launch(ROOT, cell, source_commit=COMMIT, source_sha256=ARCHIVE)

    def test_unknown_cell_and_inexact_source_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not in"):
            seed_launch(ROOT, "reference_seed_4", source_commit=COMMIT, source_sha256=ARCHIVE)
        with self.assertRaisesRegex(ValueError, "exact source"):
            seed_launch(
                ROOT, "reference_seed_20260909", source_commit="main", source_sha256=ARCHIVE
            )

    def test_changed_research_code_and_reference_inventory_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "reports/robustness").mkdir(parents=True)
            shutil.copytree(ROOT / "src/otto_recsys/research", root / "src/otto_recsys/research")
            for path in (
                "configs/research.toml",
                "configs/robustness.toml",
                "reports/robustness/reference_launch.json",
                "pyproject.toml",
                "uv.lock",
            ):
                shutil.copyfile(ROOT / path, root / path)
            training = root / "src/otto_recsys/research/training.py"
            original = training.read_bytes()
            training.write_bytes(original + b"\n# changed\n")
            with self.assertRaisesRegex(ValueError, "implementation changed"):
                load_plan(root)
            training.write_bytes(original)
            reference_path = root / "reports/robustness/reference_launch.json"
            reference = json.loads(reference_path.read_text())
            reference["model_inputs_sha256"] = "c" * 64
            reference_path.write_text(json.dumps(reference))
            with self.assertRaisesRegex(ValueError, "inventory differs"):
                load_plan(root)


if __name__ == "__main__":
    unittest.main()
