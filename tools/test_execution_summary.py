"""Offline integrity, scientific-boundary and disclosure tests for the public receipt."""

from __future__ import annotations

import copy
import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import summarize_execution as summary


def documents() -> dict[str, dict[str, Any]]:
    result = {
        "run_id": summary.RUN_ID,
        "status": "SUCCESS",
        "scientific_experiment_completed": True,
        "ranking_status": "CLOSED_SUPPORT_RANKER_NEGATIVE",
        "candidate_qualified": False,
        "independent_confirmation": False,
        "submission_created": False,
        "resumable_pending": False,
        "pending_stage": None,
        "confirmation_status": "SKIPPED_SCIENTIFIC_GATE",
        "submission_status": "SKIPPED_SCIENTIFIC_GATE",
        "fit_accounting": {
            "completed_ranker_fits": 2,
            "new_fits_this_invocation": 0,
            "closed_models_refitted": False,
            "verified_complete_checkpoint_models": [{}, {}],
        },
        "private_configuration": "PRIVATE_RESULT_DO_NOT_PUBLISH",
    }
    ranking = {
        "status": "CLOSED_SUPPORT_RANKER_NEGATIVE",
        "scientific_negative": True,
        "new_fits_this_invocation": 0,
        "candidate_qualified": False,
        "independent_confirmation": False,
        "submission_created": False,
        "selection_previously_exposed": True,
        "gate": {
            "passed": False,
            "checks": {
                "gain_vs_independent": False,
                "positive_independent_lower_bound": False,
                "other_frozen_check": True,
            },
        },
        "summary": {
            "sessions": 20000,
            "denominators": {"clicks": 19219, "carts": 5931},
            "ranked_hits": {
                "champion": {"clicks": 10311, "carts": 2709},
                "independent": {"clicks": 10311, "carts": 2821},
                "joint": {"clicks": 10311, "carts": 2838},
            },
            "comparisons": {
                "joint_vs_champion": {
                    "weighted_gain": 0.006525037936267071,
                    "independent_confirmation": False,
                },
                "joint_vs_independent": {
                    "weighted_gain": 0.0008598887202832574,
                    "descriptive_bootstrap_95_interval": [
                        -0.0004489333723377422,
                        0.0021930711032684,
                    ],
                    "independent_confirmation": False,
                },
            },
        },
        "private_weights": "PRIVATE_WEIGHTS_DO_NOT_PUBLISH",
    }
    skipped = {
        "status": "SKIPPED_SCIENTIFIC_GATE",
        "candidate_qualified": False,
        "independent_confirmation": False,
        "submission_created": False,
        "experiment_complete": False,
    }
    return {
        "result": result,
        "ranking": ranking,
        "ledger": {
            "run_id": summary.RUN_ID,
            "status": "SUCCESS",
            "scientific_experiment_completed": True,
            "new_completed_scientific_experiments_this_run": 1,
        },
        "resume": {"status": "READY", "model_fits": 0},
        "confirmation": dict(skipped),
        "submission": dict(skipped),
    }


def write_archive(
    path: Path,
    *,
    missing_manifest: bool = False,
    corrupt_member: bool = False,
    duplicate: bool = False,
    unsafe: bool = False,
    linked: bool = False,
) -> str:
    content = {
        summary.SOURCE_PINS[role]["file"]: json.dumps(value).encode()
        for role, value in documents().items()
    }
    for index in range(summary.MEMBER_COUNT - len(content)):
        content[f"unused/{index}.txt"] = b"PRIVATE_UNUSED_DO_NOT_PUBLISH"
    if unsafe:
        content["../outside.txt"] = content.pop("unused/0.txt")
    manifest = {
        name: {"bytes": len(raw), "sha256": summary.sha256(raw)} for name, raw in content.items()
    }
    if corrupt_member:
        manifest["result.json"]["sha256"] = "0" * 64
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in content.items():
            if linked and name == "unused/0.txt":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, raw)
            else:
                archive.writestr(name, raw)
        if not missing_manifest:
            archive.writestr("return_manifest.json", json.dumps(manifest))
        if duplicate:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("result.json", content["result.json"])
    return summary.sha256(path.read_bytes())


class ExecutionSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "owner.zip"

    def test_allowlist_and_negative_experiment_scope(self) -> None:
        digest = write_archive(self.path)
        with patch.object(zipfile.ZipFile, "extract", side_effect=AssertionError("no extraction")):
            result = summary.summarize_archive(self.path, digest)
        self.assertNotIn("PRIVATE_", json.dumps(result))
        self.assertEqual(result["provenance"]["verified_manifest_members"], 111)
        self.assertEqual(result["execution_status"], "SUCCESS")
        self.assertEqual(result["scientific_outcome"], "CLOSED_SUPPORT_RANKER_NEGATIVE")
        self.assertEqual(result["new_fits_this_invocation"], 0)
        self.assertFalse(result["submission_created"])
        self.assertFalse(result["selection"]["independent_confirmation"])
        self.assertFalse(
            result["existing_scored_release"]["independently_reverified_by_this_return"]
        )

    def test_outer_and_member_hash_tampering(self) -> None:
        digest = write_archive(self.path)
        self.path.write_bytes(self.path.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "archive SHA-256"):
            summary.inspect_archive(self.path, digest)
        digest = write_archive(self.path, corrupt_member=True)
        with self.assertRaisesRegex(ValueError, "member SHA-256"):
            summary.inspect_archive(self.path, digest)

    def test_missing_manifest_duplicate_and_unsafe_members(self) -> None:
        for options in (
            {"missing_manifest": True},
            {"duplicate": True},
            {"unsafe": True},
            {"linked": True},
        ):
            with self.subTest(options=options):
                digest = write_archive(self.path, **options)
                with self.assertRaises(ValueError):
                    summary.inspect_archive(self.path, digest)
        for name in ("/absolute", "a\\b", "a//b", "a/./b", "C:drive"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                summary.safe_path(name)

    def test_archive_member_and_total_bounds(self) -> None:
        digest = write_archive(self.path)
        for limit in ("ARCHIVE_LIMIT", "MEMBER_LIMIT", "TOTAL_LIMIT"):
            with (
                self.subTest(limit=limit),
                patch.object(summary, limit, 1),
                self.assertRaises(ValueError),
            ):
                summary.inspect_archive(self.path, digest)

    def test_promotion_confirmation_and_fit_accounting_rejected(self) -> None:
        for change in ("promotion", "confirmation", "fits", "pending"):
            source = documents()
            if change == "promotion":
                source["result"]["candidate_qualified"] = True
            elif change == "confirmation":
                source["ranking"]["independent_confirmation"] = True
            elif change == "fits":
                source["result"]["fit_accounting"]["new_fits_this_invocation"] = 1
            else:
                source["result"]["pending_stage"] = "unexpected"
            with (
                self.subTest(change=change),
                patch.object(summary, "inspect_archive", return_value=(source, {})),
                self.assertRaises(ValueError),
            ):
                summary.summarize_archive(self.path)

    def test_metric_denominators_and_gate_decisions_are_checked(self) -> None:
        for change in ("denominator", "hits", "gate"):
            source = copy.deepcopy(documents())
            if change == "denominator":
                source["ranking"]["summary"]["denominators"]["carts"] += 1
            elif change == "hits":
                source["ranking"]["summary"]["ranked_hits"]["joint"]["carts"] += 1
            else:
                source["ranking"]["gate"]["checks"]["gain_vs_independent"] = True
            with (
                self.subTest(change=change),
                patch.object(summary, "inspect_archive", return_value=(source, {})),
                self.assertRaises(ValueError),
            ):
                summary.summarize_archive(self.path)

    def test_public_snapshot_rejects_extra_private_fields_and_scope_changes(self) -> None:
        summary.validate_snapshot(summary.read_json(summary.OUTPUT.read_bytes()))
        for field, value in (
            ("private_config", "secret"),
            ("schema_version", True),
            ("submission_created", True),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                summary.validate_snapshot(summary.expected_snapshot() | {field: value})
        altered = summary.expected_snapshot()
        altered["provenance"]["manifest_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            summary.validate_snapshot(altered)

    def test_duplicate_json_and_prior_score_disagreement(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON"):
            summary.read_json(b'{"status":"SUCCESS","status":"FAILED"}')
        path = Path(self.temp.name) / "release.json"
        path.write_text(
            json.dumps({"submission_ref": 56542128, "private_score": 0.99, "public_score": 0.57601})
        )
        with self.assertRaisesRegex(ValueError, "scored release mismatch"):
            summary.accepted_release(path)


if __name__ == "__main__":
    unittest.main()
