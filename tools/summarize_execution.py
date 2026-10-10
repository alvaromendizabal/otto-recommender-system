"""Verify one private owner return and publish an allowlisted execution summary.

Public check: python tools/summarize_execution.py --check
Regenerate: python tools/summarize_execution.py --archive /path/to/owner-return.zip
All members are hashed in memory; none are extracted or executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/latest_execution/summary.json"
RELEASE = ROOT / "reports/submissions/similarity_stack_20260925.json"
ARCHIVE_SHA = "6d472048fa636893fc94f54e74a0426a039344f70155885261c833b347b6cda1"
MANIFEST_SHA = "86b8405c8e6daf8793415e925b3c48b960f682dc3ce9bf82f55be49e4382fa0e"
RUN_ID = "20261010T031015229433Z-70322c6a"
ARCHIVE_LIMIT = 10 * 1024 * 1024
MEMBER_LIMIT = 8 * 1024 * 1024
TOTAL_LIMIT = 20 * 1024 * 1024
MEMBER_COUNT = 111
SOURCE_PINS: dict[str, dict[str, Any]] = {
    "result": {
        "file": "result.json",
        "bytes": 2636,
        "sha256": "5c02604eb0f0b68037124fc49a7f8f9ed87fa3af6565c50d68666f9b6d514f5f",
    },
    "ledger": {
        "file": "ledger_entry.json",
        "bytes": 2637,
        "sha256": "124c9133faa721eaada7e629a4b2728b8c4daddd355aa7a16f1aacce3e14ed1b",
    },
    "ranking": {
        "file": "support_ranking.json",
        "bytes": 50496,
        "sha256": "58dce8d57f061f8dad33367a5d03109dc9eb48ea2f5ad4972b114b782012d2db",
    },
    "confirmation": {
        "file": "support_confirmation.json",
        "bytes": 780,
        "sha256": "54a256d3a4f42bd229a5240d66645a1c4e165ca5d27ced82508acc8ca8401b7d",
    },
    "submission": {
        "file": "support_submission.json",
        "bytes": 778,
        "sha256": "27199d3263f7827a441c90ddd9093bf8c1626606c16997d3a6e72ce9529e2d97",
    },
    "resume": {
        "file": "support_ranking/completed_model_resume.json",
        "bytes": 439,
        "sha256": "c2fd1b8b0fa4edb9542f81d2d3e60d516e4f3bff3ca9e7fdd20a7f565dd10013",
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("EXECUTION_SUMMARY: " + message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def read_json(raw: bytes) -> dict[str, Any]:
    value = json.loads(raw, object_pairs_hook=unique_object)
    require(isinstance(value, dict), "expected a JSON object")
    return dict(value)


def safe_path(name: str) -> None:
    require(isinstance(name, str) and bool(name), "invalid archive path")
    require(not PurePosixPath(name).is_absolute(), "unsafe archive path")
    require(not any(char in name for char in ("\\", "\x00", ":")), "unsafe archive path")
    require(all(part not in ("", ".", "..") for part in name.split("/")), "unsafe archive path")


def inspect_archive(
    path: Path, expected_sha: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    require(path.stat().st_size <= ARCHIVE_LIMIT, "archive size limit exceeded")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    require(digest.hexdigest() == expected_sha, "archive SHA-256 mismatch")
    kept: dict[str, dict[str, Any]] = {}
    selected = {pin["file"]: role for role, pin in SOURCE_PINS.items()}
    source_hashes = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        require(len(members) == MEMBER_COUNT + 1, "archive member count mismatch")
        names = [member.filename for member in members]
        require(len({name.casefold() for name in names}) == len(names), "duplicate ZIP member")
        for member in members:
            safe_path(member.filename)
            require(member.orig_filename == member.filename, "unsafe original archive path")
            require(not member.is_dir(), "directory member")
            require(not member.flag_bits & 1, "encrypted member")
            require(stat.S_IFMT(member.external_attr >> 16) in (0, stat.S_IFREG), "linked member")
            require(member.file_size <= MEMBER_LIMIT, "member size limit exceeded")
        require(sum(member.file_size for member in members) <= TOTAL_LIMIT, "total size limit")
        require("return_manifest.json" in names, "missing manifest")
        require(archive.getinfo("return_manifest.json").file_size <= 1024 * 1024, "manifest size")
        manifest_raw = archive.read("return_manifest.json")
        manifest = read_json(manifest_raw)
        require(
            set(manifest) == set(names) - {"return_manifest.json"}, "manifest coverage mismatch"
        )
        total_read = 0
        for name, receipt in manifest.items():
            safe_path(name)
            require(type(receipt["bytes"]) is int and receipt["bytes"] >= 0, "invalid member size")
            require(archive.getinfo(name).file_size == receipt["bytes"], "member size mismatch")
            member_hash = hashlib.sha256()
            size = 0
            retained = bytearray()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    size += len(block)
                    total_read += len(block)
                    require(
                        size <= MEMBER_LIMIT and total_read <= TOTAL_LIMIT, "decompression bound"
                    )
                    member_hash.update(block)
                    if name in selected:
                        require(size <= 1024 * 1024, "aggregate JSON size limit")
                        retained.extend(block)
            require(size == receipt["bytes"], "member size mismatch")
            require(member_hash.hexdigest() == receipt["sha256"], "member SHA-256 mismatch")
            if name in selected:
                role = selected[name]
                kept[role] = read_json(bytes(retained))
                source_hashes[role] = {"bytes": size, "sha256": member_hash.hexdigest()}
        require(set(kept) == set(SOURCE_PINS), "missing aggregate receipt")
    return kept, {
        "source_zip_sha256": expected_sha,
        "manifest_sha256": sha256(manifest_raw),
        "verified_manifest_members": len(manifest),
        "zip_members": len(names),
        "only_unmanifested_member": "return_manifest.json",
        "aggregate_sources": source_hashes,
    }


def accepted_release(path: Path = RELEASE) -> dict[str, Any]:
    release = read_json(path.read_bytes())
    require(
        release["submission_ref"] == 56542128
        and release["private_score"] == 0.57586
        and release["public_score"] == 0.57601,
        "existing scored release mismatch",
    )
    return {
        "submission_ref": release["submission_ref"],
        "private_score": release["private_score"],
        "public_score": release["public_score"],
        "evidence_source": "reports/submissions/similarity_stack_20260925.json",
        "independently_reverified_by_this_return": False,
    }


def public_result(
    result: dict[str, Any], ranking: dict[str, Any], provenance: dict[str, Any]
) -> dict[str, Any]:
    comparisons = ranking["summary"]["comparisons"]
    return {
        "schema_version": 1,
        "scope": "latest_inspected_owner_return",
        "version": "v27",
        "run_id": result["run_id"],
        "recorded_ended_utc": None,
        "execution_status": result["status"],
        "scientific_experiment_completed": result["scientific_experiment_completed"],
        "scientific_outcome": result["ranking_status"],
        "completed_checkpoint_models": result["fit_accounting"]["completed_ranker_fits"],
        "new_fits_this_invocation": result["fit_accounting"]["new_fits_this_invocation"],
        "selection": {
            "sessions": ranking["summary"]["sessions"],
            "previously_exposed": ranking["selection_previously_exposed"],
            "independent_confirmation": ranking["independent_confirmation"],
            "reported_weighted_delta_vs_incumbent": comparisons["joint_vs_champion"][
                "weighted_gain"
            ],
            "reported_weighted_delta_vs_matched_control": comparisons["joint_vs_independent"][
                "weighted_gain"
            ],
            "descriptive_95_interval_vs_matched_control": comparisons["joint_vs_independent"][
                "descriptive_bootstrap_95_interval"
            ],
            "frozen_gate_passed": ranking["gate"]["passed"],
            "failed_checks": [
                "minimum_gain_vs_matched_control",
                "positive_lower_bound_vs_matched_control",
            ],
        },
        "candidate_qualified": result["candidate_qualified"],
        "confirmation_status": result["confirmation_status"],
        "submission_status": result["submission_status"],
        "submission_created": result["submission_created"],
        "resumable_pending": result["resumable_pending"],
        "existing_scored_release": accepted_release(),
        "provenance": provenance,
        "verification_scope": {
            "included_bytes_verified": True,
            "private_models_executed": False,
            "private_predictions_independently_replayed": False,
            "archive_members_extracted": False,
            "live_aws_state_verified_by_reproducer": False,
        },
        "limitations": [
            "Successful execution completed a negative experiment, not a promoted release.",
            "Selection was exposed; its bootstrap interval is descriptive, not confirmation.",
            "Model checkpoints were reused; this invocation performed zero new fits.",
            "Prior competition scores retain their earlier source; this run made no submission.",
            "The return records no end timestamp. The run ID is not a completion timestamp.",
            "Latest inspected owner return does not establish globally latest cloud or live state.",
        ],
    }


def summarize_archive(path: Path, expected_sha: str = ARCHIVE_SHA) -> dict[str, Any]:
    documents, provenance = inspect_archive(path, expected_sha)
    result, ranking, ledger = (documents[name] for name in ("result", "ranking", "ledger"))
    require(result["run_id"] == RUN_ID and result["status"] == "SUCCESS", "unexpected run outcome")
    require(
        result["scientific_experiment_completed"] is True
        and result["ranking_status"] == ranking["status"] == "CLOSED_SUPPORT_RANKER_NEGATIVE"
        and ranking["scientific_negative"] is True,
        "scientific outcome disagreement",
    )
    require(
        ledger["run_id"] == RUN_ID
        and ledger["status"] == "SUCCESS"
        and ledger["scientific_experiment_completed"] is True
        and ledger["new_completed_scientific_experiments_this_run"] == 1,
        "ledger disagreement",
    )
    accounting = result["fit_accounting"]
    require(
        accounting["completed_ranker_fits"] == 2
        and accounting["new_fits_this_invocation"] == ranking["new_fits_this_invocation"] == 0
        and accounting["closed_models_refitted"] is False
        and len(accounting["verified_complete_checkpoint_models"]) == 2,
        "checkpoint accounting disagreement",
    )
    require(
        documents["resume"]["status"] == "READY" and documents["resume"]["model_fits"] == 0,
        "checkpoint resume disagreement",
    )
    for document in (result, ranking, documents["confirmation"], documents["submission"]):
        require(
            document["candidate_qualified"] is False
            and document["independent_confirmation"] is False
            and document["submission_created"] is False,
            "unexpected promotion or submission",
        )
    require(
        result["resumable_pending"] is False and result["pending_stage"] is None, "pending work"
    )
    for role in ("confirmation", "submission"):
        require(
            documents[role]["status"] == result[role + "_status"] == "SKIPPED_SCIENTIFIC_GATE"
            and documents[role]["experiment_complete"] is False,
            "skipped stage disagreement",
        )
    require(
        ranking["selection_previously_exposed"] is True
        and ranking["summary"]["sessions"] == 20000
        and ranking["gate"]["passed"] is False,
        "selection scope disagreement",
    )
    failed = {name for name, passed in ranking["gate"]["checks"].items() if passed is False}
    require(
        failed == {"gain_vs_independent", "positive_independent_lower_bound"}, "gate disagreement"
    )
    summary = ranking["summary"]
    for comparison, reference in (
        ("joint_vs_champion", "champion"),
        ("joint_vs_independent", "independent"),
    ):
        reported = summary["comparisons"][comparison]
        gain = sum(
            weight
            * (summary["ranked_hits"]["joint"][task] - summary["ranked_hits"][reference][task])
            / summary["denominators"][task]
            for task, weight in (("clicks", 0.1), ("carts", 0.3))
        )
        require(
            math.isclose(gain, reported["weighted_gain"], rel_tol=0, abs_tol=1e-15),
            "metric arithmetic",
        )
        require(reported["independent_confirmation"] is False, "comparison scope disagreement")
    return public_result(result, ranking, provenance)


def expected_snapshot() -> dict[str, Any]:
    result = {
        "run_id": RUN_ID,
        "status": "SUCCESS",
        "scientific_experiment_completed": True,
        "ranking_status": "CLOSED_SUPPORT_RANKER_NEGATIVE",
        "fit_accounting": {"completed_ranker_fits": 2, "new_fits_this_invocation": 0},
        "candidate_qualified": False,
        "confirmation_status": "SKIPPED_SCIENTIFIC_GATE",
        "submission_status": "SKIPPED_SCIENTIFIC_GATE",
        "submission_created": False,
        "resumable_pending": False,
    }
    ranking = {
        "summary": {
            "sessions": 20000,
            "comparisons": {
                "joint_vs_champion": {"weighted_gain": 0.006525037936267071},
                "joint_vs_independent": {
                    "weighted_gain": 0.0008598887202832574,
                    "descriptive_bootstrap_95_interval": [
                        -0.0004489333723377422,
                        0.0021930711032684,
                    ],
                },
            },
        },
        "selection_previously_exposed": True,
        "independent_confirmation": False,
        "gate": {"passed": False},
    }
    return public_result(
        result,
        ranking,
        {
            "source_zip_sha256": ARCHIVE_SHA,
            "manifest_sha256": MANIFEST_SHA,
            "verified_manifest_members": MEMBER_COUNT,
            "zip_members": MEMBER_COUNT + 1,
            "only_unmanifested_member": "return_manifest.json",
            "aggregate_sources": {
                role: {"bytes": pin["bytes"], "sha256": pin["sha256"]}
                for role, pin in SOURCE_PINS.items()
            },
        },
    )


def validate_snapshot(value: dict[str, Any]) -> None:
    require(
        json.dumps(value, sort_keys=True, allow_nan=False)
        == json.dumps(expected_snapshot(), sort_keys=True, allow_nan=False),
        "public summary contract mismatch",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--archive", type=Path)
    action.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.check:
        validate_snapshot(read_json(args.output.read_bytes()))
        print("PASS: v27 aggregate receipt, fixed provenance and existing release consistency")
    else:
        result = summarize_archive(args.archive)
        validate_snapshot(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print("PASS: authenticated return and metrics; published allowlisted aggregates only")


if __name__ == "__main__":
    main()
