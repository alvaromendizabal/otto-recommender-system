"""Independently check retained release receipts and controlled-study arithmetic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

RELEASE = "reports/submissions/similarity_stack_20260925.json"
SCOPE = "reports/release_scope.json"
MANIFEST = "reports/research/manifest.json"
EVALUATION = "reports/research/evaluation.json"
AUDIT = "reports/research/audit.json"
WEIGHTS = {"clicks": 0.1, "carts": 0.3, "orders": 0.6}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _count(value: Any, name: str, *, positive: bool = False) -> int:
    _require(type(value) is int and value >= int(positive), f"Invalid count: {name}")
    return int(value)


def _number(value: Any, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"Invalid number: {name}")
    return float(value)


def _equal_number(left: Any, right: Any, name: str) -> None:
    _require(
        math.isclose(_number(left, name), _number(right, name), rel_tol=0, abs_tol=1e-12),
        f"Inconsistent {name}",
    )


def review_snapshot(root: Path) -> dict[str, Any]:
    """Recompute stored evidence; do not load credentials, models or external data."""
    root = root.resolve()
    paths = (RELEASE, SCOPE, MANIFEST, EVALUATION, AUDIT)
    records = {name: json.loads((root / name).read_text()) for name in paths}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    release, scope, manifest, evaluation, audit = (records[p] for p in paths)
    _require(scope["receipt_sha256"] == hashes[RELEASE], "Release receipt checksum mismatch")
    _require(
        release["evaluation_setting"] == "post-competition Kaggle scoring",
        "Release must retain post-competition scope",
    )
    reference = _count(release["submission_ref"], "submission reference", positive=True)
    _require(reference == scope["submission_reference"], "Inconsistent submission reference")
    for split in ("private", "public"):
        value = _number(release[f"{split}_score"], split)
        _require(0 <= value <= 1, "Release score outside [0, 1]")
        _equal_number(value, scope[f"{split}_score"], f"{split} release score")
        _equal_number(
            value - release[f"previous_{split}_score"],
            release[f"delta_{split}"],
            f"{split} release delta",
        )
    identity = release["sha256"]
    _require(
        identity == scope["release_sha256"]
        and len(identity) == 64
        and all(c in "0123456789abcdef" for c in identity),
        "Invalid prediction identity",
    )
    sessions = _count(release["sessions"], "release sessions", positive=True)
    rows = _count(release["rows"], "release rows", positive=True)
    _require(
        rows == 3 * sessions and sessions == scope["sessions"] and rows == scope["rows"],
        "Incomplete release coverage",
    )
    _require(
        scope["exact_private_release_reproduction"] is False,
        "Reference must not claim exact private-release reproduction",
    )
    _require(
        manifest["status"] == evaluation["status"] == audit["status"] == "passed",
        "Controlled study is not accepted",
    )
    _require(
        manifest["seal_id"] == evaluation["seal_id"] == audit["seal_id"],
        "Controlled study model identities disagree",
    )
    for name, expected in manifest["files"].items():
        path = (root / "reports/research" / name).resolve()
        _require(
            path.is_relative_to(root / "reports/research") and path.is_file(),
            "Invalid controlled-study evidence path",
        )
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            "Controlled study checksum mismatch: " + name,
        )
    _require(
        audit["evaluation_report_sha256"] == hashes[EVALUATION],
        "Evaluation differs from its recorded audit",
    )
    scores: dict[str, float] = {}
    denominators = None
    for name in ("fusion", "core", "selected"):
        record = evaluation["scores"][name]
        counts = record["objectives"]
        ds = {task: _count(counts[task]["denominator"], task, positive=True) for task in WEIGHTS}
        _require(denominators is None or denominators == ds, "Unmatched evaluation cohorts")
        denominators = ds
        recalls = {}
        for task in WEIGHTS:
            hits = _count(counts[task]["hits"], task)
            _require(hits <= ds[task], "Hits exceed target denominator")
            recalls[task] = hits / ds[task]
            _equal_number(recalls[task], counts[task]["recall_at_20"], f"{name} {task} recall")
        value = math.fsum(WEIGHTS[t] * recalls[t] for t in WEIGHTS)
        _equal_number(value, record["weighted_recall_at_20"], f"{name} weighted recall")
        scores[name] = value
    return {
        "status": "passed",
        "scope": "Recorded aggregate evidence; no live services or retraining.",
        "verified_release": {
            "submission_reference": reference,
            "private_score": release["private_score"],
            "public_score": release["public_score"],
            "evaluation_setting": release["evaluation_setting"],
            "sessions": sessions,
            "rows": rows,
            "prediction_sha256": identity,
        },
        "controlled_study": {
            "sessions": _count(evaluation["sessions"], "evaluation sessions", positive=True),
            "weighted_recall_at_20": scores,
            "selected_minus_core": scores["selected"] - scores["core"],
            "scope": "Frozen temporal cohort; not a competition score.",
        },
        "source_sha256": hashes,
    }
