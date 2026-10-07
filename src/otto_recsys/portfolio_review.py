"""Dependency-free consistency checks for the published OTTO review evidence."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

FRONTIER = "reports/research/current_frontier_20261005.json"
RELEASE = "reports/submissions/similarity_stack_20260925.json"
PROMOTION = "reports/research/frontier_status_20260928.json"
EVALUATION = "reports/research/evaluation.json"
MANIFEST = "reports/research/manifest.json"
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
    _require(math.isclose(_number(left, name), _number(right, name),
                         rel_tol=0, abs_tol=1e-12), f"Inconsistent {name}")


def review_snapshot(root: Path) -> dict[str, Any]:
    """Recompute aggregate arithmetic; never load credentials, data, or models.

    These checks verify consistency and recorded identities, not independent
    retraining, live Kaggle results, or authenticity of the original observations.
    """
    root = root.resolve()
    paths = (FRONTIER, RELEASE, PROMOTION, EVALUATION, MANIFEST, AUDIT)
    records = {name: json.loads((root / name).read_text()) for name in paths}
    hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in paths}
    current, release, promotion, evaluation, manifest, audit = (records[p] for p in paths)
    _require(datetime.fromisoformat(current["as_of_utc"]).tzinfo is not None,
             "Snapshot must carry a timezone")
    _require(current["metric"]["weights"] == WEIGHTS, "Unexpected metric weights")

    champion = current["verified_competition_champion"]
    incumbent = promotion["competition"]["incumbent"]
    _require(champion["status"] == "verified_post_competition",
             "Release must retain post-competition scope")
    _require(release["evaluation_setting"] == "post-competition Kaggle scoring",
             "Release must retain post-competition scope")
    _require(promotion["promoted_similarity_stack"]["decision"] == "PROMOTED_AND_DEPLOYED",
             "Missing recorded promotion decision")
    reference = _count(release["submission_ref"], "submission reference", positive=True)
    _require(reference == champion["submission_reference"] == incumbent["ref"],
             "Inconsistent submission reference")
    for split in ("private", "public"):
        value = _number(release[f"{split}_score"], split)
        _require(0 <= value <= 1, "Release score outside [0, 1]")
        _equal_number(value, champion[f"{split}_score"], f"{split} release score")
        _equal_number(value, incumbent[split], f"{split} promotion score")
        _equal_number(value - release[f"previous_{split}_score"], release[f"delta_{split}"],
                      f"{split} release delta")
    _require(release["sha256"] == incumbent["sha256"] and len(release["sha256"]) == 64
             and all(c in "0123456789abcdef" for c in release["sha256"]),
             "Inconsistent prediction identity")
    sessions = _count(release["sessions"], "release sessions", positive=True)
    rows = _count(release["rows"], "release rows", positive=True)
    _require(rows == 3 * sessions and sessions == incumbent["sessions"]
             and rows == incumbent["rows"], "Incomplete release coverage")
    _require(sessions == current["delivered_scale"]["official_test_sessions"]
             and rows == current["delivered_scale"]["official_output_rows"],
             "Inconsistent release scale")

    _require(manifest["status"] == evaluation["status"] == audit["status"] == "passed",
             "Controlled study is not accepted")
    _require(manifest["seal_id"] == evaluation["seal_id"] == audit["seal_id"],
             "Controlled study model identities disagree")
    for name in ("evaluation.json", "audit.json"):
        _require(manifest["files"][name] == hashes[f"reports/research/{name}"],
                 f"Controlled study checksum mismatch: {name}")
    _require(audit["evaluation_report_sha256"] == hashes[EVALUATION],
             "Evaluation differs from its recorded audit")
    scores = {}
    denominators = None
    for name in ("fusion", "core", "selected"):
        record = evaluation["scores"][name]
        counts = record["objectives"]
        ds = {task: _count(counts[task]["denominator"], task, positive=True)
              for task in WEIGHTS}
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

    frontier = current["research_frontier"]
    comparator = frontier["corrected_comparator"]
    _require(frontier["click_cart_stack_status"] == "rejected_after_corrected_comparator"
             and comparator["decision"] == "rejected_click_gate"
             and _number(comparator["click_hit_gain"], "click gain") < 0,
             "Rejected challenger cannot become the release")
    diagnostic = frontier["time_controlled_diagnostics"]
    denominator = _count(diagnostic["cart_denominator"], "cart denominator", positive=True)
    achieved = _count(diagnostic["strongest_ranked_cart_hits"], "ranked hits")
    ceiling = _count(diagnostic["original_candidate_pool_ceiling_hits"], "coverage hits")
    _require(achieved <= ceiling <= denominator, "Ranking exceeds candidate coverage")
    _require(_count(diagnostic["within_pool_misses"], "within-pool misses") == ceiling - achieved
             and _count(diagnostic["outside_pool_misses"], "outside-pool misses")
             == denominator - ceiling, "Candidate error decomposition disagrees")
    return {
        "status": "passed",
        "as_of_utc": current["as_of_utc"],
        "scope": "Recorded aggregate evidence; no live services or retraining.",
        "verified_release": {
            "submission_reference": reference,
            "private_score": release["private_score"],
            "public_score": release["public_score"],
            "evaluation_setting": release["evaluation_setting"],
            "sessions": sessions, "rows": rows, "prediction_sha256": release["sha256"],
        },
        "controlled_study": {
            "sessions": _count(evaluation["sessions"], "evaluation sessions", positive=True),
            "weighted_recall_at_20": scores,
            "selected_minus_core": scores["selected"] - scores["core"],
            "scope": "Frozen temporal cohort; not a competition score.",
        },
        "research_status": frontier["click_cart_stack_status"],
        "candidate_diagnostic": {
            "scope": "Separate time-controlled cart study; coverage is an oracle ceiling.",
            "denominator": denominator, "ranked_hits": achieved, "coverage_hits": ceiling,
            "within_pool_misses": ceiling - achieved, "outside_pool_misses": denominator - ceiling,
        },
        "next_research_question": diagnostic["current_question"],
        "source_sha256": hashes,
    }
