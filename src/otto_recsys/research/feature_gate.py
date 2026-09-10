"""Keep an engineering release separate from the feature-research completion gate.

This verifies the versioned evidence inventory, not the scientific truth of a claim.
Family dispositions and their scope still require research review. A successful pilot,
an accepted submission, or absent coverage inventory cannot close this gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from otto_recsys.experiments.manifest import sha256_file

DISPOSITIONS = {"covered", "open", "excluded"}
RESULT_KINDS = {"controlled_result", "pilot_result"}


def feature_gate(root: Path) -> dict[str, Any]:
    path = root / "configs/feature_research.json"
    if not path.is_file():
        return {
            "feature_research_gate": "not_assessed",
            "final_training_ready": False,
            "feature_gate_reason": "No versioned feature coverage inventory is available.",
        }
    inventory = json.loads(path.read_text())
    if inventory.get("schema_version") != 1 or not inventory.get("scope"):
        raise ValueError("invalid feature research inventory schema or scope")
    evidence = inventory["evidence"]
    for entry in evidence.values():
        file = (root / entry["path"]).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file():
            raise ValueError("feature research evidence is missing or outside the project")
        if sha256_file(file) != entry["sha256"]:
            raise ValueError(f"feature research evidence checksum mismatch: {entry['path']}")
    families = inventory["families"]
    ids = [family["id"] for family in families]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("feature research families must be nonempty and unique")
    unresolved = []
    for family in families:
        disposition = family["disposition"]
        if disposition not in DISPOSITIONS or not family.get("scope"):
            raise ValueError("feature family needs a valid disposition and bounded scope")
        references = family["evidence"]
        if any(reference not in evidence for reference in references):
            raise ValueError("feature family references unknown evidence")
        if disposition == "covered" and not any(
            evidence[reference]["kind"] in RESULT_KINDS for reference in references
        ):
            raise ValueError("covered feature family requires measured result evidence")
        if disposition == "excluded" and not (
            family.get("exclusion_reason") and family.get("data_constraint")
        ):
            raise ValueError("excluded feature family requires a concrete data constraint")
        if disposition == "open":
            if not family.get("remaining_question") or not family.get("next_experiment"):
                raise ValueError("open feature family requires a question and next experiment")
            unresolved.append(family["id"])
    confirmation = inventory["temporal_confirmation"]
    if confirmation["status"] not in {"pending", "passed"}:
        raise ValueError("invalid feature confirmation state")
    if confirmation["status"] == "passed":
        references = confirmation.get("evidence", [])
        if not references or any(reference not in evidence for reference in references):
            raise ValueError("temporal confirmation requires known evidence")
        if not any(evidence[reference]["kind"] == "controlled_result" for reference in references):
            raise ValueError("temporal confirmation requires a controlled result")
    ready = not unresolved and confirmation["status"] == "passed"
    return {
        "feature_research_gate": "passed" if ready else "open",
        "final_training_ready": ready,
        "feature_coverage_scope": inventory["scope"],
        "feature_coverage_counts": {
            disposition: sum(f["disposition"] == disposition for f in families)
            for disposition in sorted(DISPOSITIONS)
        },
        "unresolved_feature_families": unresolved,
        "feature_temporal_confirmation": confirmation["status"],
        "feature_gate_next_task": inventory["next_task"],
        "feature_gate_inventory_sha256": sha256_file(path),
    }
