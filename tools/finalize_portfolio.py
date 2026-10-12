"""One-time publication migration, with exact preconditions and no model changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    path = ROOT / "scripts/reproduce_reference.py"
    text = path.read_text()
    text = text.replace("*(base + day * i for i in range(1, 5)),", "history_end=base + day,\n            fit_end=base + 2 * day,\n            selection_end=base + 3 * day,\n            evaluation_end=base + 4 * day,")
    path.write_text(text)

    # Rebuild digest propagation from the fixed redacted originals on each pass.
    # Replacing already-replaced text strands intermediate digests in a dependency chain.
    path = ROOT / "tools/sanitize_public_metadata.py"
    text = path.read_text()
    text = text.replace("    for _ in range(32):\n", "    redacted_originals = dict(changed)\n    for _ in range(32):\n")
    text = text.replace("        updated = dict(changed)\n", "        updated = dict(redacted_originals)\n")
    path.write_text(text)

    receipt_path = ROOT / "reports/privacy/redactions.json"
    receipt = json.loads(receipt_path.read_text())
    path = ROOT / "configs/feature_research.json"
    inventory = json.loads(path.read_text())
    fixes = {
        "robustness": ("8f35e73aa0c446bc0c4d562772145e0363c257565a501c06b49e082cde756205",
                       "225f8e205c0d4c036c2cbdf9c2731c0cb67020414486c9dc1f7bb2b4b1fce483"),
        "shared_run": ("77679c410647ceedf2f959e937d2d5e45f2ea7ab0ed20a5925616b578672a9be",
                       "430c726ad4930f7d23b72f0be083d7b68dc757a004a8f0c623a63d35f56f5ba6"),
    }
    changed = False
    for key, (intermediate, expected) in fixes.items():
        ref = inventory["evidence"][key]
        if sha(ROOT / ref["path"]) != expected:
            raise ValueError("public evidence changed outside the authenticated redaction")
        if ref["sha256"] not in (intermediate, expected):
            raise ValueError("unexpected reference identity; no automatic replacement authorized")
        if ref["sha256"] != expected:
            ref["sha256"] = expected
            changed = True
    if changed:
        path.write_text(json.dumps(inventory, indent=2, ensure_ascii=False) + "\n")
        receipt["files"]["configs/feature_research.json"]["public_sha256"] = sha(path)
        receipt["dependency_reference_repair"] = {
            "scope": "Two stranded intermediate digests in the feature inventory",
            "method": "Exact expected public bytes verified before updating references",
            "measurements_changed": False,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")

    path = ROOT / "docs/EMPLOYER_OVERVIEW.md"
    text = path.read_text().replace(
        "| Full research workflows | Code, protocols, tests and executed evidence | Official data and larger private artifacts required |",
        "| Native training and full reference | Actual public research stages from raw JSONL | Invented fixture or independently obtained official training data |\n| Later neural-enhanced release | Recorded delivery and experimental evidence | Exact private-state reconstruction remains incomplete |",
    )
    text = text.replace(
        "The delivered release and current experiments have separate states. The\n[research status](RESEARCH_STATUS.md) records the inspected v27 negative decision\nwithout changing the confirmed score or treating ongoing feature research as complete.",
        "The [October 12 closeout](RESEARCH_STATUS.md) preserves the verified release and\nconcludes active experimentation. Earlier decisions remain archived; open research\nhypotheses are not mislabeled completed or scheduled for further execution.",
    )
    path.write_text(text)
    path = ROOT / "docs/REVIEWER_GUIDE.md"
    text = path.read_text().replace(
        "quality checks. Further candidate-retrieval research remains open and does not change the\nreleased model.",
        "quality checks. Active experimentation is concluded; remaining hypotheses do not change the\nreleased model.",
    )
    path.write_text(text)


if __name__ == "__main__":
    main()
