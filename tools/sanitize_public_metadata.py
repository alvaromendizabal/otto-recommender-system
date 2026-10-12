"""Redact deployment account identifiers from tracked text, preserving a byte-change receipt.

This does not rewrite Git history, remove research attribution, or alter model weights.
The original hashes identify pre-redaction bytes, not publicly reproducible bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

SAFE_ACCOUNTS = {"000000000000", "123456789012"}
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".txt",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".ipynb",
    ".html",
    ".svg",
    ".log",
}
RECEIPT = "reports/privacy/redactions.json"
PATTERNS = (
    re.compile(r"arn:(?:aws|aws-us-gov|aws-cn):[^:\s\"\\]+:[^:\s\"\\]*:(\d{12}):"),
    re.compile(r"otto-recsys-(\d{12})-[a-z]+-[a-z]+-\d"),
    re.compile(r"(?:owner_account|account_id)[\"\s\\:]+(\d{12})"),
)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    result = []
    for name in raw.decode().split("\0"):
        if not name or name == RECEIPT:
            continue
        p = root / name
        if p.suffix.lower() in TEXT_SUFFIXES and not p.is_symlink():
            result.append(p)
    return result


def account_ids(text: str) -> set[str]:
    return {m.group(1) for pattern in PATTERNS for m in pattern.finditer(text)} - SAFE_ACCOUNTS


def redact(root: Path, *, apply: bool) -> dict:
    original = {p: p.read_bytes() for p in files(root)}
    texts = {p: b.decode("utf-8") for p, b in original.items()}
    accounts = set().union(*(account_ids(t) for t in texts.values()))
    changed = dict(texts)
    for p, text in changed.items():
        for account in sorted(accounts):
            text = text.replace(account, "000000000000")
        changed[p] = text
    for _ in range(32):
        replacements = {
            sha(original[p]): sha(text.encode())
            for p, text in changed.items()
            if text.encode() != original[p]
        }
        updated = dict(changed)
        for p, text in updated.items():
            for before, after in replacements.items():
                text = text.replace(before, after)
            updated[p] = text
        if updated == changed:
            break
        changed = updated
    else:
        raise ValueError("cyclic byte-hash references; review the public metadata manually")
    differences = {
        str(p.relative_to(root)): {
            "original_sha256": sha(original[p]),
            "public_sha256": sha(text.encode()),
        }
        for p, text in changed.items()
        if text.encode() != original[p]
    }
    report = {
        "schema_version": 1,
        "scope": "Current tracked text only; Git history not rewritten",
        "policy": "Deployment account numbers replaced with the reserved all-zero placeholder",
        "measurement_values_edited": False,
        "credentials_copied": False,
        "original_hashes": "Provenance for pre-redaction files; not current public byte identities",
        "files": differences,
    }
    if apply and differences:
        for name in differences:
            (root / name).write_text(changed[root / name], encoding="utf-8")
        receipt_path = root / RECEIPT
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else report
        previous["files"].update(differences)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(previous, indent=2, sort_keys=True) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    report = redact(Path(__file__).resolve().parents[1], apply=args.apply)
    print(json.dumps({"pending_files": len(report["files"]), "applied": args.apply}))
    return 1 if report["files"] and not args.apply else 0


if __name__ == "__main__":
    raise SystemExit(main())
