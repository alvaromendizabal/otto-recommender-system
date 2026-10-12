"""Check tracked public text for high-confidence secrets and deployment identifiers.

This is a publication safeguard, not a comprehensive secret audit or history rewrite.
Only filenames and rule names are emitted; matched values are never printed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from sanitize_public_metadata import account_ids, files

SECRET_PATTERNS = (
    ("aws-access-key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    (
        "private-key",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----\n[A-Za-z0-9+/]"),
    ),
    ("signed-cloud-url", re.compile(r"[?&]X-Amz-(?:Signature|Credential)=[A-Za-z0-9%]{16,}")),
)


def violations(text: str) -> list[str]:
    result = [name for name, pattern in SECRET_PATTERNS if pattern.search(text)]
    if account_ids(text):
        result.append("deployment-account-id")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = []
    for path in files(root):
        for rule in violations(path.read_text(encoding="utf-8")):
            findings.append({"path": str(path.relative_to(root)), "rule": rule})
    print(
        json.dumps(
            {
                "status": "passed" if not findings else "failed",
                "scope": "Current tracked UTF-8 publication, not Git history",
                "findings": findings,
            },
            indent=2,
        )
    )
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
