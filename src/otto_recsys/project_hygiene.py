from __future__ import annotations

import re
from pathlib import Path

FORBIDDEN_FILENAME_PATTERN = re.compile(
    r"(^|[_.-])(fix|fixed|repair|repaired|final_final|v\d+)([_.-]|$)",
    re.IGNORECASE,
)


def forbidden_project_filenames(root: str | Path) -> list[str]:
    """Return repository files that violate canonical naming policy."""
    base = Path(root)
    ignored_parts = {".git", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    violations: list[str] = []

    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.parts):
            continue
        if FORBIDDEN_FILENAME_PATTERN.search(path.name):
            violations.append(str(path.relative_to(base)))

    return sorted(violations)
