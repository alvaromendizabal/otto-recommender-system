from pathlib import Path

from otto_recsys.project_hygiene import forbidden_project_filenames


def test_repository_uses_canonical_filenames() -> None:
    root = Path(__file__).resolve().parents[1]
    assert forbidden_project_filenames(root) == []
