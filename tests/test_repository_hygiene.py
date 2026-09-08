from pathlib import Path

from otto_recsys.project_hygiene import forbidden_project_filenames


def test_repository_uses_canonical_filenames() -> None:
    root = Path(__file__).resolve().parents[1]
    assert forbidden_project_filenames(root) == []


def test_hygiene_checks_project_source_and_excludes_generated_dependencies(tmp_path: Path) -> None:
    for name in ("src/model_fixed.py", "artifacts/analysis/dependency_v2.py", "src/model.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture")
    assert forbidden_project_filenames(tmp_path) == ["src/model_fixed.py"]
