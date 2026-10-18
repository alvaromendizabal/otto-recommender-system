from pathlib import Path


def test_python_version_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / ".python-version").read_text(encoding="utf-8").strip() == "3.13.15"
