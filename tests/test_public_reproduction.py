"""The public clone-to-results contract, with explicit synthetic-data scope."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("public_reference", ROOT / "scripts/reproduce_reference.py")
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixture_has_four_disjoint_roles_and_unseen_future(tmp_path: Path) -> None:
    source = tmp_path / "train.jsonl"
    MODULE.fixture(source)
    first = source.read_bytes()
    MODULE.fixture(source)
    assert source.read_bytes() == first
    rows = [json.loads(line) for line in first.splitlines()]
    assert len(rows) == 512
    assert len({r["session"] for r in rows}) == len(rows)
    assert all(list(sorted(e["ts"] for e in r["events"])) == [e["ts"] for e in r["events"]] for r in rows)
    assert any(e["aid"] == 999 for r in rows[448:] for e in r["events"])
    assert not any(e["aid"] == 999 for r in rows[:448] for e in r["events"])


def test_verify_rejects_tampering_and_path_escape(tmp_path: Path) -> None:
    p = tmp_path / "result.txt"
    p.write_text("complete")
    receipt = {"status": "passed", "files": {p.name: MODULE.digest(p)}}
    MODULE.write_json(tmp_path / "reproduction.json", receipt)
    assert MODULE.verify(tmp_path) == receipt
    p.write_text("altered")
    with pytest.raises(ValueError, match="checksum"):
        MODULE.verify(tmp_path)
    receipt["files"] = {"../other.txt": "0" * 64}
    MODULE.write_json(tmp_path / "reproduction.json", receipt)
    with pytest.raises(ValueError, match="artifact path"):
        MODULE.verify(tmp_path)


def test_research_is_plan_only_until_explicit_execution(tmp_path: Path) -> None:
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-S", str(ROOT / "reproduce.py"), "research",
                             "--train-jsonl", str(tmp_path / "not-downloaded.jsonl"),
                             "--output", str(output)], capture_output=True, text=True, check=True)
    assert "PLAN_ONLY" in result.stdout
    assert not output.exists()
    assert '\"cloud_calls\": false' in result.stdout


@pytest.mark.parametrize("arguments", [["--workers", "0"], ["--max-seconds", "0"],
                                        ["--workers", "17"], ["--memory-gib", "0"]])
def test_invalid_resources_fail_before_creating_output(tmp_path: Path, arguments: list[str]) -> None:
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-S", str(ROOT / "reproduce.py"), "train",
                             "--output", str(output), *arguments], capture_output=True, check=False)
    assert result.returncode == 2
    assert not output.exists()


def test_missing_research_data_argument_rejected() -> None:
    result = subprocess.run([sys.executable, "-S", str(ROOT / "reproduce.py"), "research"],
                            capture_output=True, check=False)
    assert result.returncode == 2


def test_missing_receipt_is_not_a_completed_reproduction(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE.verify(tmp_path)


def test_scorer_does_not_accept_future_target_columns() -> None:
    import inspect
    source = inspect.getsource(MODULE.export_predictions)
    assert '\"evaluation\"' in source
    assert "observed.parquet" in source
    assert "labels.parquet" not in source
    assert "verify_seal" in source
    assert "validate_submission" in source
