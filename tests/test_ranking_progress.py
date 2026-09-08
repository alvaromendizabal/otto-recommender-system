"""Real process/lock tests; never use deletion as a lock-recovery strategy."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

from otto_recsys.ranking.progress import (
    RankingBusy,
    digest,
    file_digest,
    launch_guard,
    lock_status,
    read_progress,
)


@contextmanager
def held_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen([
        sys.executable, "-u", "-c",
        "import fcntl,sys; f=open(sys.argv[1],'ab'); fcntl.flock(f,fcntl.LOCK_EX); "
        "print('locked',flush=True); sys.stdin.read()", str(path),
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        yield child.pid
    finally:
        child.stdin.close()
        child.wait(timeout=5)
        child.stdout.close()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def run_fixture(tmp_path):
    root = tmp_path / "ranking"
    contract = {"outer_folds": [0], "candidate_id": "a" * 64,
                "feature_names": ["x"], "validation_scope": "synthetic test only"}
    write(root / "run_contract.json", contract)
    return root, digest(contract)


def checkpoint(root, objective="clicks", iteration=10):
    directory = root / "fold-0" / objective
    contract = {"objective": objective, "config": {"rounds": 300}}
    write(directory / "contract.json", contract)
    state = {"input_id": digest(contract, ascii_only=True), "iteration": iteration,
             "best_iteration": iteration, "best_score": 0.5, "retained_fit_seconds": 1.5,
             "complete": False}
    value = {"model": "synthetic model text", "state": state}
    value["checksum"] = digest(value, ascii_only=True)
    path = directory / "checkpoints" / f"{iteration:06d}.json"
    write(path, value)
    return path


def evaluation(root, identity, objective, hits=5):
    directory = root / "fold-0" / objective
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.txt").write_text("synthetic test model")
    score = {"hits": hits, "denominator": 10, "recall_at_20": hits / 10}
    value = {"learned": score, "baseline": score,
             "model_sha256": file_digest(directory / "model.txt")}
    write(directory / "evaluation.json", value)
    write(directory / "evaluation_receipt.json", {
        "input_id": digest({"run_id": identity, "fold": 0, "objective": objective}),
        "files": {name: file_digest(directory / name) for name in ("model.txt", "evaluation.json")},
    })


def snapshot(root):
    return {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_absent_status_is_read_only(tmp_path):
    root = tmp_path / "absent"
    result = read_progress(root)
    assert result["writer_active"] is False
    assert result["status"] == "not_started"
    assert not root.exists()


def test_stale_lock_filename_is_not_an_active_writer(tmp_path):
    path = tmp_path / ".lock"
    path.write_text("old process metadata")
    before = snapshot(tmp_path)
    assert lock_status(path)["held"] is False
    assert snapshot(tmp_path) == before
    with launch_guard(tmp_path):
        assert lock_status(tmp_path / ".launch.lock")["held"] is True
    assert path.read_text() == "old process metadata"


@pytest.mark.parametrize("name", [".lock", ".launch.lock"])
def test_real_other_process_is_detected_without_deletion(tmp_path, name):
    path = tmp_path / name
    with held_lock(path):
        inode = path.stat().st_ino
        assert lock_status(path)["held"] is True
        with pytest.raises(RankingBusy), launch_guard(tmp_path):
            raise AssertionError("must not reach work")
        assert path.stat().st_ino == inode
        assert read_progress(tmp_path)["writer_active"] is True
    assert lock_status(path)["held"] is False
    with launch_guard(tmp_path):
        pass


def test_guard_releases_on_exception_without_unlinking(tmp_path):
    with pytest.raises(ValueError), launch_guard(tmp_path):
        raise ValueError("interrupted work")
    assert (tmp_path / ".launch.lock").is_file()
    with launch_guard(tmp_path):
        pass


def test_checkpoint_state_and_progress_not_duplicate_error(tmp_path):
    root, _ = run_fixture(tmp_path)
    checkpoint(root)
    log = root / "logs/ranking.jsonl"
    log.parent.mkdir()
    log.write_text(json.dumps({"message": "heartbeat", "iteration": 10,
                               "stage": "ranking_clicks", "timestamp": "2026-09-08T01:00:00Z"})
                   + '\n{"message":"ranking_pipeline_failed"}\n{"unfinished":')
    before = snapshot(root)
    report = read_progress(root)
    assert report["objectives"][0]["iteration"] == 10
    assert report["status"] == "incomplete_or_interrupted"
    assert report["last_progress"]["iteration"] == 10
    assert report["weighted_recall_at_20"] is None
    assert snapshot(root) == before


def test_partial_scores_cannot_be_presented_as_weighted_final_score(tmp_path):
    root, identity = run_fixture(tmp_path)
    evaluation(root, identity, "clicks")
    report = read_progress(root)
    assert report["evaluated_objectives"] == 1
    assert report["expected_objectives"] == 3
    assert report["weighted_recall_at_20"] is None
    assert report["objectives"][0]["status"] == "evaluated"
    evaluation(root, identity, "carts", 6)
    evaluation(root, identity, "orders", 7)
    report = read_progress(root)
    assert report["status"] == "evaluated"
    assert report["weighted_recall_at_20"] == pytest.approx(0.65)
    with held_lock(root / ".launch.lock"):
        assert read_progress(root)["status"] == "publishing"


@pytest.mark.parametrize("damage", ["bytes", "receipt", "score", "model"])
def test_corrupt_objective_not_reported_evaluated(tmp_path, damage):
    root, identity = run_fixture(tmp_path)
    evaluation(root, identity, "clicks")
    directory = root / "fold-0/clicks"
    if damage == "bytes":
        (directory / "evaluation.json").write_text("{}")
    elif damage == "model":
        (directory / "model.txt").write_text("different model")
    elif damage == "receipt":
        path = directory / "evaluation_receipt.json"
        value = json.loads(path.read_text())
        value["input_id"] = "f" * 64
        write(path, value)
    else:
        path = directory / "evaluation.json"
        value = json.loads(path.read_text())
        value["learned"]["recall_at_20"] = 0.99
        write(path, value)
        receipt_path = directory / "evaluation_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["files"]["evaluation.json"] = file_digest(path)
        write(receipt_path, receipt)
    report = read_progress(root)
    assert report["objectives"][0]["status"] == "invalid_evidence"
    assert report["weighted_recall_at_20"] is None


def test_corrupt_latest_checkpoint_falls_back_without_changing_files(tmp_path):
    root, _ = run_fixture(tmp_path)
    checkpoint(root, iteration=10)
    path = checkpoint(root, iteration=20)
    path.write_text("{")
    before = snapshot(root)
    state = read_progress(root)["objectives"][0]
    assert state["iteration"] == 10
    assert state["rejected_checkpoints"] == 1
    assert snapshot(root) == before


def test_unknown_lock_permission_is_not_reported_idle(tmp_path, monkeypatch):
    path = tmp_path / ".lock"
    path.touch()

    def denied(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(fcntl, "flock", denied)
    assert lock_status(path)["held"] is None
    assert read_progress(tmp_path)["writer_active"] is None


def cli_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_ranking.py"
    spec = importlib.util.spec_from_file_location("ranking_progress_cli", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", [".lock", ".launch.lock"])
def test_cli_duplicate_does_no_data_or_logger_work(tmp_path, monkeypatch, capsys, name):
    cli = cli_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("duplicate launch performed work")

    for method in (
        "configure_logging", "prepare_ranking_inputs", "build_candidates", "run_ranking"
    ):
        monkeypatch.setattr(cli, method, forbidden)
    with held_lock(tmp_path / name):
        arguments = ["--output-dir", str(tmp_path), "--checkpoint-uri", "s3://otto-test/x"]
        assert cli.main(arguments) == 75
    text = capsys.readouterr().out
    assert "OTTO_RANKING_ALREADY_RUNNING" in text
    assert "Traceback" not in text
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "input_readiness.json").exists()


def test_cli_status_needs_no_aws_and_makes_no_directories(tmp_path, monkeypatch):
    cli = cli_module()

    def forbidden(*args, **kwargs):
        raise AssertionError("status cannot perform work")

    monkeypatch.setattr(cli, "_run", forbidden)
    root = tmp_path / "missing"
    assert cli.main(["--stage", "status", "--output-dir", str(root)]) == 0
    assert not root.exists()


@pytest.mark.parametrize("argument", ["-1", "nan", "inf", "86401"])
def test_cli_rejects_invalid_watch_duration(argument):
    cli = cli_module()
    with pytest.raises(SystemExit) as error:
        cli.main(["--stage", "status", "--watch-seconds", argument])
    assert error.value.code == 2
