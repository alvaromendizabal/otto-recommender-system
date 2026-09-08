"""Read-only ranker progress and admission control without changing model identity.

A lock filename is not evidence of a running process. Probe its kernel lock;
never unlink it, kill its owner, or claim that a log timestamp proves liveness.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

OBJECTIVES = ("clicks", "carts", "orders")
LOCK_NAMES = (".launch.lock", ".lock")


def digest(value: Any, *, ascii_only: bool = False) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_only, allow_nan=False
    ).encode()).hexdigest()


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        raw = handle.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise ValueError(f"Progress object exceeds the 32 MiB inspection limit: {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def lock_status(path: Path) -> dict[str, Any]:
    """Non-mutating probe. An unavailable probe is unknown, not unlocked."""
    result: dict[str, Any] = {"path": str(path), "held": None, "owner_pids": []}
    try:
        with path.open("rb") as handle:
            stat = os.fstat(handle.fileno())
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                result["held"] = True
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                result["held"] = False
            if result["held"]:
                try:
                    for line in Path("/proc/locks").read_text().splitlines():
                        fields = line.split()
                        if len(fields) < 8 or fields[1:4] != ["FLOCK", "ADVISORY", "WRITE"]:
                            continue
                        major, minor, inode = fields[5].split(":")
                        if (int(major, 16), int(minor, 16), int(inode)) == (
                            os.major(stat.st_dev), os.minor(stat.st_dev), stat.st_ino
                        ) and int(fields[4]) > 0:
                            result["owner_pids"].append(int(fields[4]))
                except (OSError, ValueError):
                    pass  # PID namespaces may hide owners; the kernel lock is still held.
    except FileNotFoundError:
        result["held"] = False
    except OSError as error:
        result["probe_error"] = str(error)
    return result


class RankingBusy(RuntimeError):
    """A healthy writer must not be displaced by a duplicate command."""


@contextmanager
def launch_guard(output: Path) -> Iterator[None]:
    """Block duplicate CLI work before logs, downloads or candidate publication.

    The admission lock serializes new CLI instances through notebook publication.
    The unchanged ranker .lock remains authoritative for direct/legacy writers.
    This is local admission control, not a distributed lease. A legacy writer
    starting after the probe is still protected by the ranker's own final lock.
    """
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".launch.lock").open("ab") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RankingBusy("A ranking pipeline already owns this output directory") from error
        try:
            status = lock_status(output / ".lock")
            if status["held"] is True:
                raise RankingBusy("An existing ranker owns this output directory")
            if status["held"] is None:
                raise OSError("Cannot establish ranking workspace ownership; refusing to write")
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _checkpoint(directory: Path) -> dict[str, Any]:
    contract = read_object(directory / "contract.json")
    identity = digest(contract, ascii_only=True)
    rejected = 0
    paths = sorted((directory / "checkpoints").glob("*.json"), reverse=True)
    for path in paths:
        if not re.fullmatch(r"\d{6}\.json", path.name):
            continue
        try:
            saved = read_object(path)
            state = saved["state"]
            if (saved["checksum"] != digest(
                    {"model": saved["model"], "state": state}, ascii_only=True)
                    or state["input_id"] != identity):
                raise ValueError("checkpoint checksum or input mismatch")
            iteration, best = state["iteration"], state["best_iteration"]
            if (type(iteration) is not int or type(best) is not int
                    or not 0 < best <= iteration <= contract["config"]["rounds"]
                    or int(path.stem) != iteration
                    or type(state["complete"]) is not bool
                    or not math.isfinite(state["best_score"])
                    or not 0 <= state["best_score"] <= 1
                    or not math.isfinite(state["retained_fit_seconds"])
                    or state["retained_fit_seconds"] < 0):
                raise ValueError("invalid checkpoint progress")
            return {
                "status": "fit_complete" if state["complete"] else "checkpointed",
                "iteration": iteration, "planned_rounds": contract["config"]["rounds"],
                "best_iteration": best, "inner_recall_at_20": state["best_score"],
                "retained_fit_seconds": state["retained_fit_seconds"],
                "checkpoint": path.name, "rejected_checkpoints": rejected,
                "verification": "checksum and input identity; not an inference replay",
            }
        except (OSError, ValueError, TypeError, KeyError):
            rejected += 1
    return {"status": "no_verified_checkpoint", "rejected_checkpoints": rejected}


def _evaluation(directory: Path, run_id: str, fold: int, objective: str) -> dict[str, Any]:
    receipt = read_object(directory / "evaluation_receipt.json")
    identity = digest({"run_id": run_id, "fold": fold, "objective": objective})
    if receipt["input_id"] != identity or set(receipt["files"]) != {"evaluation.json", "model.txt"}:
        raise ValueError("evaluation receipt identity/file mismatch")
    for name, checksum in receipt["files"].items():
        if file_digest(directory / name) != checksum:
            raise ValueError(f"evaluation artifact checksum mismatch: {name}")
    result = read_object(directory / "evaluation.json")
    if result["model_sha256"] != receipt["files"]["model.txt"]:
        raise ValueError("evaluated model identity mismatch")
    for system in ("learned", "baseline"):
        metric = result[system]
        h, d = metric["hits"], metric["denominator"]
        if (type(h) is not int or type(d) is not int or not 0 <= h <= d or d <= 0
                or not math.isclose(metric["recall_at_20"], h / d, rel_tol=1e-12)):
            raise ValueError("evaluation score/count mismatch")
    if result["learned"]["denominator"] != result["baseline"]["denominator"]:
        raise ValueError("unmatched evaluation denominators")
    return {"status": "evaluated", "evaluation": result}


def read_progress(output: Path) -> dict[str, Any]:
    """Read a bounded snapshot without writes, training, S3 calls, or lock removal."""
    started = time.perf_counter()
    locks = [lock_status(output / name) for name in LOCK_NAMES]
    held = any(row["held"] is True for row in locks)
    probe_unknown = any(row["held"] is None for row in locks)
    result: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(), "scope": "local workspace snapshot",
        "workspace": str(output.resolve()), "writer_active": True if held else
        None if probe_unknown else False, "locks": locks, "objectives": [],
        "status": "running" if held else "unknown" if probe_unknown else "not_started",
        "weighted_recall_at_20": None,
    }
    path = output / "run_contract.json"
    if path.is_file():
        try:
            contract = read_object(path)
            folds = contract["outer_folds"]
            if (not folds or len(set(folds)) != len(folds)
                    or any(type(fold) is not int or fold < 0 or fold > 999 for fold in folds)):
                raise ValueError("invalid outer folds in run contract")
            run_id = digest(contract)
            result.update(run_id=run_id, candidate_id=contract["candidate_id"],
                          validation_scope=contract["validation_scope"],
                          feature_count=len(contract["feature_names"]))
            for fold in folds:
                for objective in OBJECTIVES:
                    directory = output / f"fold-{fold}" / objective
                    row: dict[str, Any] = {
                        "fold": fold, "objective": objective, "status": "pending"
                    }
                    try:
                        if (directory / "evaluation_receipt.json").is_file():
                            row.update(_evaluation(directory, run_id, fold, objective))
                        elif (directory / "contract.json").is_file():
                            row.update(_checkpoint(directory))
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        row.update(status="invalid_evidence", detail=str(error))
                    result["objectives"].append(row)
            rows = result["objectives"]
            evaluated = sum(row["status"] == "evaluated" for row in rows)
            result["evaluated_objectives"] = evaluated
            result["expected_objectives"] = len(rows)
            if evaluated == len(rows):
                weighted = 0.0
                for objective, weight in zip(OBJECTIVES, (0.1, 0.3, 0.6), strict=True):
                    metrics = [row["evaluation"]["learned"] for row in rows
                               if row["objective"] == objective]
                    weighted += weight * sum(m["hits"] for m in metrics) / sum(
                        m["denominator"] for m in metrics
                    )
                result["weighted_recall_at_20"] = weighted
                result["status"] = "publishing" if held else "evaluated"
            elif not held:
                result["status"] = "incomplete_or_interrupted"
            if any(row["status"] == "invalid_evidence" for row in rows):
                result["evidence_valid"] = False
        except (OSError, ValueError, KeyError, TypeError) as error:
            result.update(status="invalid_contract", detail=str(error))
    log = output / "logs/ranking.jsonl"
    if log.is_file():
        try:
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 65536))
                lines = handle.read(65536).decode("utf-8", errors="replace").splitlines()
            # Ignore duplicate failed launches; report a progress event, not arbitrary stderr.
            for line in reversed(lines):
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("message") not in {
                        "heartbeat", "ranker_checkpoint_complete", "ranking_evaluation_bucket",
                        "ranker_fit_complete", "ranking_complete",
                    }:
                        continue
                    result["last_progress"] = {key: event[key] for key in (
                        "timestamp", "stage", "message", "iteration", "best_iteration",
                        "elapsed_seconds", "total_elapsed_seconds", "rss_mb", "cpu_percent",
                    ) if key in event}
                    break
                except (ValueError, TypeError):
                    continue
        except OSError:
            result["log_readable"] = False
    result["inspection_elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result
