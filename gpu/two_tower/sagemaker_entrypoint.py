from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument("--train-rows", type=int, required=True)
    parser.add_argument("--valid-rows", type=int, required=True)
    parser.add_argument("--checkpoint-steps", type=int, default=20)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_stage(name: str, command: list[str], heartbeat_seconds: float) -> int:
    started = time.perf_counter()
    print(f"[{_utc_stamp()}] STAGE_START name={name}", flush=True)
    process = subprocess.Popen(command)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(heartbeat_seconds):
            elapsed = time.perf_counter() - started
            print(
                f"[{_utc_stamp()}] HEARTBEAT stage={name} "
                f"elapsed_seconds={elapsed:.1f} pid={process.pid}",
                flush=True,
            )

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return_code = process.wait()
    finally:
        stop.set()
        thread.join(timeout=heartbeat_seconds + 1.0)
    elapsed = time.perf_counter() - started
    status = "PASS" if return_code == 0 else "FAIL"
    print(
        f"[{_utc_stamp()}] STAGE_COMPLETE name={name} status={status} "
        f"elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return return_code


def _channel(name: str, default: str) -> str:
    return os.environ.get(f"SM_CHANNEL_{name.upper().replace('-', '_')}", default)


def main() -> int:
    args = _parse_args()
    overall_started = time.perf_counter()
    checkpoint_dir = Path("/opt/ml/checkpoints")
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{_utc_stamp()}] OTTO_SAGEMAKER_ENTRYPOINT_START "
        f"code_commit={args.code_commit} resume={args.resume}",
        flush=True,
    )

    install_rc = _run_stage(
        "dependencies",
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            "requirements-dev.txt",
        ],
        args.heartbeat_seconds,
    )
    if install_rc != 0:
        return install_rc

    gate_rc = _run_stage(
        "gpu_package_quality_gate",
        [sys.executable, "run_quality_gate.py"],
        args.heartbeat_seconds,
    )
    if gate_rc != 0:
        return gate_rc

    runtime_rc = _run_stage(
        "gpu_runtime_validation",
        [sys.executable, "runtime_validation.py"],
        args.heartbeat_seconds,
    )
    if runtime_rc != 0:
        return runtime_rc

    train_command = [
        sys.executable,
        "-u",
        "train.py",
        "--ranking-cache",
        _channel("ranking", "/opt/ml/input/data/ranking"),
        "--hard-negatives",
        _channel("hard-negatives", "/opt/ml/input/data/hard-negatives"),
        "--item-data",
        _channel("items", "/opt/ml/input/data/items"),
        "--output-dir",
        str(checkpoint_dir),
        "--validation-fold",
        str(args.validation_fold),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--max-seq-len",
        str(args.max_seq_len),
        "--train-rows",
        str(args.train_rows),
        "--valid-rows",
        str(args.valid_rows),
        "--checkpoint-steps",
        str(args.checkpoint_steps),
        "--heartbeat-seconds",
        str(args.heartbeat_seconds),
        "--code-commit",
        args.code_commit,
    ]
    if args.resume:
        train_command.append("--resume")

    train_rc = _run_stage("training", train_command, args.heartbeat_seconds)
    if train_rc != 0:
        return train_rc

    for filename in (
        "best_model.pt",
        "metrics.json",
        "training_manifest.json",
        "run_contract.json",
        "progress.json",
    ):
        source = checkpoint_dir / filename
        if source.is_file():
            shutil.copy2(source, model_dir / filename)

    elapsed = time.perf_counter() - overall_started
    print(
        f"[{_utc_stamp()}] OTTO_SAGEMAKER_ENTRYPOINT_PASSED "
        f"total_seconds={elapsed:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
