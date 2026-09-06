from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


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
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
    )
    parser.add_argument(
        "--resume-if-available",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
    )
    return parser.parse_args()


def _utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "captured_at": _utc_stamp(),
        "job_name": os.environ.get("TRAINING_JOB_NAME")
        or os.environ.get("SM_TRAINING_JOB_NAME"),
        "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE"),
        "num_cpus": os.environ.get("SM_NUM_CPUS"),
        "num_gpus": os.environ.get("SM_NUM_GPUS"),
        "run_id": os.environ.get("OTTO_RUN_ID"),
        "python": sys.version,
    }
    try:
        import torch

        snapshot["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            snapshot["torch"]["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # pragma: no cover - defensive diagnostics
        snapshot["torch_error"] = repr(exc)

    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if nvidia.returncode == 0:
        snapshot["nvidia_smi"] = [
            line.strip() for line in nvidia.stdout.splitlines() if line.strip()
        ]
    else:
        snapshot["nvidia_smi_error"] = nvidia.stderr.strip()
    return snapshot


def _write_failure_artifacts(
    *,
    stage: str,
    message: str,
    code_commit: str,
    return_code: int | None = None,
    command: list[str] | None = None,
    traceback_text: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    output_dir = Path(os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output"))
    output_data_dir = Path(
        os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    payload: dict[str, Any] = {
        "status": "failed",
        "failed_at": _utc_stamp(),
        "stage": stage,
        "message": message,
        "code_commit": code_commit,
        "return_code": return_code,
        "command": command,
        "elapsed_seconds": elapsed_seconds,
        "runtime": _runtime_snapshot(),
    }
    if traceback_text:
        payload["traceback"] = traceback_text

    try:
        _write_json_atomic(output_data_dir / "failure.json", payload)
        concise = (
            f"OTTO failure stage={stage} return_code={return_code} "
            f"message={message} code_commit={code_commit}"
        )
        _write_text_atomic(output_dir / "failure", concise[:1023] + "\n")
        print(
            f"[{_utc_stamp()}] FAILURE_ARTIFACT_WRITTEN stage={stage} "
            f"path={output_data_dir / 'failure.json'}",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover - diagnostics must not mask failure
        print(
            f"[{_utc_stamp()}] FAILURE_ARTIFACT_WRITE_FAILED "
            f"stage={stage} error={exc!r}",
            file=sys.stderr,
            flush=True,
        )


def _run_stage(
    name: str, command: list[str], heartbeat_seconds: float
) -> tuple[int, float]:
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
    return return_code, elapsed


def _channel(name: str, default: str) -> str:
    return os.environ.get(f"SM_CHANNEL_{name.upper().replace('-', '_')}", default)


def _run_or_record_failure(
    *,
    stage: str,
    command: list[str],
    heartbeat_seconds: float,
    code_commit: str,
) -> int:
    return_code, elapsed = _run_stage(stage, command, heartbeat_seconds)
    if return_code != 0:
        _write_failure_artifacts(
            stage=stage,
            message=f"stage command exited with return code {return_code}",
            code_commit=code_commit,
            return_code=return_code,
            command=command,
            elapsed_seconds=elapsed,
        )
    return return_code


def _runtime_bootstrap_commands() -> tuple[tuple[str, list[str]], ...]:
    return (
        (
            "dependencies",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-r",
                "requirements.txt",
            ],
        ),
        (
            "source_runtime_compile",
            [
                sys.executable,
                "-m",
                "compileall",
                "-q",
                "otto_two_tower",
                "train.py",
                "prepare.py",
                "runtime_validation.py",
                "sagemaker_entrypoint.py",
            ],
        ),
        (
            "gpu_runtime_validation",
            [sys.executable, "runtime_validation.py"],
        ),
    )


def main() -> int:
    args = _parse_args()
    overall_started = time.perf_counter()
    current_stage = "bootstrap"
    checkpoint_dir = Path("/opt/ml/checkpoints")
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    output_data_dir = Path(
        os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{_utc_stamp()}] OTTO_SAGEMAKER_ENTRYPOINT_START "
        f"code_commit={args.code_commit} resume={args.resume} "
        f"resume_if_available={args.resume_if_available}",
        flush=True,
    )

    try:
        for current_stage, bootstrap_command in _runtime_bootstrap_commands():
            bootstrap_rc = _run_or_record_failure(
                stage=current_stage,
                command=bootstrap_command,
                heartbeat_seconds=args.heartbeat_seconds,
                code_commit=args.code_commit,
            )
            if bootstrap_rc != 0:
                return bootstrap_rc

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
        if args.stop_after_step is not None:
            train_command.extend(["--stop-after-step", str(args.stop_after_step)])
        if args.resume:
            train_command.append("--resume")
        if args.resume_if_available:
            train_command.append("--resume-if-available")

        current_stage = "training"
        train_rc = _run_or_record_failure(
            stage=current_stage,
            command=train_command,
            heartbeat_seconds=args.heartbeat_seconds,
            code_commit=args.code_commit,
        )
        if train_rc != 0:
            return train_rc

        current_stage = "publish_outputs"
        for filename in (
            "best_model.pt",
            "metrics.json",
            "training_manifest.json",
            "run_contract.json",
            "progress.json",
            "resume_event.json",
            "resume_proof.json",
        ):
            source = checkpoint_dir / filename
            if source.is_file():
                shutil.copy2(source, model_dir / filename)

        elapsed = time.perf_counter() - overall_started
        summary = {
            "status": "passed",
            "completed_at": _utc_stamp(),
            "total_seconds": elapsed,
            "code_commit": args.code_commit,
            "run_id": os.environ.get("OTTO_RUN_ID"),
            "resume": args.resume,
            "resume_if_available": args.resume_if_available,
            "runtime": _runtime_snapshot(),
        }
        _write_json_atomic(output_data_dir / "entrypoint_summary.json", summary)
        print(
            f"[{_utc_stamp()}] OTTO_SAGEMAKER_ENTRYPOINT_PASSED "
            f"total_seconds={elapsed:.3f}",
            flush=True,
        )
        return 0
    except Exception as exc:
        traceback_text = traceback.format_exc()
        print(traceback_text, file=sys.stderr, flush=True)
        _write_failure_artifacts(
            stage=current_stage,
            message=str(exc) or exc.__class__.__name__,
            code_commit=args.code_commit,
            return_code=1,
            traceback_text=traceback_text,
            elapsed_seconds=time.perf_counter() - overall_started,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
