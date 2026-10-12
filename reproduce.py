"""One entry point for the public review and native reference pipeline."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("review", "train", "research", "verify"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument("--train-jsonl", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research.toml")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-gib", type=int, default=2)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--execute", action="store_true", help="Authorize the full-data research run")
    args = parser.parse_args()
    if min(args.threads, args.workers, args.memory_gib, args.max_seconds) < 1:
        parser.error("resource limits must be positive")
    if args.workers > 16:
        parser.error("the native research pipeline supports at most 16 workers")
    output = args.output.resolve()
    if args.mode == "research" and not args.train_jsonl:
        parser.error("research requires --train-jsonl pointing to locally obtained OTTO training data")
    plan = {
        "mode": args.mode, "output": str(output), "threads": args.threads,
        "workers": args.workers, "memory_gib": args.memory_gib,
        "wall_time_limit_seconds": args.max_seconds, "cloud_calls": False,
        "submission_uploads": False, "dataset_downloads": False,
        "scope": "Public reference pipeline, not exact replay of the later neural release",
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.mode == "research" and not args.execute:
        print("PLAN_ONLY: review the resource limits and repeat with --execute to start.")
        return 0
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"):
        env[key] = str(args.threads)
    command = [sys.executable, str(ROOT / "scripts/reproduce_reference.py"), args.mode,
               "--output", str(output), "--config", str(args.config.resolve()),
               "--threads", str(args.threads), "--workers", str(args.workers),
               "--memory-gib", str(args.memory_gib)]
    if args.train_jsonl:
        command += ["--train-jsonl", str(args.train_jsonl.resolve())]
    process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=os.name != "nt")
    try:
        return process.wait(timeout=args.max_seconds)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False)
        else:
            import signal
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait()
        print("STOPPED: deadline or interruption; completed native checkpoints remain in the output.")
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
