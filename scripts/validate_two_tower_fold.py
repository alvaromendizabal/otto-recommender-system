from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from otto_recsys.cloud.fold_validation import (
    canonical_bundle_paths,
    extract_bundle,
    load_quality_pins,
    run_stage,
    utc_now,
    validate_clean_worktree,
    validate_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hermetically validate and optionally integrate the managed "
            "Fold 0 bundle."
        )
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--commit-message",
        default="Add managed Fold 0 neural retrieval experiment",
    )
    parser.add_argument("--push", action="store_true")
    return parser.parse_args()


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        text=True,
    )


def require_success(name: str, completed: subprocess.CompletedProcess[str]) -> None:
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with return code {completed.returncode}")


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    repo = args.repo.resolve()
    archive = args.archive.resolve()

    print("=" * 72)
    print("OTTO — MANAGED FOLD 0 HERMETIC VALIDATION")
    print(f"started_at={utc_now()}")
    print("=" * 72, flush=True)

    observed_sha, members = validate_clean_worktree(
        repo=repo,
        archive_path=archive,
        expected_head=args.expected_head,
        expected_sha256=args.expected_sha256,
    )

    if not args.apply:
        print(f"archive_sha256={observed_sha}")
        print(f"bundle_files={len(canonical_bundle_paths(members))}")
        print("OTTO_FOLD0_VALIDATED_NO_REPOSITORY_CHANGES")
        return 0

    print(f"[{utc_now()}] FOLD_VALIDATION_APPLY_START", flush=True)
    extract_bundle(archive, repo)
    pins = load_quality_pins(repo / "gpu" / "two_tower" / "requirements-dev.txt")

    # Recreate the same complete, lockfile-governed control-plane environment
    # used by the pristine worktree. This is deterministic and does not mutate
    # pyproject.toml or uv.lock because --frozen is mandatory.
    result = run_stage(
        "real_uv_sync_frozen_dev_ml",
        ["uv", "sync", "--frozen", "--extra", "dev", "--extra", "ml"],
        cwd=repo,
    )
    if not result.passed:
        raise RuntimeError("failed to synchronize locked dev + ml environment")
    validate_stage(repo, pins)

    bundle_paths = canonical_bundle_paths(members)
    completed = git(repo, "add", "--", *bundle_paths)
    require_success("git add", completed)
    require_success(
        "git diff --cached --check",
        git(repo, "diff", "--cached", "--check"),
    )

    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=repo,
        check=False,
    )
    if staged.returncode == 1:
        require_success("git commit", git(repo, "commit", "-m", args.commit_message))
    elif staged.returncode != 0:
        raise RuntimeError("unable to inspect staged changes")

    if args.push:
        require_success("git push", git(repo, "push"))

    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    require_success("git status", dirty)
    if dirty.stdout.strip():
        raise RuntimeError(
            f"repository is not clean after integration:\n{dirty.stdout}"
        )

    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    elapsed = time.perf_counter() - started
    print(f"HEAD={head}")
    print("working_tree=clean")
    print(f"elapsed_seconds={elapsed:.3f}")
    print("OTTO_MANAGED_FOLD0_IMPLEMENTATION_PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"OTTO_MANAGED_FOLD0_IMPLEMENTATION_FAILED error={exc}", file=sys.stderr)
        raise SystemExit(1) from exc
