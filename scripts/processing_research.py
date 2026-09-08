"""Standard-library bootstrap for the immutable SageMaker research job inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified(path: Path, expected: str) -> None:
    if digest(path) != expected:
        raise ValueError(f"input checksum mismatch: {path.name}")


def extract(archive: Path, destination: Path) -> None:
    """Accept ordinary files and directories, with no links or path traversal."""
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        members = bundle.getmembers()
        for member in members:
            path = destination / member.name
            if not (member.isfile() or member.isdir()) or not path.resolve().is_relative_to(
                destination.resolve()
            ):
                raise ValueError("input archive contains an unsafe member")
        bundle.extractall(destination, members=members, filter="data")


def assemble(directory: Path, output: Path, manifest_sha256: str) -> None:
    manifest_path = directory / "manifest.json"
    verified(manifest_path, manifest_sha256)
    manifest = json.loads(manifest_path.read_text())
    with output.open("wb") as target:
        for part in manifest["parts"]:
            path = directory / part["path"]
            if not path.resolve().is_relative_to(directory.resolve()):
                raise ValueError("bundle part escapes its input directory")
            verified(path, part["sha256"])
            if path.stat().st_size != part["bytes"]:
                raise ValueError("bundle part has an incorrect length")
            with path.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024**2)
    verified(output, manifest["sha256"])
    if output.stat().st_size != manifest["bytes"]:
        raise ValueError("assembled bundle has an incorrect length")


def stage_test(inputs: Path, project: Path, files: list[dict[str, Any]]) -> None:
    """Verify existing account-owned S3 inputs staged directly by SageMaker."""
    if not files:
        raise ValueError("competition inference requires a verified test inventory")
    destination = project / "artifacts/test"
    destination.mkdir(parents=True, exist_ok=True)
    for entry in files:
        name = entry["path"]
        if Path(name).name != name or Path(name).suffix not in {".json", ".parquet"}:
            raise ValueError("test inventory contains a noncanonical filename")
        source = inputs / "test" / name
        verified(source, entry["sha256"])
        if source.stat().st_size != entry["bytes"]:
            raise ValueError("staged competition input has an incorrect length")
        shutil.copyfile(source, destination / name)


def prepare(inputs: Path, workspace: Path, launch: dict[str, Any]) -> Path:
    source = inputs / "source/source.tar.gz"
    verified(source, launch["source_sha256"])
    project = workspace / "project"
    extract(source, project)
    root = project / "artifacts/research"
    root.mkdir(parents=True, exist_ok=True)
    corpus = root / "corpus"
    corpus.mkdir(exist_ok=True)
    for entry in launch["corpus"]:
        name = Path(entry["key"]).name
        path = inputs / "corpus" / name
        verified(path, entry["sha256"])
        if path.stat().st_size != entry["bytes"]:
            raise ValueError("corpus input has an incorrect length")
        shutil.copyfile(path, corpus / name)
    assembled = workspace / "retrieval_inputs.tar"
    assemble(inputs / "retrieval", assembled, launch["retrieval_manifest_sha256"])
    extract(assembled, root)
    assembled.unlink()
    models = inputs / "model/model_inputs.tar"
    verified(models, launch["model_inputs_sha256"])
    extract(models, root)
    if launch.get("task", "study") == "delivery":
        delivery = inputs / "delivery/delivery.tar"
        verified(delivery, launch["delivery_sha256"])
        extract(delivery, project / "artifacts")
        stage_test(inputs, project, launch["test_files"])
    return project


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("/opt/ml/processing/input"))
    parser.add_argument("--workspace", type=Path, default=Path("/opt/ml/processing/work"))
    args = parser.parse_args()
    launch_path = args.inputs / "launch/launch.json"
    launch = json.loads(launch_path.read_text())
    task = launch.get("task", "study")
    if task not in {"study", "delivery"}:
        raise ValueError("processing task must be study or delivery")
    print(
        json.dumps({"timestamp": datetime.now(UTC).isoformat(), "stage": "verify_inputs"}),
        flush=True,
    )
    args.workspace.mkdir(parents=True, exist_ok=True)
    project = prepare(args.inputs, args.workspace, launch)
    environment = dict(os.environ)
    environment.update(
        UV_PYTHON_INSTALL_DIR=str(args.workspace / "python"),
        UV_CACHE_DIR=str(args.workspace / "uv_cache"),
        UV_PROJECT_ENVIRONMENT=str(project / ".venv"),
        PIP_ROOT_USER_ACTION="ignore",
        PYTHONUNBUFFERED="1",
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    environment.pop("VIRTUAL_ENV", None)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.12.10"],
        check=True,
        env=environment,
    )
    command = [sys.executable, "-m", "uv"]
    subprocess.run(
        [*command, "sync", "--frozen", "--extra", "ml", "--extra", "cloud"],
        cwd=project,
        env=environment,
        check=True,
    )
    if task == "delivery":
        analysis = project.parent / "analysis"
        subprocess.run(
            [*command, "venv", str(analysis), "--python", "3.12.13", "--no-project"],
            cwd=project,
            env=environment,
            check=True,
        )
        subprocess.run(
            [
                *command,
                "pip",
                "install",
                "--python",
                str(analysis / "bin/python"),
                "--require-hashes",
                "-r",
                "notebooks/requirements.txt",
            ],
            cwd=project,
            env=environment,
            check=True,
        )
    subprocess.run(
        [
            str(project / ".venv/bin/python"),
            "-m",
            "otto_recsys.cloud.research_job"
            if task == "study"
            else "otto_recsys.cloud.delivery_job",
            str(launch_path),
        ],
        cwd=project,
        env=environment,
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
