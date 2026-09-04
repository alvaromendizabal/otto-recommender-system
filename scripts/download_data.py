from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from otto_recsys.aws.storage import require_bucket, upload_file
from otto_recsys.data.manifest import build_manifest, write_manifest
from otto_recsys.logging_utils import configure_logging
from otto_recsys.runtime import Heartbeat

DATASET = "otto/recsys-dataset"

EXPECTED_FILES = {
    "otto-recsys-train.jsonl": 11_307_535_945,
    "otto-recsys-test.jsonl": 750_426_722,
}


def kaggle_executable() -> str:
    candidate = shutil.which("kaggle")
    if candidate:
        return candidate

    fallback = Path.home() / ".local" / "bin" / "kaggle"
    if fallback.exists():
        return str(fallback)

    raise RuntimeError("Kaggle CLI was not found")


def main() -> int:
    logger = configure_logging("download_data")
    started = time.perf_counter()

    raw_dir = Path("data/raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_or_invalid = [
        name
        for name, expected_size in EXPECTED_FILES.items()
        if not (raw_dir / name).exists()
        or (raw_dir / name).stat().st_size != expected_size
    ]

    if missing_or_invalid:
        logger.info(
            "download_start",
            extra={
                "event": "download_start",
                "missing_or_invalid": missing_or_invalid,
            },
        )

        process = subprocess.Popen(
            [
                kaggle_executable(),
                "datasets",
                "download",
                "-d",
                DATASET,
                "-p",
                str(raw_dir),
                "--unzip",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        with Heartbeat(
            logger,
            stage="kaggle_download",
            interval_seconds=30.0,
            pid_provider=lambda: process.pid,
        ):
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)

            return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"Kaggle download failed with exit code {return_code}"
            )

    for name, expected_size in EXPECTED_FILES.items():
        path = raw_dir / name

        if not path.exists():
            raise FileNotFoundError(path)

        observed_size = path.stat().st_size
        if observed_size != expected_size:
            raise RuntimeError(
                f"{name}: expected {expected_size} bytes, "
                f"observed {observed_size}"
            )

    files = [raw_dir / name for name in EXPECTED_FILES]

    manifest = build_manifest(
        files,
        source=f"kaggle:{DATASET}",
        logger=logger,
    )

    manifest_path = raw_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    bucket = require_bucket()

    for path in files:
        upload_file(
            path,
            f"raw/official/{path.name}",
            logger=logger,
            bucket=bucket,
        )

    upload_file(
        manifest_path,
        "raw/official/manifest.json",
        logger=logger,
        bucket=bucket,
    )

    elapsed = round(time.perf_counter() - started, 3)

    logger.info(
        "download_complete",
        extra={
            "event": "download_complete",
            "manifest_id": manifest.manifest_id,
            "elapsed_seconds": elapsed,
        },
    )

    print(
        f"OTTO_DATA_INGESTION_PASSED "
        f"manifest_id={manifest.manifest_id} "
        f"elapsed_seconds={elapsed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
