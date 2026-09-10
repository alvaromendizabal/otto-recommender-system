"""Collect a completed domain study, verify selected checkpoints and publish audits.

Uses the normal AWS role chain for read-only access. No job is launched, no ranker
is fitted and no remote object is changed. Large candidate caches stay in S3.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging, utc_now_iso
from otto_recsys.research.domain_diagnostics import error_slices, profile
from otto_recsys.research.domain_feature_audit import ARMS, OBJECTIVES, audit
from otto_recsys.research.protocol import atomic_json


def collect(launch_path: Path, job_name: str, output: Path, reports: Path) -> dict[str, Any]:
    launch = json.loads(launch_path.read_text())
    if launch["task"] != "domain_features":
        raise ValueError("collector requires a domain-feature launch")
    sdk = importlib.import_module("boto3")
    configuration = importlib.import_module("botocore.config").Config(
        region_name=launch["region"],
        connect_timeout=10,
        read_timeout=120,
        retries={"mode": "standard", "total_max_attempts": 6},
    )
    job = sdk.client("sagemaker", config=configuration).describe_processing_job(
        ProcessingJobName=job_name
    )
    if job["ProcessingJobStatus"] != "Completed":
        raise ValueError(
            "managed study has not completed; retain and resume its existing checkpoints"
        )
    if f":{launch['owner_account']}:processing-job/" not in job["ProcessingJobArn"]:
        raise ValueError("managed job belongs to a different account")
    parsed = urlsplit(launch["checkpoint_uri"])
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError("invalid checkpoint URI")
    bucket, prefix = parsed.netloc, parsed.path.strip("/") + "/"
    expected_source = f"s3://{bucket}/{prefix.removesuffix('checkpoints/')}source/source.tar.gz"
    if not any(
        row["InputName"] == "source" and row["S3Input"]["S3Uri"] == expected_source
        for row in job["ProcessingInputs"]
    ):
        raise ValueError("managed job used a different source archive")
    output.mkdir(parents=True, exist_ok=True)
    logger = configure_logging("domain_collection", log_dir=output / "collection_logs")
    client = sdk.client("s3", config=configuration)

    def download(entry: tuple[str, Path, str | None]) -> None:
        key, destination, expected = entry
        head = client.head_object(
            Bucket=bucket, Key=key, ExpectedBucketOwner=launch["owner_account"]
        )
        digest = expected or head.get("Metadata", {}).get("sha256")
        if not digest or len(digest) != 64:
            raise ValueError("checkpoint lacks SHA-256 provenance")
        if destination.is_file() and sha256_file(destination) == digest:
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        client.download_file(
            bucket, key, str(temporary), ExtraArgs={"ExpectedBucketOwner": launch["owner_account"]}
        )
        if temporary.stat().st_size != head["ContentLength"] or sha256_file(temporary) != digest:
            raise ValueError("downloaded checkpoint failed size or SHA-256 verification")
        temporary.replace(destination)
        logger.info(
            "verified_checkpoint", extra={"artifact": destination.relative_to(output).as_posix()}
        )

    files = [
        "results.json",
        "study_contract.json",
        "screening.json",
        "job_status.json",
        "fit_cache/manifest.json",
        "selection_cache/manifest.json",
        "fit_cache/contract.json",
        "selection_cache/contract.json",
        "logs/domain_feature_study.jsonl",
        *(
            f"models/{arm}/{name}"
            for arm in ARMS
            for name in ("selection_statistics.parquet", "selection_metrics.json")
        ),
        *(
            f"models/{arm}/{objective}/{name}"
            for arm in ARMS
            for objective in OBJECTIVES
            for name in ("model.txt", "manifest.json", "contract.json")
        ),
    ]
    downloads = [(prefix + name, output / name, None) for name in files]
    corpus = output / "corpus"
    for name in ("manifest.json", "queries.parquet", "observed.parquet"):
        entry = next(row for row in launch["corpus"] if Path(row["key"]).name == name)
        downloads.append((entry["key"], corpus / name, entry["sha256"]))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, downloads))
    shutil.copyfile(launch_path, output / "launch.json")
    shutil.copyfile(corpus / "queries.parquet", output / "queries.parquet")
    status = json.loads((output / "job_status.json").read_text())
    if (
        status["status"] != "passed"
        or status["source_commit"] != launch["source_commit"]
        or status["source_sha256"] != launch["source_sha256"]
    ):
        raise ValueError("managed completion does not match the published source")
    return publish_reports(output, reports, job)


def publish_reports(output: Path, reports: Path, job: dict[str, Any]) -> dict[str, Any]:
    """Also supports an already verified local collection without AWS credentials."""
    launch = json.loads((output / "launch.json").read_text())
    status = json.loads((output / "job_status.json").read_text())
    result = json.loads((output / "results.json").read_text())
    if (
        job["ProcessingJobStatus"] != "Completed"
        or status["status"] != "passed"
        or status["study_id"] != result["study_id"]
        or status["source_sha256"] != launch["source_sha256"]
    ):
        raise ValueError("report publication requires a matched completed managed study")
    checked = audit(output)
    diagnostics = error_slices(output / "corpus", output)
    support = profile(output / "corpus")
    reports.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "results": result,
        "audit": checked,
        "screening": json.loads((output / "screening.json").read_text()),
        "slices": diagnostics,
    }
    for name, value in artifacts.items():
        atomic_json(reports / f"domain_feature_{name}.json", value)
    atomic_json(reports / "domain_prefix_profile.json", support)
    seconds = Decimal(str((job["ProcessingEndTime"] - job["ProcessingStartTime"]).total_seconds()))
    record = {
        "status": "passed",
        "managed_status": job["ProcessingJobStatus"],
        "job_name": job["ProcessingJobName"],
        "study_id": result["study_id"],
        "source_commit": launch["source_commit"],
        "source_sha256": launch["source_sha256"],
        "checkpoint_uri": launch["checkpoint_uri"],
        "audited_at_utc": utc_now_iso(),
        "processing_started_at_utc": job["ProcessingStartTime"].isoformat(),
        "processing_finished_at_utc": job["ProcessingEndTime"].isoformat(),
        "processing_seconds": str(seconds),
        "pipeline_seconds": status["elapsed_seconds"],
        "estimated_instance_compute_usd": str(
            seconds / Decimal(3600) * Decimal(launch["pricing"]["rate_usd_per_hour"])
        ),
        "cost_scope": (
            "Instance compute estimate only; not an invoice; "
            "excludes storage, requests, logs and transfer"
        ),
        "baseline_models_reused": 3,
        "new_models_fitted": 12,
        "resources": job["ProcessingResources"]["ClusterConfig"],
        "launch_sha256": sha256_file(output / "launch.json"),
        **{
            f"{name}_sha256": sha256_file(reports / f"domain_feature_{name}.json")
            for name in artifacts
        },
        "profile_sha256": sha256_file(reports / "domain_prefix_profile.json"),
        "collector_sha256": sha256_file(Path(__file__)),
        "evaluation_access": False,
        "kaggle_promotion": False,
    }
    atomic_json(reports / "domain_feature_run.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--launch", type=Path, default=Path("reports/research/domain_feature_launch.json")
    )
    parser.add_argument("--job-name", default="otto-domain-features-09629193067f")
    parser.add_argument("--output", type=Path, default=Path("artifacts/domain_feature_audit"))
    parser.add_argument("--reports", type=Path, default=Path("reports/research"))
    args = parser.parse_args()
    print(json.dumps(collect(args.launch, args.job_name, args.output, args.reports), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
