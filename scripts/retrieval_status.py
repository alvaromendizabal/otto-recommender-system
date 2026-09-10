"""Show the retrieval experiment and recent CloudWatch progress; never starts a job."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--receipt", type=Path, default=Path("reports/research/retrieval_run.json"))
    args = parser.parse_args()
    if not 15 <= args.interval <= 60:
        parser.error("--interval must be between 15 and 60 seconds")
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads((root / args.receipt).read_text())
    name = receipt["observed"]["ProcessingJobName"]
    sdk = importlib.import_module("boto3")
    configuration = importlib.import_module("botocore.config").Config(
        region_name=(
            receipt["region"]
            if "region" in receipt
            else receipt["observed"]["ProcessingJobArn"].split(":")[3]
        ),
        connect_timeout=5,
        read_timeout=20,
        retries={"mode": "standard", "total_max_attempts": 3},
    )
    sagemaker = sdk.client("sagemaker", config=configuration)
    logs = sdk.client("logs", config=configuration)
    seen = set()
    while True:
        job = sagemaker.describe_processing_job(ProcessingJobName=name)
        now = datetime.now(UTC)
        state = job["ProcessingJobStatus"]
        start = job.get("ProcessingStartTime")
        elapsed = (job.get("ProcessingEndTime", now) - start).total_seconds() if start else 0
        print(
            f"\n{now.isoformat(timespec='seconds')} {name} {state} elapsed={elapsed:.0f}s",
            flush=True,
        )
        streams = logs.describe_log_streams(
            logGroupName="/aws/sagemaker/ProcessingJobs",
            logStreamNamePrefix=name,
            limit=10,
        )
        for stream in streams.get("logStreams", []):
            events = logs.get_log_events(
                logGroupName="/aws/sagemaker/ProcessingJobs",
                logStreamName=stream["logStreamName"],
                limit=15,
                startFromHead=False,
            )
            for event in events.get("events", []):
                key = (event["timestamp"], event["message"])
                if key not in seen:
                    print(event["message"], flush=True)
                    seen.add(key)
        if job.get("FailureReason"):
            print(job["FailureReason"], flush=True)
        if state in {"Completed", "Failed", "Stopped"} or not args.watch:
            return int(state in {"Failed", "Stopped"})
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
