"""Convert the exact Kaggle competition prefixes, rejecting the released full test set."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from otto_recsys.data.convert import convert_jsonl_to_parquet
from otto_recsys.data.manifest import build_manifest, write_manifest
from otto_recsys.experiments.manifest import sha256_file
from otto_recsys.logging_utils import configure_logging

RAW_SHA256 = "8b46acddd68b46c83a474e809c21dbe681fed4c16304b0d0b6a907bd3e32790e"
RAW_BYTES = 402_090_304
SOURCE = "kaggle:competitions/otto-recommender-system/test.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/competition/raw/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/competition/processed/test"))
    args = parser.parse_args()
    if args.raw.name != "test.jsonl" or args.raw.stat().st_size != RAW_BYTES:
        raise ValueError("competition inference requires the 402,090,304-byte truncated test.jsonl")
    logger = configure_logging("competition_input")
    manifest = build_manifest([args.raw], source=SOURCE, logger=logger)
    if manifest.files[0].sha256 != RAW_SHA256:
        raise ValueError("test bytes differ from the downloaded Kaggle competition file")
    raw_manifest = args.raw.parent / "manifest.json"
    write_manifest(manifest, raw_manifest)
    converted = convert_jsonl_to_parquet(
        args.raw, args.output, raw_manifest, logger=logger, heartbeat_seconds=15
    )
    if converted.status != "complete" or converted.sessions_processed != 1_671_803:
        raise ValueError("competition test conversion is incomplete")
    report = {
        "schema_version": 1, "status": "passed", "source": SOURCE,
        "download_url": "https://www.kaggle.com/competitions/otto-recommender-system/data?select=test.jsonl",
        "raw_sha256": RAW_SHA256, "raw_bytes": RAW_BYTES,
        "conversion": asdict(converted),
        "files": {p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                  for p in sorted(args.output.glob("part-*.parquet"))},
    }
    destination = Path("reports/submissions/competition_input.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"status": "passed", "events": converted.events_processed,
                      "sessions": converted.sessions_processed, "parts": converted.parts_written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
