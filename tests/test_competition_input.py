from __future__ import annotations

import hashlib
import json

import pytest

from otto_recsys.research.competition_input import check_competition_input


def test_missing_or_full_test_provenance_is_rejected_before_prediction(tmp_path):
    expected = {"conversion": {"source_name": "test.jsonl", "status": "complete"}}
    with pytest.raises(ValueError, match="conversion manifest"):
        check_competition_input(tmp_path, expected)
    (tmp_path / "manifest.json").write_text(json.dumps({"source_name": "otto-recsys-test.jsonl"}))
    with pytest.raises(ValueError, match="future events"):
        check_competition_input(tmp_path, expected)


def test_renamed_or_modified_partitions_cannot_reuse_valid_provenance(tmp_path):
    conversion = {"source_name": "test.jsonl", "sessions_processed": 1, "events_processed": 1}
    (tmp_path / "manifest.json").write_text(json.dumps(conversion))
    part = tmp_path / "part-000000.parquet"
    part.write_bytes(b"attested-prefix")
    expected = {"conversion": conversion, "source": "competition", "raw_sha256": "a" * 64,
                "files": {part.name: {"sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
                                       "bytes": part.stat().st_size}}}
    assert check_competition_input(tmp_path, expected)["events"] == 1
    part.write_bytes(b"future-events-added")
    with pytest.raises(ValueError, match="partition bytes"):
        check_competition_input(tmp_path, expected)
