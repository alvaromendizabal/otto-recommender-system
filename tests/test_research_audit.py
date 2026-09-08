"""The publication audit rejects corrupted evidence and checks original targets."""

from __future__ import annotations

import copy
import logging

import pytest
from test_research_evaluation import build_evaluated_study

from otto_recsys.research.audit import audit_source, audit_statistics, audit_study


def test_independent_audit_reconstructs_truth_and_detects_tampered_claims(tmp_path):
    report = build_evaluated_study(tmp_path)
    audited = audit_study(
        tmp_path,
        tmp_path / "source",
        sessions=80,
        threads=1,
        memory_gib=1,
        logger=logging.getLogger("audit-test"),
    )
    assert audited["native_replay"]["comparisons"] == 80 * 18
    assert audited["native_replay"]["mismatches"] == 0
    assert audited["source"]["roles"] == {"evaluation": 80, "fit": 80, "selection": 80}
    false_claim = copy.deepcopy(report)
    false_claim["scores"]["selected"]["weighted_recall_at_20"] += 0.01
    with pytest.raises(ValueError, match="weighted metric"):
        audit_statistics(tmp_path, false_claim)
    (tmp_path / "source/part-0000.parquet").write_bytes(b"tampered source")
    with pytest.raises(ValueError, match="source partition checksum"):
        audit_source(tmp_path, tmp_path / "source", threads=1, memory_gib=1)
    (tmp_path / "evaluation/part-0000.parquet").write_bytes(b"tampered metric counts")
    with pytest.raises(ValueError, match="evaluation part checksum"):
        audit_statistics(tmp_path, report)
