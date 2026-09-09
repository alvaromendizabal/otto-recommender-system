"""Reject confounded comparisons and distinguish predictions from model metadata."""

from __future__ import annotations

import copy
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pytest

from otto_recsys.research.robustness import fingerprint
from otto_recsys.research.robustness_audit import top_items, verify_reference_contract


def test_reference_contract_allows_only_the_model_seed_to_change():
    original = {
        "config": {"seed": 4, "rounds": 10},
        "fit_cache_id": "same-queries",
        "code": {"training.py": "frozen-code"},
        "candidate_budget": 400,
    }
    changed = copy.deepcopy(original)
    changed["config"]["seed"] = 5
    verify_reference_contract(changed, original["config"], fingerprint(original))
    for key, value in (
        ("fit_cache_id", "different-queries"),
        ("candidate_budget", 200),
        ("code", {"training.py": "changed-code"}),
    ):
        corrupt = {**changed, key: value}
        with pytest.raises(ValueError, match="data or method"):
            verify_reference_contract(corrupt, original["config"], fingerprint(original))
    changed["config"]["rounds"] = 20
    with pytest.raises(ValueError, match="settings"):
        verify_reference_contract(changed, original["config"], fingerprint(original))


def test_prediction_identity_depends_on_ranked_items_and_deterministic_ties():
    aids = np.array([4, 1, 3, 2])
    scores = np.array([0.2, 0.8, 0.8, 0.2])
    first = top_items(aids, scores)
    assert first == [1, 3, 2, 4]
    # Different score values can still produce identical recommendation lists.
    assert top_items(aids, scores * 2) == first
    assert top_items(aids, -scores) != first
    assert len(top_items(np.arange(30), np.arange(30, dtype=float))) == 20


@pytest.mark.parametrize(
    "aids,scores", [([1, 1], [0.1, 0.2]), ([1, 2], [0.1]), ([1, 2], [0.1, float("nan")])]
)
def test_invalid_native_prediction_comparisons_are_rejected(aids, scores):
    with pytest.raises(ValueError, match="prediction comparison"):
        top_items(np.array(aids), np.array(scores))


def test_native_prediction_comparison_replays_real_models(tmp_path):
    from test_research_evaluation import build_evaluated_study

    from otto_recsys.research.robustness_audit import compare_predictions

    build_evaluated_study(tmp_path)
    seal = json.loads((tmp_path / "evaluation_seal.json").read_text())
    reference = tmp_path / "reference_models"
    reference.mkdir()
    for objective, model in seal["models"].items():
        shutil.copyfile(tmp_path / model["path"], reference / f"{objective}.txt")
    result = compare_predictions(tmp_path, reference, seal, sessions=16, seed=4)
    assert result["sessions"] == 16
    for value in result["objectives"].values():
        assert value["ordered_top20_changed"] == value["top20_set_changed"] == 0
        assert value["reference_ranking_sha256"] == value["replication_ranking_sha256"]


def test_incomplete_replication_cannot_be_published_as_verified(tmp_path):
    from otto_recsys.research.robustness import seed_launch
    from otto_recsys.research.robustness_audit import verify_replication

    repository = Path(__file__).resolve().parents[1]
    (tmp_path / "corpus").mkdir()
    (tmp_path / "evaluation").mkdir()
    shutil.copyfile(
        repository / "reports/research/temporal_manifest.json", tmp_path / "corpus/manifest.json"
    )
    shutil.copyfile(
        repository / "reports/research/evaluation.json", tmp_path / "evaluation/report.json"
    )
    (tmp_path / "job_status.json").write_text(
        '{"status": "running", "stage": "reserved_evaluation"}'
    )
    launch = seed_launch(
        repository, "reference_seed_20260909", source_commit="a" * 40, source_sha256="b" * 64
    )
    with pytest.raises(ValueError, match="incomplete"):
        verify_replication(
            repository, tmp_path, launch, logger=logging.getLogger("incomplete-test")
        )
    assert not (tmp_path / "robustness_audit/report.json").exists()
