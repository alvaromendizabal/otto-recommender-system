"""Earlier-window execution must not inherit later data, selected features or seeds."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from otto_recsys.research.temporal_robustness import (
    original_source,
    protocol_values,
    verify_window_launch,
    verify_window_verification,
    window_launch,
    window_verification_launch,
)

ROOT = Path(__file__).resolve().parents[1]


def launch(seed=20260908, window="early"):
    return window_launch(
        ROOT, f"{window}_seed_{seed}", source_commit="a" * 40, source_sha256="b" * 64
    )


def test_shared_window_inputs_are_independent_of_seed_and_isolated_from_later_windows():
    first, second, middle = launch(), launch(20260909), launch(window="middle")
    assert first["input_checkpoint_uri"] == second["input_checkpoint_uri"]
    assert first["input_checkpoint_uri"] != middle["input_checkpoint_uri"]
    assert first["checkpoint_uri"] != second["checkpoint_uri"]
    for key in (
        "protocol", "train_files", "conversion_manifest", "retrieval", "features",
        "evaluation_seed", "bootstrap_replicates",
    ):
        assert first[key] == second[key]
    assert protocol_values(first)["history_end"] < protocol_values(middle)["history_end"]
    assert first["training"]["seed"] != second["training"]["seed"]
    assert not {"corpus", "model_inputs_sha256", "retrieval_manifest_sha256"} & first.keys()
    assert {
        p["path"]: p["sha256"] for p in first["train_files"]
    } == original_source(ROOT)["parts"]
    assert verify_window_launch(ROOT, first) == first["robustness"]
    with pytest.raises(ValueError, match="earlier cell"):
        launch(window="reference")


@pytest.mark.parametrize("change", [
    "future", "screen", "cohort", "bootstrap", "raw", "namespace",
    "training", "budget", "extra_inputs",
])
def test_window_launch_rejects_scientific_and_input_drift(change):
    bad = launch()
    if change == "future":
        bad["protocol"]["history_end"] = "2022-08-20T22:00:00+00:00"
    elif change == "screen":
        bad["features"]["max_retained"] = 64
    elif change == "cohort":
        bad["protocol"]["seed"] += 1
    elif change == "bootstrap":
        bad["evaluation_seed"] += 1
    elif change == "raw":
        bad["train_files"][0]["sha256"] = "c" * 64
    elif change == "namespace":
        bad["input_checkpoint_uri"] = launch(window="middle")["input_checkpoint_uri"]
    elif change == "training":
        bad["training"]["rounds"] += 1
    elif change == "budget":
        bad["resources"]["maximum_runtime_seconds"] *= 2
    else:
        bad["corpus"] = ["later-window-input"]
    with pytest.raises(ValueError, match="differs"):
        verify_window_launch(ROOT, bad)


def test_window_verifier_binds_completed_training_and_original_events():
    training = launch()
    verifier = window_verification_launch(
        ROOT, training, source_commit="c" * 40, source_sha256="d" * 64
    )
    assert verifier["train_files"] == training["train_files"]
    assert verifier["training_launch"] == training
    assert verifier["resources"]["maximum_runtime_seconds"] == 1800
    verify_window_verification(ROOT, verifier)
    changed = copy.deepcopy(verifier)
    changed["training_launch"]["training"]["seed"] += 1
    with pytest.raises(ValueError, match="differs"):
        verify_window_verification(ROOT, changed)


def test_input_audit_rejects_reference_corpus_before_reading_any_earlier_features(tmp_path):
    from otto_recsys.research.window_audit import verify_window_inputs

    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus/contract.json").write_bytes(
        (ROOT / "reports/research/temporal_contract.json").read_bytes()
    )
    with pytest.raises(ValueError, match="corpus contract differs"):
        verify_window_inputs(ROOT, tmp_path, launch())


def test_completed_parts_publish_data_before_receipts_and_ignore_heartbeats(tmp_path):
    from otto_recsys.cloud.window_job import CompletedParts

    published = []

    class Storage:
        def publish(self, path):
            published.append(path.relative_to(tmp_path).as_posix())

    handler = CompletedParts(tmp_path, Storage())
    logger = logging.getLogger("completed-window-part-test")
    for message, extras in (
        ("heartbeat", {}), ("research_feature_part_complete", {"bucket": 2})
    ):
        record = logger.makeRecord(
            logger.name, logging.INFO, "fixture", 1, message, (), None, extra=extras
        )
        handler.handle(record)
    assert published == ["screening_cache/part-0002.parquet", "screening_cache/part-0002.json"]


@pytest.mark.parametrize("auditing", [False, True])
def test_window_worker_orders_stages_and_preserves_training_receipt(
    tmp_path, monkeypatch, auditing
):
    from otto_recsys.cloud import window_job
    from otto_recsys.research import window_audit

    monkeypatch.chdir(ROOT)
    training = launch(20260909)
    calls = []
    root = tmp_path / "research"
    root.mkdir()
    original = root / "job_status.json"
    original.write_text('{"status":"passed","original":true}')
    before = original.read_bytes()

    class Storage:
        def __init__(self, root, uri, **kwargs):
            self.uri = uri

        def restore(self):
            calls.append(("restore", self.uri))

        def publish(self, path):
            assert path.is_file()

    def prepare(repo, root, source, actual, **kwargs):
        calls.append(("prepare", actual["protocol"]["seed"]))
        return {"input_id": "prepared-fixture"}

    def train(root, config, **kwargs):
        calls.append(("train", config["seed"]))
        return {"study_id": "study-fixture"}

    def evaluate(root, **kwargs):
        calls.append(("evaluate", kwargs["seed"]))
        return {"input_id": "evaluation-fixture"}

    def audit(repo, root, source, actual, **kwargs):
        assert actual == training
        calls.append(("audit", actual["training"]["seed"]))
        (root / "robustness_audit/report.json").write_text('{"audit_id":"fixture"}')
        return {"audit_id": "fixture"}

    monkeypatch.setattr(window_job, "ResearchCheckpoints", Storage)
    monkeypatch.setattr(window_job, "prepare_window", prepare)
    monkeypatch.setattr(window_job, "run_ablations", train)
    monkeypatch.setattr(window_job, "run_evaluation", evaluate)
    monkeypatch.setattr(window_audit, "verify_window", audit)
    actual = (
        window_verification_launch(ROOT, training, source_commit="c" * 40, source_sha256="d" * 64)
        if auditing else training
    )
    state = window_job.run(actual, root, tmp_path / "source")
    assert state["status"] == "passed"
    assert [c[0] for c in calls[:2]] == ["restore", "restore"]
    if auditing:
        assert original.read_bytes() == before
        assert calls[2:] == [("audit", 20260909)]
    else:
        assert calls[2:] == [
            ("prepare", 20260908), ("train", 20260909), ("evaluate", 20260908)
        ]
        assert json.loads(original.read_text())["preparation_id"] == "prepared-fixture"
