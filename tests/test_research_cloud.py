"""Verify byte integrity and restart behavior without credentials or paid resources."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import tarfile
from pathlib import Path

import pytest

from otto_recsys.cloud.research_checkpoints import ResearchCheckpoints


class Storage:
    def __init__(self):
        self.objects = {}
        self.uploads = 0
        self.corrupt_download = False

    def upload_file(self, path, bucket, key, ExtraArgs):
        assert bucket == "research-bucket"
        assert ExtraArgs["ExpectedBucketOwner"] == "123456789012"
        assert ExtraArgs["ChecksumAlgorithm"] == "SHA256"
        self.objects[key] = (Path(path).read_bytes(), ExtraArgs["Metadata"])
        self.uploads += 1

    def head_object(self, Bucket, Key, ExpectedBucketOwner):
        assert ExpectedBucketOwner == "123456789012"
        content, metadata = self.objects[Key]
        return {"ContentLength": len(content), "Metadata": metadata}

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self

    def paginate(self, Bucket, Prefix, ExpectedBucketOwner):
        assert ExpectedBucketOwner == "123456789012"
        for key in self.objects:
            if key.startswith(Prefix):
                yield {"Contents": [{"Key": key}]}

    def download_file(self, bucket, key, path, ExtraArgs):
        assert ExtraArgs["ExpectedBucketOwner"] == "123456789012"
        Path(path).write_bytes(b"wrong" if self.corrupt_download else self.objects[key][0])


def test_remote_checkpoints_validate_bytes_and_recover_partial_local_state(tmp_path):
    client = Storage()
    store = ResearchCheckpoints(
        tmp_path,
        "s3://research-bucket/otto/run1/",
        region="us-west-2",
        owner_account="123456789012",
        logger=logging.getLogger("checkpoint-test"),
        client=client,
    )
    model = tmp_path / "model.txt"
    model.write_text("native model bytes")
    store.publish(model)
    store.publish(model)
    assert client.uploads == 1
    model.write_text("partial local write")
    assert store.restore() == 1
    assert model.read_text() == "native model bytes"
    assert store.restore() == 0
    model.unlink()
    client.corrupt_download = True
    with pytest.raises(ValueError, match="SHA-256"):
        store.restore()
    assert not model.exists()
    client.corrupt_download = False
    assert store.restore() == 1
    client.objects["otto/run1/../escape.txt"] = (b"no", {})
    with pytest.raises(ValueError, match="invalid relative path"):
        store.restore()


def test_bootstrap_rejects_corrupt_parts_and_unsafe_archives(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/processing_research.py"
    spec = importlib.util.spec_from_file_location("processing_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "p1.bin").write_bytes(b"first")
    (tmp_path / "p2.bin").write_bytes(b"second")
    manifest = {
        "bytes": 11,
        "sha256": hashlib.sha256(b"firstsecond").hexdigest(),
        "parts": [
            {"path": name, "sha256": module.digest(tmp_path / name), "bytes": size}
            for name, size in (("p1.bin", 5), ("p2.bin", 6))
        ],
    }
    receipt = tmp_path / "manifest.json"
    receipt.write_text(json.dumps(manifest))
    module.assemble(tmp_path, tmp_path / "assembled.tar", module.digest(receipt))
    assert (tmp_path / "assembled.tar").read_bytes() == b"firstsecond"
    (tmp_path / "p2.bin").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        module.assemble(tmp_path, tmp_path / "assembled.tar", module.digest(receipt))
    archive = tmp_path / "archive.tar"
    with tarfile.open(archive, "w") as bundle:
        member = tarfile.TarInfo("../escape.txt")
        member.size = 4
        bundle.addfile(member, io.BytesIO(b"evil"))
    with pytest.raises(ValueError, match="unsafe"):
        module.extract(archive, tmp_path / "project")
    assert not (tmp_path / "escape.txt").exists()


@pytest.mark.parametrize("role", ["train", "test"])
def test_existing_competition_inputs_are_verified_before_staging(tmp_path, role):
    path = Path(__file__).resolve().parents[1] / "scripts/processing_research.py"
    spec = importlib.util.spec_from_file_location("processing_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "inputs" / role / "part-0000.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified fixture input")
    inventory = [
        {"path": source.name, "bytes": source.stat().st_size, "sha256": module.digest(source)}
    ]
    module.stage_events(tmp_path / "inputs", tmp_path / "project", inventory, role=role)
    assert (tmp_path / "project/artifacts" / role / source.name).read_bytes() == source.read_bytes()
    source.write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checksum"):
        module.stage_events(tmp_path / "inputs", tmp_path / "project", inventory, role=role)


def test_managed_job_failure_publishes_status_and_preserves_original_error(tmp_path, monkeypatch):
    from otto_recsys.cloud import research_job

    published = []

    class Checkpoints:
        def __init__(self, *args, **kwargs):
            pass

        def restore(self):
            return 0

        def publish(self, path):
            published.append(path)

    def fail(*args, **kwargs):
        raise ValueError("invalid verified input fixture")

    monkeypatch.setattr(research_job, "ResearchCheckpoints", Checkpoints)
    monkeypatch.setattr(research_job, "run_ablations", fail)
    launch = {
        "source_commit": "fixture",
        "source_sha256": "fixture",
        "checkpoint_uri": "s3://research-bucket/otto/run1",
        "region": "us-west-2",
        "owner_account": "123456789012",
        "training": {},
        "resources": {},
    }
    with pytest.raises(ValueError, match="invalid verified input"):
        research_job.run(launch, tmp_path)
    status = json.loads((tmp_path / "job_status.json").read_text())
    assert status["status"] == "failed"
    assert status["stage"] == "ablations"
    assert status["elapsed_seconds"] >= 0
    assert tmp_path / "job_status.json" in published
    assert tmp_path / "logs/managed_research.jsonl" in published


def test_replication_keeps_bootstrap_seed_separate_and_publishes_its_protocol(
    tmp_path, monkeypatch
):
    from otto_recsys.cloud import research_job
    from otto_recsys.research.robustness import seed_launch

    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    launch = seed_launch(
        root, "reference_seed_20260909", source_commit="a" * 40, source_sha256="b" * 64
    )
    calls = []

    class Checkpoints:
        def __init__(self, *args, **kwargs):
            pass

        def restore(self):
            return 0

        def publish(self, path):
            assert path.is_file()

    def train(_root, config, **kwargs):
        calls.append(("train", config["seed"]))
        return {"study_id": "test-study"}

    def evaluate(_root, **kwargs):
        calls.append(("evaluation", kwargs["seed"]))
        assert kwargs["bootstrap_replicates"] == 1000
        return {"input_id": "test-evaluation"}

    monkeypatch.setattr(research_job, "ResearchCheckpoints", Checkpoints)
    monkeypatch.setattr(research_job, "run_ablations", train)
    monkeypatch.setattr(research_job, "run_evaluation", evaluate)
    status = research_job.run(launch, tmp_path)
    assert calls == [("train", 20260909), ("evaluation", 20260908)]
    assert status["status"] == "passed"
    assert status["robustness"] == launch["robustness"]
    assert status["evaluation_seed"] == 20260908


@pytest.mark.parametrize("task", ["study", "delivery"])
def test_bootstrap_overrides_inherited_container_environment(tmp_path, monkeypatch, task):
    path = Path(__file__).resolve().parents[1] / "scripts/processing_research.py"
    spec = importlib.util.spec_from_file_location("processing_research", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    launch = tmp_path / "input/launch/launch.json"
    launch.parent.mkdir(parents=True)
    launch.write_text(json.dumps({"task": task}))
    project = tmp_path / "project"
    project.mkdir()
    calls = []
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venv")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setattr(module, "prepare", lambda *args: project)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    monkeypatch.setattr(
        module.sys,
        "argv",
        [str(path), "--inputs", str(tmp_path / "input"), "--workspace", str(tmp_path / "work")],
    )
    assert module.main() == 0
    assert len(calls) == (3 if task == "study" else 5)
    for _, kwargs in calls:
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(project / ".venv")
        assert "VIRTUAL_ENV" not in kwargs["env"]
    assert calls[-1][0][0] == str(project / ".venv/bin/python")
    assert (
        calls[-1][0][2] == f"otto_recsys.cloud.{'research' if task == 'study' else 'delivery'}_job"
    )


@pytest.mark.parametrize("fail_prediction", [False, True])
def test_delivery_orders_restore_explanation_and_prediction_and_publishes_status(
    tmp_path, monkeypatch, fail_prediction
):
    from otto_recsys.cloud import delivery_job

    actions = []

    class Checkpoints:
        def __init__(self, *args, **kwargs):
            pass

        def restore(self):
            actions.append("restore")

        def publish(self, path):
            assert path.is_file()

    def explain(*args, **kwargs):
        actions.append("explain")
        return {"input_id": "explanation"}

    def predict(*args, **kwargs):
        actions.append("predict")
        if fail_prediction:
            raise ValueError("invalid prediction input")
        return {"input_id": "prediction", "sha256": "fixture"}

    monkeypatch.setattr(delivery_job, "ResearchCheckpoints", Checkpoints)
    monkeypatch.setattr(delivery_job, "explain", explain)
    monkeypatch.setattr(delivery_job, "refresh_history", lambda *args: actions.append("refresh"))
    monkeypatch.setattr(delivery_job, "notebook_prediction", predict)
    monkeypatch.setattr(delivery_job, "export_replay", lambda *args, **kwargs: None)
    launch = {
        "region": "us-west-2",
        "owner_account": "123456789012",
        "study_checkpoint_uri": "s3://bucket/study",
        "checkpoint_uri": "s3://bucket/research",
        "prediction_checkpoint_uri": "s3://bucket/inference",
        "source_commit": "fixture",
        "source_sha256": "fixture",
        "training": {"seed": 4, "threads": 1},
        "resources": {"feature_workers": 1},
    }
    if fail_prediction:
        with pytest.raises(ValueError, match="invalid prediction"):
            delivery_job.run(launch, tmp_path)
    else:
        assert delivery_job.run(launch, tmp_path)["status"] == "passed"
    assert actions == ["restore", "restore", "restore", "explain", "refresh", "predict"]
    status = json.loads((tmp_path / "delivery_status.json").read_text())
    assert status["status"] == ("failed" if fail_prediction else "passed")
