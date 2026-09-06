from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from otto_recsys.cloud.ann_recovery import prepare_reference_reuse, publish_reference_reuse
from otto_recsys.cloud.sagemaker_pipeline import (
    canonical_sha256,
    create_deterministic_source_archive,
    source_s3_uri,
)
from otto_recsys.experiments.manifest import sha256_file


@pytest.fixture
def recovery(tmp_path: Path) -> dict:
    source = Path("gpu/two_tower").resolve()
    archive = tmp_path / "source.tar.gz"
    source_hash = create_deterministic_source_archive(source, archive)
    contract = {
        "source_sha256": source_hash,
        "code_commit": "a" * 40,
        "training_run_id": "training",
        "reference_run_id": "export",
        "reference_input_id": "reference",
        "reference_manifest_sha256": "b" * 64,
    }
    run_id = canonical_sha256(contract)
    uri = f"s3://bucket/retrieval/two-tower/ann/fold-0/{run_id}/checkpoints/"
    objects = {source_s3_uri("bucket", "a" * 40, source_hash): archive.read_bytes()}
    for objective in ("clicks", "carts", "orders"):
        key = uri + f"reference/{objective}/part-000.npz"
        part = tmp_path / (objective + ".npz")
        np.savez(part, sessions=np.array([0, 1]), counts=np.tile([1, 1, 1, 1, 1, 1, 0.05], (2, 1)))
        objects[key] = part.read_bytes()
        objects[key + ".json"] = json.dumps(
            {
                "input_id": run_id,
                "sha256": sha256_file(part),
                "bytes": part.stat().st_size,
                "elapsed_seconds": 1.2,
                "completed_at_utc": "2026-09-06T21:27:00+00:00",
            }
        ).encode()

    def download(key: str, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(objects[key])

    return {
        "objects": objects,
        "uri": uri,
        "kwargs": {
            "bucket": "bucket",
            "fold": 0,
            "previous_run_id": run_id,
            "previous_contract": contract,
            "contract": {**contract, "code_commit": "c" * 40},
            "reference": {
                "validation_fold": 0,
                "input_id": "reference",
                "ranking_manifest": {"config": {"buckets": 1}},
            },
            "source_root": source,
            "workspace": tmp_path / "recovery",
            "keys": {key.split("/", 3)[3] for key in objects},
            "download": download,
        },
    }


def test_reference_recovery_preserves_work_and_retries_interrupted_publication(
    recovery: dict,
) -> None:
    kwargs, objects = recovery["kwargs"], recovery["objects"]
    parts = prepare_reference_reuse(**kwargs)
    assert len(parts) == 3
    calls = []
    destination = "s3://bucket/new/checkpoints/"
    stop = True

    def upload(path: Path, uri: str) -> None:
        calls.append(uri)
        if stop and uri.endswith(".json"):
            raise OSError("interrupted receipt upload")
        objects[uri] = path.read_bytes()

    def publish() -> dict:
        return publish_reference_reuse(
            parts=parts,
            run_id="new",
            checkpoint_uri=destination,
            existing_keys={key.split("/", 3)[3] for key in objects},
            download=kwargs["download"],
            upload=upload,
        )

    with pytest.raises(OSError, match="interrupted"):
        publish()
    assert calls[0].endswith(".npz") and calls[1] == calls[0] + ".json"
    assert calls[0] in objects and calls[1] not in objects
    stop = False
    report = publish()
    assert len(report["published"]) == 3 and report["reused_compute_seconds"] == pytest.approx(3.6)
    for part in parts:
        assert objects[destination + part.name] == part.path.read_bytes()
        receipt = json.loads(objects[destination + part.name + ".json"])
        assert receipt["reused_from_run_id"] == kwargs["previous_run_id"]
        assert receipt["input_id"] == "new" and receipt["elapsed_seconds"] == 1.2
    count = len(calls)
    assert len(publish()["retained"]) == 3
    assert len(calls) == count  # Idempotent retries never overwrite committed parts.
    objects[destination + parts[0].name] = b"corrupt destination"
    with pytest.raises(ValueError, match="destination"):
        publish()


@pytest.mark.parametrize("damage", ["input", "source", "receipt", "payload", "schema", "identity"])
def test_incompatible_or_corrupt_work_is_rejected_before_publication(
    recovery: dict, damage: str
) -> None:
    kwargs, objects = recovery["kwargs"], recovery["objects"]
    key = recovery["uri"] + "reference/clicks/part-000.npz"
    if damage == "input":
        kwargs["contract"]["reference_input_id"] = "different"
    elif damage == "source":
        source_key = next(key for key in objects if key.endswith("source.tar.gz"))
        objects[source_key] = b"wrong archive"
    elif damage == "identity":
        kwargs["previous_contract"]["sample_sessions"] = 8
    elif damage == "receipt":
        receipt = json.loads(objects[key + ".json"])
        receipt["input_id"] = "other"
        objects[key + ".json"] = json.dumps(receipt).encode()
    elif damage == "payload":
        objects[key] = b"interrupted"
    else:
        import hashlib

        stream = io.BytesIO()
        np.savez(stream, sessions=np.array([0, 0]), counts=np.ones((2, 7)))
        objects[key] = stream.getvalue()
        receipt = json.loads(objects[key + ".json"])
        receipt.update(sha256=hashlib.sha256(objects[key]).hexdigest(), bytes=len(objects[key]))
        objects[key + ".json"] = json.dumps(receipt).encode()
    with pytest.raises(ValueError):
        prepare_reference_reuse(**kwargs)


@pytest.mark.parametrize("name", ["ranking_metrics.py", "ann_benchmark.py"])
def test_changed_metric_derivation_cannot_reuse_old_counts(
    recovery: dict, tmp_path: Path, name: str
) -> None:
    kwargs = recovery["kwargs"]
    source = tmp_path / "changed" / "otto_two_tower"
    source.mkdir(parents=True)
    for filename in ("ranking_metrics.py", "ann_benchmark.py"):
        text = (kwargs["source_root"] / "otto_two_tower" / filename).read_text()
        if filename == name:
            text = (
                text.replace("row[:20].tolist()", "row[:10].tolist()")
                if filename == "ann_benchmark.py"
                else text + "\n# changed metric contract\n"
            )
        (source / filename).write_text(text)
    kwargs["source_root"] = source.parent
    with pytest.raises(ValueError, match="implementation changed"):
        prepare_reference_reuse(**kwargs)


def test_only_committed_reference_parts_are_eligible(recovery: dict) -> None:
    kwargs, objects = recovery["kwargs"], recovery["objects"]
    key = recovery["uri"] + "reference/clicks/part-000.npz.json"
    kwargs["keys"].remove(key.split("/", 3)[3])
    # Unrelated files, including an index, are never read or promoted.
    kwargs["keys"].add(recovery["uri"].split("/", 3)[3] + "indices/clicks/index.faiss.json")
    objects.pop(key)
    parts = prepare_reference_reuse(**kwargs)
    assert len(parts) == 2 and all("clicks" not in part.name for part in parts)


def test_valid_metric_roundoff_does_not_discard_completed_work(recovery: dict) -> None:
    objects = recovery["objects"]
    key = recovery["uri"] + "reference/clicks/part-000.npz"
    with np.load(io.BytesIO(objects[key]), allow_pickle=False) as data:
        sessions, counts = data["sessions"], data["counts"]
    counts[0, 3] = np.nextafter(1.0, 2.0)
    stream = io.BytesIO()
    np.savez(stream, sessions=sessions, counts=counts)
    objects[key] = stream.getvalue()
    receipt = json.loads(objects[key + ".json"])
    receipt.update(sha256=hashlib.sha256(objects[key]).hexdigest(), bytes=len(objects[key]))
    objects[key + ".json"] = json.dumps(receipt).encode()
    assert len(prepare_reference_reuse(**recovery["kwargs"])) == 3
