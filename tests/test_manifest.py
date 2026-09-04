import json
from pathlib import Path

from otto_recsys.experiments.manifest import (
    RunManifest,
    canonical_json_sha256,
)


def test_configuration_hash_is_order_independent() -> None:
    first = {"alpha": 1, "beta": 2}
    second = {"beta": 2, "alpha": 1}

    assert canonical_json_sha256(first) == canonical_json_sha256(second)


def test_run_manifest_round_trip(tmp_path: Path) -> None:
    manifest = RunManifest.start(
        "unit-test",
        config={"seed": 42},
        seed=42,
    )

    completed = manifest.finish(
        status="completed",
        elapsed_seconds=1.25,
        metrics={"recall": 0.5},
        artifacts={"model": "s3://example/model"},
    )

    output = tmp_path / "manifest.json"
    completed.write_json(output)

    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["status"] == "completed"
    assert loaded["seed"] == 42
    assert loaded["metrics"]["recall"] == 0.5
    assert loaded["git_commit"]
