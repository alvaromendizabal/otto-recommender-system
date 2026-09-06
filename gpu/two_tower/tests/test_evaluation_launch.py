"""Exercise the production definition, AWS serializer, worker, and restart together."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
from test_evaluation import fixture_inputs

from otto_two_tower.evaluation_cli import hyperparameters_to_argv, parse_args


@pytest.mark.parametrize("depth_parameter", ["candidate-depth", "k"])
def test_pipeline_to_worker_and_process_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    depth_parameter: str,
    toolkit_argv: Callable[[dict[str, str]], list[str]],
) -> None:
    repository = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(repository / "src"))
    from otto_recsys.cloud.two_tower_evaluation import evaluation_definition
    from otto_recsys.cloud.two_tower_fold import FoldTrainingConfig, build_fold_pipeline_definition

    inputs = fixture_inputs(tmp_path)
    training = build_fold_pipeline_definition(
        role_arn="role",
        image_uri="image",
        source_uri="source",
        commit="old",
        run_id="training",
        config=FoldTrainingConfig(bucket="bucket"),
    )
    definition = evaluation_definition(
        training_definition=training,
        bucket="bucket",
        training_run_id="training",
        evaluation_id="evaluation",
        source_uri="source",
        commit=inputs.code_commit,
        training_manifest={"input_id": inputs.training_input_id, "validation_fold": 0},
        input_manifests={"ranking": inputs.expected_ranking_id, "items": inputs.expected_items_id},
    )
    hyperparameters = definition["Steps"][0]["Arguments"]["HyperParameters"]
    production_argv = toolkit_argv(hyperparameters)
    assert hyperparameters_to_argv(hyperparameters) == production_argv
    assert parse_args(production_argv).k == 800

    # Redirect only input/output paths and search sizes to a real, small model.
    hyperparameters.pop("candidate-depth")
    hyperparameters[depth_parameter] = "5"
    for name in ("ranking_cache", "item_data", "model_dir", "output_dir"):
        hyperparameters[name.replace("_", "-")] = str(getattr(inputs, name))
    hyperparameters.update({"batch-size": "1", "chunk-size": "5", "heartbeat-seconds": "0.01"})
    argv = toolkit_argv(hyperparameters)
    assert hyperparameters_to_argv(hyperparameters) == argv
    command = [sys.executable, "evaluate.py", *argv, "--allow-cpu"]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "SM_MODEL_DIR": str(tmp_path / "model_output"),
        "CUDA_VISIBLE_DEVICES": "",
    }

    def run_worker() -> None:
        completed = subprocess.run(
            command,
            cwd=repository / "gpu/two_tower",
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert "OTTO_TWO_TOWER_PREDICTIONS_PASSED" in completed.stderr

    run_worker()
    manifest_path = inputs.output_dir / "prediction_manifest.json"
    first = json.loads(manifest_path.read_text())
    assert first["sessions"] == 2
    assert first["search"]["k"] == 5
    assert len(first["parts"]) == 6
    saved_model = (inputs.model_dir / "best_model.pt").read_bytes()
    damaged = inputs.output_dir / first["parts"][0]["path"]
    original_bytes = damaged.read_bytes()
    retained = inputs.output_dir / first["parts"][1]["path"]
    retained_mtime = retained.stat().st_mtime_ns
    # Simulate an incomplete attempt: no completion marker and one truncated part.
    manifest_path.unlink()
    damaged.write_bytes(b"interrupted write")
    run_worker()
    assert retained.stat().st_mtime_ns == retained_mtime
    assert damaged.read_bytes() == original_bytes
    assert (inputs.model_dir / "best_model.pt").read_bytes() == saved_model
    second = json.loads(manifest_path.read_text())
    assert second["input_id"] == first["input_id"]
    assert json.loads((tmp_path / "model_output/prediction_manifest.json").read_text()) == second
    records = [
        json.loads(line)
        for line in (inputs.output_dir / "logs/two_tower_evaluation.jsonl").read_text().splitlines()
    ]
    assert all(datetime.fromisoformat(record["timestamp"]).tzinfo for record in records)
    for event in ("heartbeat", "part_complete", "part_reused", "evaluation_complete"):
        assert any(record["message"] == event for record in records)
    totals = [record for record in records if record["message"] == "evaluation_complete"]
    assert len(totals) == 2
    assert all(record["elapsed_seconds"] > 0 for record in totals)
