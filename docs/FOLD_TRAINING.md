# Fold 0 training and evidence

The next experiment trains the objective-conditioned two-tower retriever on folds 1–4
and holds out Fold 0. Run from the SageMaker Studio terminal in the repository root.
Set `OTTO_BUCKET` to the existing project bucket before launching or monitoring.
Only one `ml.g6.xlarge` worker is requested. The configured limit is eight epochs and
21,600 seconds **per training attempt**. Capacity/service retries can extend total
elapsed time; this is not a six-hour total cost cap.

## 1. Reproduce the locked environment and validate

After integrating the reviewed repository change:

```bash
cd "$HOME/otto-recommender-system" &&
git pull --ff-only origin main &&
uv sync --frozen --extra dev --extra ml &&
.venv/bin/python scripts/run_quality_gate.py &&
uv pip check --python .venv/bin/python
```

Continue when `OTTO_QUALITY_GATE_PASSED` appears and dependency validation succeeds.
The commands do not enable `set -e` or exit the interactive shell.

## 2. Register and start the bounded Fold 0 experiment

```bash
cd "$HOME/otto-recommender-system" &&
.venv/bin/python scripts/launch_two_tower_fold.py \
  --bucket "$OTTO_BUCKET" \
  --config configs/two_tower.toml \
  --profile fold0 \
  --start
```

`--start` requests paid managed training. Without that flag, the launcher validates
and registers the definition, then prints `OTTO_TWO_TOWER_FOLD_REGISTERED_NO_GPU_STARTED`.
After `OTTO_TWO_TOWER_FOLD_STARTED_SAFE_TO_DISCONNECT`, SageMaker owns execution.
Wait for that marker before closing the launching terminal.

An identical repeated command recovers the active execution reference. `--force`
cannot create an overlapping execution for the same pipeline. Start requests use
an idempotency token derived from the run and previous executions. A partial
checkpoint alone does not count as completed training.

## 3. Follow progress in another terminal

```bash
cd "$HOME/otto-recommender-system" &&
.venv/bin/python scripts/two_tower_fold_status.py \
  --bucket "$OTTO_BUCKET" \
  --fold 0 --watch --show-logs --publish-report
```

The monitor polls every 30 seconds. It prints UTC timestamps, total monitoring
time, pipeline and training status, checkpoint sizes, available epoch/step
information, and recent CloudWatch logs. Training emits its own heartbeat with
GPU telemetry. Empty training logs during provisioning are expected; use the
secondary status to distinguish provisioning from model training.

Ctrl+C detaches the monitor without stopping the remote training job. The default
monitoring limit is six hours; reaching it also leaves the remote job running.
Rerun the monitor to reconnect. Failed/stopped execution exits nonzero.

## 4. Publish evidence after success

`--publish-report` writes `reports/metrics/two_tower_fold0_training.json` only when
the pipeline succeeds and its training manifest is available. Refresh
`notebooks/05_two_tower_results.ipynb` after that file exists. The notebook opens
from either the repository root or the notebooks directory.

A successful training run is an engineering milestone. It does not establish
retrieval quality. Next evaluate held-out standalone, base, and base-plus-neural
candidate recall at 20/50/100/200/400/800; unique positive hits; paired uncertainty;
ANN quality/latency; and time/resource use. Apply the decision criteria in
[TWO_TOWER_EXPERIMENTS.md](TWO_TOWER_EXPERIMENTS.md) before training other folds.
Do not infer neural Recall@20 from validation loss or the resume proof.

## Registration regression and validation scope

The rejected definition encoded `Metadata.ValidationFold` as the JSON number `0`.
The [AWS pipeline definition schema](https://github.com/aws-sagemaker-mlops/sagemaker-model-building-pipeline-definition-JSON-schema/blob/main/schema/pipeline-definition.schema.json)
requires metadata values to be strings. The builder now emits `"0"`; the shared
validator rejects invalid metadata before registration. Tests cover all five fold
IDs, invalid metadata types/lengths, registration without training, repeated
launches, partial-checkpoint resumption, idempotency, and monitor terminal states.
The local check covers metadata; SageMaker registration validates the full definition.

On September 6, 2026 the exact failed S3 definition was reproduced and compared
with the corrected builder output: only the metadata value's type differed.
SageMaker `CreatePipeline` accepted that corrected definition, with no execution
started. This proves registration compatibility, not a completed Fold 0 run.

Canonical filenames are enforced by `tests/test_repository_hygiene.py`. Existing
source files are edited in place; Git records their history.
