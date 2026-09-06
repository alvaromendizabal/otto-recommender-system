# Evaluate the saved Fold 0 retriever

Fold 0 training succeeded on September 6, 2026: four epochs, 9,600 optimizer
steps, 324.385 seconds in the trainer, and 621 billable instance seconds. The
best validation loss was 4.544512 at epoch 1. The existing model is the input to
this workflow. Training does not need to be repeated.

**Full-catalogue export is also complete.** On September 6, 2026, the saved model
exported all 103,468 held-out sessions across three objectives and 1,852,162
catalogue items. All 96 prediction parts and their receipts are in S3. Export
took 323.287 seconds; AWS recorded 627 billable instance seconds. The paired comparison also completed: **+1.029 pp weighted candidate ceiling**,
with a 95% paired interval of **+0.931 to +1.120 pp**. All 32 count parts, metrics,
and logs are in S3. Section 4 audits those saved results without repeating
training, export, or baseline retrieval. Sections 2–3 are retained as operational
reference for deliberate future runs.

The first evaluation attempt failed during argument parsing, before inference:
SageMaker serialized the hyperparameter `k` as `-k`, while the worker accepted
only `--k`. AWS recorded 160 billable seconds and no evaluation checkpoint
objects. The saved training model was unaffected. The pipeline now uses
`candidate-depth`; the worker also accepts the original `-k` and `--k` spellings.
The existing files are updated in place.

## 1. Integrate and validate

The completed comparison publishes `reports/metrics/two_tower_fold0_retrieval.json`.
Studio may have an untracked copy from `--publish-report`. Save that exact path
in Git's local stash before pulling the verified public copy; unrelated paths
are not included. The stash remains recoverable with `git stash list` and
`git stash show --include-untracked`. Do not apply it over the published report.

```bash
cd "$HOME/otto-recommender-system" &&
git stash push --include-untracked -m "Preserve local retrieval report" -- \
  reports/metrics/two_tower_fold0_retrieval.json &&
git pull --ff-only origin main &&
uv sync --frozen --extra dev --extra ml &&
.venv/bin/python scripts/run_quality_gate.py &&
uv pip check --python .venv/bin/python
```

The CPU control plane stays in its existing locked environment. PyTorch stays
in the separate managed neural package. CI also runs the neural tests with
PyTorch 2.13.0 CPU wheels and the managed container's NumPy/PyArrow versions.
Those tests exercise the actual model and retrieval code; they do not certify
GPU throughput or CUDA behavior.

The launch contract is now tested across the production pipeline builder,
the pinned AWS training toolkit's actual argument serializer, and a subprocess
running `evaluate.py` with real saved model weights on a small CPU fixture.
The subprocess restarts with a missing completion marker and a corrupt part;
tests verify that valid parts and the saved weights are unchanged, the damaged
part is regenerated, and UTC heartbeats and attempt totals are recorded.

## 2. Export exact full-catalogue predictions

Set `OTTO_BUCKET` to the existing project bucket. From a clean checkout matching
`origin/main`:

```bash
.venv/bin/python scripts/launch_two_tower_evaluation.py \
  --bucket "$OTTO_BUCKET" --fold 0 --start --watch --download
```

This explicitly starts **one paid managed evaluation worker** using the training
run's proven image, role, and instance type. The default per-attempt limit is
7,200 seconds. Automatic paid retries are disabled. Source compilation, lint,
typing, CPU-safe preflight, archive checks, and an S3 download/hash comparison
precede launch. The full neural CPU test suite is an additional CI gate.
Before any upload or pipeline mutation, the launcher passes the exact generated
hyperparameters to the packaged worker parser in a dependency-free subprocess.
Unknown arguments, invalid search sizes, and nonfinite heartbeat intervals fail
locally. `evaluation_launch_contract_complete status=passed` confirms this gate.

Without `--start` or `--watch`, the command only registers the pipeline and
prints `OTTO_TWO_TOWER_EVALUATION_REGISTERED_NO_GPU_STARTED`. A repeated start
retains an executing or successful run. A failed execution can be deliberately
retried with the same command and the same source/configuration.

For the argument-parsing failure described above, merge the launch-contract PR,
pull and validate as in section 1, then use `--start --watch --download` above.
The changed source creates a new evaluation identity and uses the same trained
weights. The failed attempt has no prediction parts to recover. Do not use
watch-only mode to start recovery: it reconnects to the last tracked execution.

The worker validates the original training manifest identities and input file
checksums, loads `best_model.pt` with strict state-dictionary matching, and
exports candidates for every held-out session and each objective. It searches
all catalogue items with FP32 inner products, ranks score ties by ascending aid,
and retains the top 800. Unknown future-positive items remain misses in the
comparison denominator. Observed items remain eligible recommendations.

Candidate embeddings and prediction parts are written atomically, with
SHA-256 receipts. Each rerun verifies completed parts before skipping them;
missing or corrupt parts are recomputed. SageMaker synchronizes the checkpoint
directory with S3. An interruption may repeat work from the current part or a
part that had not finished uploading. It does not require retraining.

Terminal heartbeats show UTC time, objective, bucket, progress, elapsed time,
and available GPU telemetry. JSONL logs persist alongside the artifacts.
Before inference starts, the monitor reports the worker's AWS secondary status
(for example, starting or downloading). On failure it prints the training-job
failure reason and billable seconds. A retry can reuse only parts that were
successfully uploaded and pass their checksums; a failure message does not claim
that such parts exist.
Closing the monitor does not stop the managed worker. Reconnect with:

```bash
.venv/bin/python scripts/launch_two_tower_evaluation.py \
  --bucket "$OTTO_BUCKET" --fold 0 --watch --download
```

Wait for `OTTO_TWO_TOWER_EVALUATION_EXPORT_PASSED`. The command prints the local
prediction directory and records it in `artifacts/two_tower_evaluation/latest.json`.
The checkpoint directory also contains full-catalogue embeddings; the download
command excludes those to avoid an unnecessary multi-gigabyte transfer.

## 3. Compare with frozen retrieval

The comparison uses the existing revisit, time/type/buy co-visitation, and
Item2Vec sources. It hashes their actual files and pins their content in a
comparison contract. These S3 downloads reuse unchanged local files:

```bash
aws s3 sync "s3://$OTTO_BUCKET/candidates/ranking-training-cache/" \
  data/interim/ranking_training_cache/ --only-show-errors &&
aws s3 sync "s3://$OTTO_BUCKET/retrieval/covisit/" \
  data/interim/covisit/ --only-show-errors &&
aws s3 sync "s3://$OTTO_BUCKET/retrieval/item2vec/" \
  models/item2vec/ --exclude '*' --include 'item_vectors.kv*' \
  --include 'manifest.json' --only-show-errors &&
aws s3 sync "s3://$OTTO_BUCKET/retrieval/faiss/" \
  models/faiss/ --only-show-errors
```

Use a Studio CPU instance with at least 12 GiB RAM, 8 GiB available RAM, and
20 GiB free disk, as for the existing incremental-recall workflow. The baseline
index is about 1.47 GB, vectors about 1 GB, and the three graph files about 2.6 GB.
The evaluator processes one held-out bucket at a time with a bounded DuckDB
memory budget. It leaves the CPU dependency lock unchanged.

```bash
OTTO_PREDICTIONS="$(.venv/bin/python -c \
  'import json; print(json.load(open("artifacts/two_tower_evaluation/latest.json"))["predictions_dir"])')" &&
.venv/bin/python scripts/evaluate_two_tower_retrieval.py \
  --ranking-cache data/interim/ranking_training_cache \
  --predictions "$OTTO_PREDICTIONS" \
  --covisit-dir data/interim/covisit \
  --vectors models/item2vec/item_vectors.kv \
  --index models/faiss/item.index \
  --output-dir data/interim/two_tower_comparison/fold-0 \
  --checkpoint-uri "s3://$OTTO_BUCKET/retrieval/two-tower/comparisons/fold-0" \
  --region us-west-2 \
  --source-k 1200 --ann-k 800 --ef-search 1024 \
  --threads 4 --memory-limit 4GB --publish-report
```

The completed run used a **4GB** DuckDB memory limit. Retain this value when
resuming its identity; changing it creates a separate comparison namespace.

Wait for `OTTO_TWO_TOWER_RETRIEVAL_EVALUATION_PASSED`. Repeating the command
reuses verified comparison parts. With `--checkpoint-uri`, the evaluator adds
its immutable comparison identity to the S3 prefix, restores verified remote
parts, and checks write access before loading the baseline. It uploads each
new count file before its checksum receipt, and reports
`comparison_part_durable` only after both transfers succeed. JSONL logs are
uploaded after every committed part; final metrics are uploaded before success
is printed. A failed upload stops computation and leaves the local part intact.
Malformed or corrupt remote parts are rejected and recomputed.

The current CPU comparison runs on the Studio host. Closing its terminal or
stopping the host may interrupt computation; reconnect and rerun the same
command to restore completed parts. It can recover on another workspace after
the frozen inputs and predictions are downloaded again. A part interrupted
before its remote receipt is committed may need to be recomputed. Changing
source, input contents, runtime versions, or comparison settings creates a
different identity and cannot silently reuse old results.

For a browser-independent process while Studio remains running, prefix the
comparison command with `nohup` and append
`> data/interim/two_tower_comparison/fold-0/console.log 2>&1 < /dev/null &`.
Create the output directory first, and monitor with
`tail -f data/interim/two_tower_comparison/fold-0/console.log`.
This does not keep the process alive if the Studio host itself stops; S3
checkpoints preserve verified progress for the next attempt.

The metrics distinguish `elapsed_seconds_this_attempt` from
`completed_bucket_compute_seconds`, which sums the retained per-bucket timings
across attempts. Local JSONL logs are under the comparison output's `logs/`.

The public result is `reports/metrics/two_tower_fold0_retrieval.json`. The executed
results notebook contains training, export, paired quality, and independent audit
evidence. Export batch timings are separate from serving-latency measurements.

## 4. Audit the completed comparison from durable counts

This small download contains the comparison contract, metrics, and 32 count/receipt
pairs. It does not download weights, embeddings, graphs, or prediction arrays.
The following command is safe to rerun: unchanged S3 files are reused. The audit
validates hashes and sessions and independently reconstructs all point estimates
and paired intervals; it does not replay raw-label retrieval.

```bash
OTTO_BUCKET=otto-recsys-560403859723-us-west-2
OTTO_COMPARISON_ID=3f67e29fd5e74436b25d79f75eb30d13c51672202a8a1e0e4ee73fb077e92543
OTTO_AUDIT_DIR="artifacts/comparison_audit/$OTTO_COMPARISON_ID"
aws s3 sync \
  "s3://$OTTO_BUCKET/retrieval/two-tower/comparisons/fold-0/$OTTO_COMPARISON_ID/" \
  "$OTTO_AUDIT_DIR/" --region us-west-2 --only-show-errors &&
.venv/bin/python scripts/audit_two_tower_comparison.py \
  --comparison-dir "$OTTO_AUDIT_DIR" \
  --expected-input-id "$OTTO_COMPARISON_ID" --buckets 32 \
  --report "$OTTO_AUDIT_DIR/audit.json"
```

Expected completion: `OTTO_TWO_TOWER_COMPARISON_AUDIT_PASSED`. Progress has UTC
part/iteration events, a 15-second heartbeat for long audits, and final status
and elapsed time on both success and failure. The report is atomically written;
local logs are appended under the comparison directory. The source count parts
are read-only. An interrupted audit restarts the small aggregation from those
saved parts; it does not restart the expensive experiment. The independently
verified public report and its log are committed under `reports/`.

## Interpretation and next decision

- Neural depths: 20, 50, 100, 200, 400, 800.
- Fixed base: up to 1,200 per co-visitation/revisit source plus 800 Item2Vec
  candidates. Recomputed Fold 0 results are the valid comparator; previous
  all-fold aggregate metrics are not interchangeable with this cohort.
- Neural standalone, base, and union are ideal final top-20 recall ceilings.
  Only standalone depth 20 is an actual ordered neural Recall@20. Union
  results do not represent a scored or budget-matched top-20 recommendation list.
- Both hits and denominators are capped at 20 per session/objective. Raw
  neural-only positive hits are reported separately because they may add no
  capped gain when the base already covers 20 positives.
- Uncertainty uses 500 seeded paired session bootstrap samples, preserving
  correlations across objectives and candidate depths.
- Fold 0 was used for early stopping and checkpoint selection. These are
  exploratory validation results; intervals do not account for model selection.
- Exhaustive retrieval provides the reference. ANN fidelity and serving latency
  remain a separate required benchmark; the exporter records batch timings,
  which must not be presented as single-request serving latency.

The complementary-gain gate has passed on exploratory Fold 0. Retain this model
as an additional source. ANN fidelity, latency, and candidate-budget trade-offs
remain to be measured before more folds are launched. See the next experiment
contract in `TWO_TOWER_EXPERIMENTS.md`.
