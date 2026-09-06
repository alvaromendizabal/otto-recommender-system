# Benchmark approximate neural retrieval

The saved Fold 0 model and exact top-800 export are complete. The independently
audited comparison found **+1.029 percentage points** of complementary candidate
ceiling. This workflow measures how much quality an approximate index preserves
and what search latency it achieves. It does not retrain the model.

**Implementation status:** the real-model CPU contract suite and recovery tests
pass. Full-catalogue managed ANN measurements remain pending a completed run.
The executed [benchmark notebook](../notebooks/06_ann_benchmark.ipynb) clearly
shows that pending state and renders measured results when they are available.

The first managed attempt on September 6, 2026 stopped during argument parsing.
SageMaker JSON-decodes hyperparameter strings before constructing the command:
the API value `"true"` became the CLI value `True`. The original parser accepted
only lowercase values. AWS recorded 306 billable seconds and zero ANN checkpoint
objects; the saved model and all 369 exact-export objects remain available.
The next attempt passed argument parsing and saved all 96 reference-count parts,
then stopped at an incorrect requirement that catalogue IDs be sorted. AWS
recorded 205 billable seconds and 198 checkpoint objects. The actual catalogue
has 1,852,162 unique IDs in its trained vocabulary order; its inverse map is valid.
Sorting the IDs alone would misalign items and vectors. Search and score replay
now use the validated inverse map while preserving all original embedding rows.
The third attempt passed catalogue validation and imported those 96 parts, then
failed while restoring the first part from S3. Entering a botocore `StreamingBody`
directly as a context manager returned the raw `HTTPResponse`, which has no
`iter_chunks` method. The former in-memory test response did not reproduce that
SDK behavior. Both receipt and artifact reads now use `contextlib.closing` to
retain the SDK wrapper, including its content-length and checksum validation.
Interrupted downloads are closed and partial files are removed before retry.
AWS recorded 306 billable seconds; all 96 imported reference parts remain durable.

The [execution evidence](../reports/metrics/two_tower_fold0_ann_launch.json)
preserves all three failed attempts; the [catalogue audit](../reports/metrics/two_tower_fold0_catalogue.json)
records checks against actual S3 inputs. Neither is an ANN quality measurement.

The shared preflight now reproduces JSON decoding before CLI conversion, and the
ANN parser canonicalizes both boolean spellings. Regression tests load the
captured launch through the pinned AWS toolkit's real hyperparameter reader,
framework-parameter split, and serializer. The real-model subprocess test uses
that complete path before testing export and recovery. Startup errors also emit
UTC start/completion records with exit code and total time to CloudWatch, even
when argument parsing fails before the benchmark logger is initialized.

Recovery tests now exercise actual botocore streaming/checksum response classes,
the boto3 HTTP stack against a local S3 protocol fixture, and a complete small
model benchmark restored into a fresh directory. They cover multi-chunk reads,
truncation, checksum errors, timeouts, stream closure, receipt order, and reuse
of all completed stages. The worker test suite treats Python warnings as errors.
A production S3 reference part was also restored through the corrected reader
and its recorded SHA-256 verified without recomputation. The HTTP fixture does
not simulate multipart uploads or certify AWS IAM, CUDA, or full-catalogue scale.

**Container warning:** the recorded `LD_PRELOAD=/libchangehostname.so` loader
message originates before `ann_worker_start`. AWS's
[image startup script](https://github.com/aws/deep-learning-containers/blob/main/scripts/docker/pytorch/start_with_right_hostname.sh)
sets that variable for its hostname workaround. The saved job definition does
not set it. This warning is separate from the fatal S3 exception and remains
unresolved in the managed image; no blanket warning filter or untested container
entrypoint override has been added. Passing Python warning checks does not imply
a warning-free AWS bootstrap.

## Experiment and metrics

The canonical parameters are in [`configs/two_tower_ann.toml`](../configs/two_tower_ann.toml).
FAISS IVFFlat stores the original FP32 vectors and searches selected inverted
lists. It introduces approximation through list selection, without quantizing
the stored vectors. Increasing `nprobe` trades more search work for fidelity;
see the [FAISS search documentation](https://github.com/facebookresearch/faiss/wiki/Faster-search).

| Contract | Setting |
|---|---|
| Frozen inputs | Existing best model, 1,852,162-item embeddings, exact top 800 |
| Query split | 4,096 seeded sessions: 2,048 tuning, 2,048 confirmation |
| Index | Three objective-specific IVFFlat inner-product indexes |
| Training | 1,024 centroids, 65,536 sampled vectors, 20 iterations |
| Tuning sweep | `nprobe = 32, 64, 128, 256` |
| Selection | Smallest probe count with mean top-800 overlap ≥98% for every tuning objective |
| Confirmation | Evaluate only the selected setting on the reserved sessions |
| Full-fold export | Enabled only if every confirmation objective also meets the target |
| Compute | Existing proven image and `ml.g6.xlarge`, one worker, four CPU search threads |
| Runtime cap | 7,200 seconds per managed attempt; no automatic paid retry |

The **98% target is a prospective design decision**, not a measured result.
Failing that target is a valid completed experiment. It leaves
`confirmation_fidelity_passed=false`, and full-fold ANN export is skipped.
It does not trigger another job or silently relax the threshold.

| Measurement | Meaning |
|---|---|
| Official weighted Recall@20 | `0.1 × clicks + 0.3 × carts + 0.6 × orders`, with each objective's hits summed across sessions and divided by its summed `min(20, number of true items)` |
| NDCG@20, MRR@20 | Ranking position diagnostics, averaged over labeled sessions per objective |
| Hit rate@20, precision@20 | Labeled-session hit rate and mean top-20 precision, reported per objective |
| ANN overlap@20/400/800 | Fraction of saved exact neighbor IDs recovered at the same depth |
| Candidate ceiling@400/800 | Maximum final Recall@20 an ideal ranker could recover from that larger neural pool |
| Exact-positive retention | Fraction of exact-retrieved true items also found by ANN; this is not base-exclusive retention |
| Batch-1 latency | Warm CPU search plus FP32 reranking: p50, p95, p99; excludes query encoding, network, and index loading |
| Batch throughput | Separately timed search batches of 128 precomputed queries |
| Resource cost | Serialized index bytes, retained build time, load time, peak process RSS, worker time, and AWS billable seconds |

The metric follows the [official OTTO challenge definition](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md).
Unknown future-positive items remain misses. Observed items remain eligible.
Returned candidates are reranked by FP32 score, then ascending item ID. Ties
outside the retrieved pool may affect strict neighbor-ID overlap.

The exact reference is evaluated across all **103,468** held-out sessions.
After the fidelity gate passes, ANN predictions and official ranking metrics
are also generated across that full cohort. Weighted Recall@20 differences
include 500 seeded paired-session bootstrap draws. The confirmation cohort
is independent of ANN parameter selection, but Fold 0 was already used to
select the model checkpoint. All Fold 0 results remain exploratory validation.
The full-fold aggregate also includes the ANN tuning sessions.

## 1. Update and validate in Studio

Pull the reviewed changes from the existing project directory; generated benchmark output stays under
ignored `artifacts/`, so the launcher does not overwrite published reports.

```bash
cd "$HOME/otto-recommender-system" &&
git pull --ff-only origin main &&
uv sync --frozen --extra dev --extra ml &&
.venv/bin/python scripts/run_quality_gate.py &&
uv pip check --python .venv/bin/python
```

Require `OTTO_QUALITY_GATE_PASSED` before launching. Keep the existing locked CPU
environment. The GPU package has a separate exact dependency profile including
FAISS 1.15.0, NumPy 2.4.3, PyArrow 23.0.1, and boto3 1.43.89. Staging chooses that
profile without editing the training requirements. Botocore 1.43.89,
s3transfer 0.19.2, and urllib3 2.7.0 are also pinned in the ANN and test profiles
so CI and the worker use the same S3 streaming stack. PyTorch comes from the
previously successful managed image. The small real-model tests run against
PyTorch 2.13.0 CPU wheels in CI; they do not certify full-scale CUDA behavior.

## 2. Launch one managed benchmark

Run from the clean checkout matching `origin/main`:

```bash
OTTO_BUCKET=otto-recsys-560403859723-us-west-2
.venv/bin/python scripts/launch_two_tower_ann.py \
  --bucket "$OTTO_BUCKET" --region us-west-2 --fold 0 \
  --reuse-reference-run 2734f718f9ef0db5c4957365a0720c376497da361f890eead7518f89d95a8b76 \
  --start --watch --download
```

This starts **one paid managed worker**. The launcher reuses the successful
training run's image, role, and instance configuration, validates completed
training/export provenance, runs pinned source checks, and tests the exact
worker argument list before any cloud write. It verifies the staged archive
and its S3 round trip before starting compute. Arguments are tested against the
AWS training toolkit's actual JSON loading, framework-parameter filtering, and
CLI serialization, including boolean values. The launcher also downloads only
the catalogue manifest and its two small lookup arrays (about 15 MB), checks
their hashes against the frozen exact-export contract, and executes the same
lookup validator used by the packaged worker before any paid start.

The recovery flag above verifies the previous run contract and source archive,
requires identical reference inputs and metric derivation code, and validates
each committed reference-count part. It transfers eligible data before issuing
new receipts that record the original run, checksum, completion time and compute
time. An interrupted transfer can be retried. Existing valid destination parts
are retained; conflicts are rejected. Indexes, queries, timings and selections
are never imported through this reference-only path. Omit the flag when there
is no previous compatible reference work to recover.

The startup marker is `OTTO_ANN_TRACKED_SAFE_TO_DISCONNECT`. SageMaker owns the
job after launch: closing the terminal or Studio does not stop it. Initially
the monitor may show `Starting` or `Downloading`; the reference channel alone
contains approximately 4.6 GB. During execution, UTC heartbeats report stage,
objective, bucket or examples where applicable, elapsed time, CPU/RAM, and
available GPU/VRAM telemetry. CPU search can run while GPU utilization is low.

Completion prints `OTTO_TWO_TOWER_ANN_BENCHMARK_COMPLETED`, AWS billable seconds,
the selected probe count, the fidelity result, and full-fold ranking metrics
when the gate passed. Inspect the fidelity flag as well as execution status.
No other folds are scheduled by this command.

Without `--start` or `--watch`, the launcher registers the definition and prints
`OTTO_ANN_REGISTERED_NO_COMPUTE_STARTED`; it does not start paid compute.

## 3. Reconnect or resume

Reconnect to the tracked execution at any time:

```bash
.venv/bin/python scripts/launch_two_tower_ann.py \
  --bucket otto-recsys-560403859723-us-west-2 --region us-west-2 \
  --fold 0 --watch --download
```

The monitor's time limit or Ctrl-C detaches it; the remote worker continues.
A failed or runtime-limited worker can be deliberately retried with the same
`--start --watch --download` command and unchanged source/configuration. A
repeated start retains an active or successful run, preventing duplicate work.
Changing source, runtime contract, or parameters creates a separate identity.
Resume never accepts an incompatible checkpoint silently.

Across code revisions, use `--reuse-reference-run` only for reference counts
whose provenance and derivation still match. The launcher writes its audit to
`control/reference_reuse.json` in the new run's S3 prefix. The worker logs
`artifact_restored` or `artifact_reused` as it consumes those parts. Catalogue
preflight evidence is recorded in `control/catalogue_preflight.json`.

## Durable progress and evidence

State lives under
`s3://<bucket>/retrieval/two-tower/ann/fold-0/<run-id>/`:

| Location | Durable contents |
|---|---|
| `control/` | Source/input contract, pipeline definition, per-execution AWS time/status |
| `checkpoints/contract.json`, `cohort.json` | Runtime/configuration and frozen session split |
| `checkpoints/indices/` | Trained centroids, independently committed index shards, merged indexes |
| `checkpoints/queries/` | Query embeddings for the tuning/confirmation cohort |
| `checkpoints/tuning/`, `confirmation/` | Prediction batches, timing observations, checksum receipts |
| `checkpoints/prediction_export/` | Full-fold ANN Parquet predictions, counts, and manifest when accepted |
| `checkpoints/metrics.json` | Official and diagnostic metrics, paired intervals, fidelity and latency |
| `checkpoints/logs/` | Progress snapshots and uniquely timestamped attempt logs |

The worker explicitly uploads each artifact before its checksum receipt and
reports it complete only after those operations succeed. A fresh worker
restores verified S3 parts. Valid index shards and query-result batches are
reused; missing or corrupt parts are rebuilt. A failure during a part can
repeat that part, while previously committed parts survive. Input downloads
and the memory-mapped concatenation of saved embeddings may be repeated;
their durable source data already exists. No model retraining is required.

CloudWatch receives live worker logs. S3 receives a log snapshot after each
committed artifact and a timestamped final attempt log on normal exit or a
handled exception. A hard kill can omit the final attempt event; prior S3
snapshots and CloudWatch logs remain. `elapsed_seconds_this_attempt` measures
the current run; `retained_artifact_compute_seconds` totals the retained part
timings across attempts. AWS time is saved per execution under `control/`.

`--download` fetches compact reports/logs into `artifacts/two_tower_ann/<run-id>/`
and saves their location in `artifacts/two_tower_ann/latest.json`. It verifies
the report receipt before making it available to notebook 06. Large indexes
and predictions remain in S3. Review actual results before committing the
compact report as `reports/metrics/two_tower_fold0_ann.json` and re-executing
the notebook. That publication is a normal documented PR.

## Next decision: retained gain over the base

ANN fidelity alone does not prove preservation of the model's complementary
positives. After a successful full-fold export, run the existing resumable
CPU comparison against the same frozen baseline. This is a separate step;
it does not overwrite the already audited exact-search comparison.

First ensure the frozen CPU inputs are present using the downloads and resource
preflight in [FOLD_EVALUATION.md](FOLD_EVALUATION.md#3-compare-with-frozen-retrieval).
Then fetch only the accepted ANN predictions:

```bash
OTTO_BUCKET=otto-recsys-560403859723-us-west-2
OTTO_ANN_KEY="$(.venv/bin/python -c \
  'import json; p=json.load(open("artifacts/two_tower_ann/latest.json")); print(p["checkpoint_key"])')" &&
OTTO_ANN_RUN="$(.venv/bin/python -c \
  'import json; p=json.load(open("artifacts/two_tower_ann/latest.json")); print(p["run_id"])')" &&
aws s3 sync \
  "s3://$OTTO_BUCKET/${OTTO_ANN_KEY}prediction_export/" \
  "artifacts/two_tower_ann/$OTTO_ANN_RUN/prediction_export/" \
  --exclude 'counts/*' --region us-west-2 --only-show-errors &&
.venv/bin/python scripts/evaluate_two_tower_retrieval.py \
  --ranking-cache data/interim/ranking_training_cache \
  --predictions "artifacts/two_tower_ann/$OTTO_ANN_RUN/prediction_export" \
  --covisit-dir data/interim/covisit \
  --vectors models/item2vec/item_vectors.kv --index models/faiss/item.index \
  --output-dir "data/interim/two_tower_ann_comparison/$OTTO_ANN_RUN" \
  --checkpoint-uri "s3://$OTTO_BUCKET/retrieval/two-tower/ann-comparisons/fold-0" \
  --region us-west-2 --source-k 1200 --ann-k 800 --ef-search 1024 \
  --threads 4 --memory-limit 4GB
```

This CPU process runs on Studio; stopping that host interrupts it, and the same
command resumes from S3-committed count parts. Its `--ann-k` and `--ef-search`
configure the frozen baseline Item2Vec source, not the new neural IVF index.
The new output directory and S3 family preserve the original evidence. Do not
use `--publish-report` here: that older option targets the original exact-report
filename. Review and publish ANN comparison evidence through its own normal PR.

Only after fidelity, actual Recall@20 change, resource use, and retained base
increment are acceptable should the remaining OOF folds be scheduled. Ranker
training needs checkpoint selection nested inside its training folds and an
untouched temporal final evaluation. No SOTA claim follows from this benchmark
alone.

## Tests

CI runs the root and isolated neural quality gates. Tests cover exact launch
JSON decoding and serialization, the captured managed invocation, boolean
normalization/rejection, bootstrap failure totals, no-cloud-write argument
rejection, canonical archive staging,
active/completed-run reuse, terminal status, official capped denominators,
ranking diagnostics, paired uncertainty, real model/index execution, full-fold
export integrity, process restart, corrupt prediction recovery, unchanged good
index shards, S3 restoration, interrupted receipt publication, mismatched
identities, denied access, and UTC heartbeat/elapsed-time events. Catalogue tests
cover shuffled and sparse IDs, duplicate IDs, stray inverse-map rows, missing
neighbors, deterministic tie ordering, and exact-score alignment. The real-model
fixture uses shuffled sparse IDs through export, ANN search, and process recovery.
Production-input preflight tests reject bad hashes and mappings before cloud
writes. Cross-revision recovery tests reject changed data, changed metric code,
invalid arrays and corrupt receipts; verify data-before-receipt publication; and
prove that reference recovery completes before a paid start without modifying an
already active worker.
