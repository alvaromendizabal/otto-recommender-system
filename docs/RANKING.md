# From audited retrieval to learned ranking

Status: **observed feature preparation is complete. Candidate materialization,
objective-specific LambdaRank evaluation, S3 recovery, and results-notebook
publication are implemented. A full-data ranking score has not yet been published.**
Synthetic integration tests exercise real Parquet, DuckDB, Gensim, FAISS and
LightGBM operations; their scores are not OTTO performance evidence.

## Run the next experiment

Use the existing SageMaker Studio CPU workspace and frozen inputs. Do not rerun
retriever training, ANN benchmarking, or the completed observed-feature build.
From the repository root and locked project environment:

```bash
.venv/bin/python scripts/run_ranking.py \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking \
  --publish-report --execute-notebooks
```

This command prepares baseline candidates, trains the click/cart/order rankers,
evaluates their complete outer queries against a matched ranking baseline, saves
measured metrics, creates `notebooks/08_ranking_evaluation.ipynb`, and executes the
analytical notebooks in an isolated pinned analysis environment. The executed
ranking notebook is copied to its canonical path and uploaded with its receipt
to the model run's S3 namespace. The notebook is created only after a measured
ranking report exists; it is not a placeholder result. The command creates no
SageMaker training job, endpoint, instance, or IAM resource. It uses the running
workspace and the existing S3 bucket; ordinary AWS workspace/storage charges apply.

Defaults: outer Fold 0; independent inner ranker selection; 100 candidates per
session/objective; complete 128-session feature batches; four CPU threads; 4 GB
DuckDB memory; a 20 GiB estimated in-memory training budget. The guard also checks
available RAM and rejects oversized fits before allocation. This is an estimate,
not a proof of maximum native-memory consumption. Rows are never silently sampled.
Training datasets are released before outer evaluation and subsequent objectives.

`--stage candidates` prepares only candidates; `--stage train` uses a verified
local candidate cache. `--outer-folds 0 1 2 3 4` evaluates all configured outer
folds, each with independent ranker fits. Changing folds, feature definitions,
inputs or model settings requires a new output directory so prior work survives.
Changing the candidate budget also requires a new `--candidate-dir`. Multiple
outer fits do not retroactively certify the frozen upstream retriever provenance.

## Required existing inputs

| Path | Purpose |
|---|---|
| `data/interim/ranking_training_cache` | Frozen examples, observed items and labels |
| `data/interim/ranking_features` | Audited observed features and full query ledger |
| `data/interim/covisit` | Saved time/type/buy matrices and manifests |
| `models/item2vec/item_vectors.kv` | Saved Item2Vec vectors and any sidecars |
| `models/faiss/item.index` | Saved baseline ANN index |

The candidate contract verifies actual input hashes and the Item2Vec validation
manifest. It reads the already committed observed-feature buckets without
modifying them or their code dependencies. A fresh workspace needs these frozen
inputs restored first; matching candidate/model checkpoints are then restored
from S3. S3 access uses the existing Studio AWS CLI credential chain.

The completed observed features contain 515,702 sessions, 1,544,172 session/items
and 1,547,106 evaluation queries in 32 checksum-committed buckets (28.53 MiB).
The independent DuckDB audit found zero key/count mismatches. Notebook 07 contains
the measured timings, split counts and observed-prefix distributions. Its durable
namespace remains unchanged:

```text
s3://otto-recsys-560403859723-us-west-2/ranking/features/82e8eac76c63d4d8a34b611bca0f3ae329623ff5cd80e18ca8bc238ddbd65795/
```

## Candidate and feature protocol

Generate revisit, time/type/buy co-visitation and Item2Vec sources from observed
inputs without inserting label items. Deduplicate by session/objective/item,
retaining each source's presence, rank and score. Compress membership using source
agreement, reciprocal-rank sum, Item2Vec score and ascending item ID for ties.
The 100-candidate budget is a fixed first baseline, not an optimized configuration.
It must be compared with other budgets by an inner-validation experiment.

Stream complete session groups through the canonical observed-feature join into
objective-specific float32 Parquet parts. Retain session event counts/duration,
item repeat/type counts, observed item recency and event share. Missing source
scores/ranks and unseen-item recency remain explicit missing values. Baseline
rankers exclude the three absent neural-source columns; IDs, targets and split
assignments never enter their feature matrix. The full query ledger is stored
separately, including sessions with no candidates. Missing positives remain misses.

## Validation and metrics

Within each outer fold, fitting uses other-fold sessions outside inner partition
0. Early stopping uses only inner partition 0 of those other folds. Outer sessions
are excluded from both fitting and checkpoint selection. Candidate groups remain
complete during validation; there is no forced-positive evaluation or hidden
negative sampling. The matched source-agreement/RRF baseline and learned ranker
are scored on exactly the same candidates and query denominators.

The official metric is:

```text
0.10 * Recall@20(clicks) + 0.30 * Recall@20(carts) + 0.60 * Recall@20(orders)
```

Per-session hits and true-item denominators are capped at 20 and pooled within
objectives. Across folds, pool numerators/denominators before applying the weights;
do not average fold scores equally. Report NDCG@20, MRR@20, hit rate, candidate
ceiling and fit/evaluation durations separately. Evaluation duration includes I/O,
scoring, sorting and metrics; it is not serving latency. The notebook validates
the reported aggregate against the saved fold counts before publication.

**Limit:** the frozen cache lacks future-label timestamps and certified upstream
retriever fit provenance. Results are exploratory nested session validation, not
an untouched temporal holdout or fully certified nested retrieval/ranking. The
previous Fold 0 neural checkpoint was selected using that fold. This baseline
runner therefore refuses neural-candidate contracts; it does not relabel that
checkpoint as independently selected.

The earlier uncompressed ANN K=800 experiment raised the fixed candidate ceiling
from 0.731544 to 0.741809 (gain 0.010265, paired 95% interval
[0.009283, 0.011173]). Notebook 06 contains that evidence. It is neither a ranked
Recall@20 score nor the coverage of this compressed candidate pool.

## Progress and recovery

UTC JSONL logs record starts, bucket/iteration progress, heartbeats, failures,
attempt elapsed time and retained computation totals. Local files use atomic
replacement; bucket data is uploaded before its completion receipt. Restoration
checks contract identity, bytes, row counts and SHA-256. Missing/corrupt candidate
buckets are rebuilt independently. Iteration snapshots use the existing ranker's
checksum contract. Finished objective models and evaluations are reused unchanged.
An interruption can repeat the active bucket, uncommitted training interval, or
unfinished objective evaluation; verified completed work remains reusable.

Use one writer per remote experiment namespace. Local filesystem locks do not
claim to be a distributed S3 lease. Model/feature contracts, data checksums, source
hashes and relevant runtime versions define identity; report-only commits do not
invalidate the completed observed-feature cache.

| Artifact | Location |
|---|---|
| Candidate buckets and receipts | `data/interim/ranking_candidates/parts` |
| Candidate log | `data/interim/ranking_candidates/logs/ranking_candidates.jsonl` |
| Models, snapshots and metrics | `artifacts/ranking` |
| Pipeline log | `artifacts/ranking/logs/ranking.jsonl` |
| Compact measured report | `reports/metrics/ranking_evaluation.json` |
| Executed results notebook | `notebooks/08_ranking_evaluation.ipynb` |
| Durable candidates | `<checkpoint-uri>/candidates/<candidate-id>` |
| Durable model run and notebook | `<checkpoint-uri>/models/<run-id>` |

`project_status.py` reports a ranking score only when a valid measured report is
present. Generated local/S3 reports are not automatically committed to GitHub;
review and publish the actual outputs through a results PR after the run.

## Remaining experiments and submission

Measure this baseline before selecting more complex rankers. Add paired session
confidence intervals, candidate-budget/source/feature ablations, and certified
neural fits with independent inner checkpoint selection. Preserve temporal fit
cutoffs for learned representations and co-visitation statistics. Do not call a
previously explored window an untouched test set.

Then implement full-test candidate generation and prediction, validate complete
session/objective coverage, exactly 20 unique integer IDs per row, deterministic
ordering and hashes, and check current Kaggle submission availability/rules.
A generated submission file and an accepted Kaggle submission are distinct
milestones; neither is claimed by this implementation.
