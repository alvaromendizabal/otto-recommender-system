# From audited retrieval to learned ranking

Status: **observed feature preparation and explicit nested session assignments implemented
and executed; no ranker has been trained or evaluated yet.**
The retriever, full-catalogue exports, ANN fidelity/search-cost benchmark, and
Fold 0 baseline comparisons are complete. Reuse their saved artifacts.

## Decision supported by the completed experiment

At neural K=800, ANN raises the fixed baseline candidate ceiling from 0.731544
to 0.741809, a gain of 0.010265 (95% paired session interval
[0.009283, 0.011173]). Exact search added 0.010291. This supports including the
ANN source in the next ranking experiment. It does not prove that a learned
ranker will realize the extra coverage, that the two methods are equivalent,
or that this configuration is competitive on the Kaggle leaderboard.

The underlying reports, audits, and charts are in notebook 06. The official
metric is 0.10 × clicks Recall@20 + 0.30 × carts Recall@20 + 0.60 × orders
Recall@20, with per-session positive hits and true-item denominators capped at
20 and aggregated within each objective. Candidate ceilings are reported
separately from actual ordered top-20 performance.

## 1. Freeze validation before more training

Create a versioned split manifest with session IDs, event cutoffs, label
windows, training partitions, inner checkpoint-selection partitions, outer
evaluation partitions, and hashes. Sessions must not cross training and
evaluation within a fit. Time-based eligibility applies to feature data and
learned preprocessing as well as model labels.

Fold 0 selected the current neural checkpoint. Keep its published scores
explicitly exploratory. Each future outer fold must receive predictions from
a retriever trained without its labels and selected using an inner partition.
Simply training the other four current folds does not correct checkpoint
selection on the held-out fold. Hard-negative generation must obey the same
fit boundaries. Frozen unsupervised item representations and co-visitation
graphs need documented fit cutoffs and availability assumptions.

Identify a labeled temporal holdout not used in previous development before
claiming an untouched final evaluation. If none remains in the available data,
report nested validation and its history honestly; do not rename an already
explored window as an untouched test set. Kaggle test labels remain unavailable.

Required tests: disjoint sessions; chronological eligibility; unchanged split
hash on rerun; no outer labels in fitting or checkpoint selection; incompatible
split manifests rejected before computation.

## 2. Build candidates and features in resumable partitions

Keep one row per `(session, objective, aid)` and a deterministic query group.
Start with the measured revisit/co-visitation/Item2Vec pool and neural K=800.
Deduplicate by item while retaining each source's rank, score, and presence.
Compute source rank/score interactions, number of agreeing sources, session
event counts and duration, item repeat/type counts, and item/session recency
using observed events only. Record units, missing-value meanings, and feature
availability in a feature manifest. Evaluate further features by ablation.

Generate candidates without evaluation labels. Missing true positives must
remain misses in the denominator; never insert them into evaluation candidates.
If training uses forced positives or negative sampling, label that policy and
keep it out of evaluation. Score the complete held-out candidate groups.

Partition by fold, objective, and session bucket with a bounded memory budget.
Write data atomically, verify its checksum, upload it, then publish the receipt.
The contract must include source and dependency hashes, input checksums,
split/feature schemas, candidate budgets, seeds, and sampling policy. Reruns
skip only matching verified parts. A hard interruption can repeat the active
part; previously committed parts survive in S3.

Required tests: duplicate prevention, deterministic ties, feature cutoff
boundaries, label-blind candidate generation, full denominator preservation,
bounded partition sizes, corrupt/missing receipts, and recovery in a fresh
workspace without rebuilding valid parts.

## 3. Train and compare objective-specific rankers

Use the existing locked LightGBM stack for the first LambdaRank baseline,
grouped by session/objective. Keep validation groups complete and use the
inner split for early stopping. Save model, feature order, fit IDs, selected
iteration, parameters, seeds, and source/runtime versions for every fit.

Compare matched experiments: existing ranking baseline; learned ranker without
neural candidates; the same ranker with neural candidates. Then evaluate
candidate compression budgets and feature/source ablations. Hyperparameter
search must use only inner validation, with a stated compute budget. More
complex models need demonstrated gains over this measured baseline.

Report official weighted Recall@20 and each objective's Recall@20. Include
NDCG@20, MRR@20, candidate coverage, paired session intervals, fit/prediction
time, peak memory, and end-to-end latency measured separately from warm index
search. Aggregate the official numerators/denominators across outer folds;
do not substitute an unweighted average of fold scores.

Long stages must emit UTC start, progress, heartbeat, failure, and completion
events with attempt elapsed time and retained-work totals. CPU telemetry uses
100% per occupied core and may exceed 100% for parallel kernels. Persist logs
and model checkpoints to the experiment's durable S3 namespace.

## 4. Validate a submission and publish evidence

After selecting the configuration under the frozen protocol, fit on eligible
training data and generate the competition's required session/objective rows.
Test schema, complete test-session coverage, objective suffixes, exactly 20
unique integer item IDs per row, deterministic ordering, and artifact hashes.
Confirm current submission availability and competition rules before upload.

Publish executed analytical notebooks, model/feature descriptions, evaluation
limits, ablations, and compact metrics through a documented PR with passing CI.
Keep large data and model weights in durable artifact storage. A generated
submission file and an accepted Kaggle submission are separate milestones.

## Completed feature preparation

The full frozen cache has been prepared and independently reconciled:

| Artifact | Rows |
|---|---:|
| Observed session features | 515,702 |
| Observed session/item features | 1,544,172 |
| Evaluation query ledger | 1,547,106 |

There are 32 checksum-committed buckets and 28.53 MiB of Parquet data.
The full preparation took 10.279 seconds in the recorded CPU environment;
a local verified resume reused all 32 buckets in 2.623 seconds. These exclude
initial input download and S3 publication. The independent DuckDB audit found
zero session, item, query, or duplicate-key mismatches. Notebook 07 is executed
and includes the exact split counts, observed-prefix distribution and timings.

`ranking/features.py` provides observed feature aggregation, a full query ledger,
and a label-blind candidate join preserving each source's presence/rank/score.
It never inserts label items into candidate membership. `ranking/splits.py`
assigns independent inner partitions and resolves disjoint fit/inner/outer roles.
The feature manifest explicitly records that future-label timestamps are missing
from the existing cache and that retriever fit provenance is not yet certified.
This is not an untouched temporal holdout or a completed nested-model evaluation.

`ranking/feature_cache.py` verifies input content hashes and frozen fold/bucket
assignments, writes bounded bucket files, commits checksum receipts last, rejects
incompatible contracts, and locks against concurrent writers in one workspace.
The AWS checkpoint adapter uploads data before receipts and restores verified
parts into fresh workspaces. Missing/corrupt buckets are rebuilt; intact buckets
retain their computation. Use one writer per remote experiment namespace.
Feature code, input hashes, split seeds and runtime versions define the identity;
report-only Git commits do not invalidate saved feature work.

The complete artifacts live at:

```text
s3://otto-recsys-560403859723-us-west-2/ranking/features/82e8eac76c63d4d8a34b611bca0f3ae329623ff5cd80e18ca8bc238ddbd65795/
```

## What to do now

Open `notebooks/07_ranking_features.ipynb` to view the executed outputs and run
`scripts/project_status.py` to check the published milestones. Preparation is
already complete; there is no need to repeat ANN benchmarks or training.

Next development: materialize the baseline candidate pool using these features,
then train/evaluate the first objective-specific LambdaRank models. Add neural
features with enforced retriever fit/checkpoint-selection provenance; the
existing exploratory Fold 0 checkpoint cannot be relabeled as independently
selected. Full-test prediction and submission remain later milestones.

## Reproduce or restore the feature preparation

From the locked project environment with the existing frozen ranking cache:

```bash
.venv/bin/python scripts/build_ranking_features.py \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking/features
.venv/bin/python scripts/audit_ranking_features.py
```

The first command restores compatible S3 buckets before doing computation.
The CLI requires a durable S3 destination and emits UTC start, bucket progress,
CPU/RAM heartbeats, failure and total-time events. Output defaults to
`data/interim/ranking_features`; logs are in its `logs/ranking_features.jsonl`.
A changed contract requires a separate experiment output directory, preserving
previous results. The audit uses independent DuckDB aggregations of all frozen
events and labels. The analytical notebook reads the compact committed evidence
and does not require downloading feature data or starting cloud jobs.

LightGBM expects contiguous groups whose sizes sum to the training row count;
keep `(objective, session)` groups complete when materializing ranked candidates.
See the [official LGBMRanker documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html).
