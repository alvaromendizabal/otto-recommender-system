# From audited retrieval to learned ranking

Status: **implementation plan; no ranker has been trained or evaluated yet.**
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

## Current command

```bash
.venv/bin/python scripts/project_status.py
```

This reports the completed evidence and the next implementation task. It does
not start cloud compute. There is no ranking launcher to run yet; implementing
the frozen split and candidate/feature pipeline is the next development task.
