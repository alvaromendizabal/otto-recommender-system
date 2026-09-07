# From audited retrieval to learned ranking

**Status:** observed feature preparation is complete and independently audited.
The integrated candidate/ranker pipeline and measured-results notebook publisher
are implemented. Full-data LambdaRank evaluation and Kaggle submission remain
unmeasured. Synthetic integration results are engineering tests, not OTTO scores.

## Run the next experiment

From the existing SageMaker Studio checkout and locked CPU environment:

```bash
.venv/bin/python scripts/run_ranking.py \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking \
  --publish-report --execute-notebooks
```

The canonical command first checks the frozen upstream files and restores the
published observed-feature cache automatically when it is missing or incomplete.
It then prepares candidates, trains the click/cart/order rankers, evaluates their
complete outer queries, saves metrics and generates/executes
`notebooks/08_ranking_evaluation.ipynb`. Notebook 08 is created only when a measured
ranking report exists, never as a placeholder or synthetic performance claim.
Its executed output and receipt are uploaded with the model run.

To check and restore prerequisites without starting candidate work or training:

```bash
.venv/bin/python scripts/run_ranking.py --stage preflight \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking
```

The command uses the already-running CPU workspace and existing S3 storage. It
creates no new SageMaker job, instance, endpoint or IAM resource. Ordinary
workspace and S3 charges apply. Do not repeat completed retriever training,
ANN benchmarking or observed-feature computation to resolve a missing local file.

## Published evidence versus local readiness

`project_status.py` describes versioned experimental evidence. A published stage
can be complete even when its large files are absent from a particular workspace.
`run_ranking.py` now checks readiness separately, before candidate materialization.

The earlier startup failure for
`data/interim/ranking_features/feature_contract.json` was a missing prerequisite
restoration step, not a failed model fit. The published contract, summary,
independent audit and exact S3 URI are read from `reports/metrics`. Together they
pin the content identity and checksum inventory; the runner never guesses a
latest run or rebuilds features to match a different environment.

Recovery uses authenticated S3 reads only for the source feature namespace.
All receipt hashes must match the audited aggregate before data installation.
Every downloaded Parquet file is checked for SHA-256, byte size and row count;
its bucket receipt is installed last. Correct existing files retain their bytes
and modification times. A fully restored cache needs no further S3 reads.
Interrupted transfers retain previously verified buckets and files. An incompatible
local experiment is rejected without replacing it. Original feature manifests,
code dependency hashes and completed calculation times remain unchanged.

UTC start/progress/15-second heartbeat/completion events expose restoration and
checksum verification. `artifacts/ranking/input_readiness.json` records the source
identity, verified/reused/restored buckets, downloaded files/bytes, timings and
`feature_computation_performed=false`. Candidate runs publish this receipt with
their outputs. A preflight-only run retains the receipt locally.

## Required frozen upstream inputs

| Default location | Required artifact |
|---|---|
| `data/interim/ranking_training_cache` | Manifest, examples, observed items and labels |
| `data/interim/covisit` | Time/type/buy matrices and their manifests |
| `models/item2vec/item_vectors.kv` | Item2Vec vectors, sidecars and manifest |
| `models/faiss/item.index` | Baseline ANN index and manifest |
| `data/interim/ranking_features` | Restored automatically from the audited publication |

The preflight reports all absent upstream paths together. Large retriever inputs
are not silently regenerated or downloaded from guessed locations; supply their
explicit CLI paths when using a different existing layout. Checksums tie the
training examples and labels to the published observed-feature contract.
`--feature-evidence` selects another explicit publication directory; it must
contain a mutually consistent contract, summary, audit and publication receipt.

The unchanged observed-feature publication contains 515,702 sessions,
1,544,172 session/item rows and 1,547,106 queries in 32 committed buckets.
Notebook 07 includes the independent zero-mismatch audit. The exact S3 location is:

```text
s3://otto-recsys-560403859723-us-west-2/ranking/features/82e8eac76c63d4d8a34b611bca0f3ae329623ff5cd80e18ca8bc238ddbd65795/
```

## Candidate and model protocol

Revisit, time/type/buy co-visitation and Item2Vec supply candidates without using
future labels to select membership. Deduplicate by session/objective/item and
retain each source's presence, rank and score. The first baseline compresses to
100 candidates using source agreement, reciprocal-rank sum, Item2Vec score and
ascending item ID. This budget is fixed, not selected as an optimum; its coverage
must be measured rather than borrowed from an uncompressed pool.

Complete 128-session groups stream through the observed-feature join into float32
objective Parquet parts. Features include source evidence, session event counts
and duration, repeat/type counts, observed item recency and event share. Missing
scores/ranks and unseen-item recency remain explicit missing values. IDs, targets,
folds and inner assignments are excluded from the feature matrix. The full query
ledger retains zero-candidate queries; missing positives remain misses.

Default execution evaluates outer Fold 0. Fits use other-fold sessions outside
inner partition 0; early stopping uses only inner partition 0 in those other
folds. Outer sessions enter neither fitting nor checkpoint selection. Learned and
source-agreement/RRF baseline rankings use the identical candidate pool and
complete denominators. `--outer-folds 0 1 2 3 4` fits all five outer folds.

The official metric is `0.10*Recall@20(clicks) + 0.30*Recall@20(carts) +
0.60*Recall@20(orders)`. Cap each session's hits and true-item denominator at 20,
pool objective numerators/denominators across folds, then apply the weights.
Report NDCG@20, MRR@20, hit rate, candidate ceiling and timing separately.
Outer evaluation time includes I/O, prediction, sorting and metrics, not serving
latency. The report publisher checks aggregate scores against their fold counts.

**Evaluation limits:** the frozen cache lacks future-label timestamps and
certified upstream fit provenance. This is exploratory nested session validation,
not an untouched temporal test or fully certified nested retrieval/ranking.
The earlier Fold 0 neural checkpoint was selected on that fold; this baseline
runner refuses neural-candidate contracts rather than relabeling it independent.

## Resources and recovery

Defaults use four CPU threads, 4 GB DuckDB memory and a 20 GiB estimated training
budget, also bounded by available RAM. The estimate is not a guaranteed maximum
for native allocations. Oversized fits fail before allocation; rows are never
silently sampled. Native training datasets are released before outer evaluation.

`--stage candidates` stops after preparation; `--stage train` uses a verified
local candidate cache. Changed inputs/features/model settings or outer folds
require a separate output directory. A changed candidate budget also requires
a separate `--candidate-dir`. Preserve old experiments rather than overwrite them.

Candidate data is uploaded before completion receipts. Ranker checkpoints and
completed objective model/evaluation receipts are verified on reuse. An interrupted
active bucket, uncommitted training interval or unfinished objective evaluation
may repeat, but verified completed work survives. Use one writer per remote run;
workspace filesystem locks do not claim to implement distributed S3 leases.

| Evidence | Location |
|---|---|
| Input readiness | `artifacts/ranking/input_readiness.json` |
| Candidate buckets | `data/interim/ranking_candidates/parts` |
| Candidate progress | `data/interim/ranking_candidates/logs/ranking_candidates.jsonl` |
| Model/checkpoints/metrics | `artifacts/ranking` |
| Overall progress and restoration | `artifacts/ranking/logs/ranking.jsonl` |
| Compact measured ranking report | `reports/metrics/ranking_evaluation.json` |
| Executed ranking notebook | `notebooks/08_ranking_evaluation.ipynb` |
| Durable outputs | `<checkpoint-uri>/candidates/<id>` and `/models/<id>` |

Generated results are local and in S3 until reviewed and committed through a
results PR. Neither a model score nor a completed submission is inferred from
passing engineering tests.

## Next modeling decisions

The saved feature audit shows 232,838 one-event prefixes out of 515,702 sessions.
Prioritize short-prefix diagnostics and reliable item/source context rather than
assuming a longer sequence model helps every query. Test recency and repeat intent,
objective-conditioned transitions, source-score normalization and interactions,
and time-windowed item trends only when their fit cutoffs can be certified.
These are candidate feature families, not demonstrated performance improvements.

First measure the matched ranker baseline. Then compare candidate budgets and
feature/source ablations using inner validation, plus paired session uncertainty
on fixed outer predictions. Certify neural-source fitting and checkpoint selection
before adding those sources to an independently evaluated ranker. More features
or a more complicated model are not evidence of better recommendations.

The earlier ANN K=800 experiment increased an uncompressed candidate ceiling from
0.731544 to 0.741809, not achieved ranked Recall@20. Preserve this distinction.
Full-test candidate generation/prediction, complete submission validation and
current Kaggle submission availability must be checked separately. Generating a
submission file and obtaining an accepted Kaggle submission are distinct milestones.
