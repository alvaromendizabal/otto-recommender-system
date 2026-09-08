# Ranking evidence and research protocol

## Completed baseline

The existing Studio run completed all three objective fits, full outer-query
evaluations and notebook publication on **2026-09-08 at 02:16:05 UTC**. Run identity:

```text
b2804b2a1495d60f7e9286dda93dd9c493eafd982bed7c41af6b08b7e904e220
```

The matched weighted Recall@20 increased from **0.3730864841** to
**0.4973169468** on 103,468 Fold 0 outer sessions. The 100-candidate pool's
weighted coverage ceiling is **0.5677209482**. Notebook 08 contains scores,
NDCG/MRR, per-objective ceilings, selection iterations, timings and limitations.
The native models and original executed notebook remain in the durable S3 run.

The publication independently checks saved evaluation JSON bytes against source
receipts, receipt identities, the combined report hash, and pooled score
arithmetic. Native model checksums are recorded from the source receipts;
publication is not a fresh full-data inference replay or independent byte hash
of every native model. See `reports/metrics/ranking_evaluation_provenance.json`.

This is a **30-feature exploratory baseline**, not a completed exhaustive feature
study, certified untouched holdout or Kaggle submission. The earlier all-fold
retrieval score and larger-pool ANN ceilings are different experiments.

## What the repeated lock errors meant

The uploaded attempts reused all 32 observed-feature buckets and all 32 candidate
buckets, then encountered the ranker's output lock. Another run was already
advancing. Its S3 checkpoints progressed through clicks, carts and orders, and
its final log reported `ranking_pipeline_complete` with status `passed`.

Never remove a lock file or kill a process simply because a duplicate launch was
rejected. A filename is not proof of lock ownership, and unlinking a held lock
can permit concurrent writers against different inodes.

The canonical CLI now probes existing ownership before initializing shared logs,
restoring inputs or publishing candidates. A separate admission lock protects
new CLI instances through notebook publication; the unchanged ranker lock still
protects direct and older writers. This is local admission, not a distributed
S3 lease. A legacy writer starting after the probe is still checked at the
ranker's own final lock.

```bash
.venv/bin/python scripts/run_ranking.py --stage status
```

For a bounded observation period, add `--watch-seconds 300`. Status is read-only,
needs no AWS credentials and does not create output folders or logs. It reports
kernel ownership when observable, checksum-verified checkpoint/evaluation state
and the latest meaningful progress record. A partial objective result is not
reported as a final weighted score. Unknown lock permissions remain unknown,
not idle. Duplicate execution returns exit 75 and `OTTO_RANKING_ALREADY_RUNNING`,
not a model-success marker.

## Frozen inputs and durable artifacts

| Default path | Purpose |
|---|---|
| `data/interim/ranking_training_cache` | Frozen manifest, examples, observed items and labels |
| `data/interim/ranking_features` | Audited observed features and complete query ledger |
| `data/interim/covisit` | Saved time/type/buy matrices and manifests |
| `models/item2vec/item_vectors.kv` | Item2Vec vectors, sidecars and manifest |
| `models/faiss/item.index` | Frozen baseline ANN index and manifest |
| `data/interim/ranking_candidates` | Complete candidate features and receipts |
| `artifacts/ranking` | Local contracts, model snapshots, evaluations and logs |

The unchanged feature identity is
`82e8eac76c63d4d8a34b611bca0f3ae329623ff5cd80e18ca8bc238ddbd65795`.
It covers 515,702 sessions, 1,544,172 session/item rows and 1,547,106 queries.
The candidate identity is
`9bd6ecee61ea322e14f4beefc142306683c65f7d12452db30f67ccdebed39548`,
with 51,570,200 candidate rows per objective in 32 buckets.

All namespaces use the existing project bucket:

```text
s3://otto-recsys-560403859723-us-west-2/ranking/features/<feature-id>/
s3://otto-recsys-560403859723-us-west-2/ranking/candidates/<candidate-id>/
s3://otto-recsys-560403859723-us-west-2/ranking/models/<run-id>/
```

The normal CLI preflight restores missing observed features from the exact
published contract/summary/audit/URI, not a guessed latest run. It verifies
receipt inventories, SHA-256, bytes and Parquet row counts. Data is installed
before completion receipts. Correct files and their modification times survive
an interrupted restoration. Original feature computation and manifests are not
regenerated. Other missing upstream paths are reported together rather than
silently rebuilt.

## Reproduction, not a required rerun

The following command reproduces or resumes the existing configuration from
verified artifacts; it is not necessary merely to view the completed baseline:

```bash
.venv/bin/python scripts/run_ranking.py \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking \
  --publish-report --execute-notebooks
```

`--stage preflight` verifies/restores inputs only; `--stage candidates` stops after
candidate preparation; `--stage train` uses an already verified candidate cache.
Changed candidate budgets require a separate `--candidate-dir`; changed input,
feature, model or fold contracts require a separate output directory. Preserve
old content identities rather than replace valid experiments.

Defaults use 100 candidates, complete 128-session join batches, four CPU threads,
4 GB DuckDB memory and a 20 GiB estimated training budget, further bounded by
available RAM. The estimate is not a guaranteed native-memory maximum. Oversized
fits are rejected before allocation and no rows are silently sampled. Native
training datasets are released before outer evaluation.

## Validation and feature boundaries

Candidates use revisit, time/type/buy co-visitation and Item2Vec, without inserting
future positive items. Source agreement, reciprocal-rank sum, Item2Vec score and
ascending item ID define the fixed compression baseline. Score/rank absence
remains explicit missingness. Baseline fitting excludes the schema's three
absent neural columns, IDs, targets, query identities and split assignments.

For each outer fold, other-fold sessions outside inner partition zero fit the
model. Inner partition zero selects stopping iterations. Outer sessions enter
neither ranker fitting nor ranker checkpoint selection. Complete query ledgers
preserve zero-candidate misses. Learned and matched baseline rankings use the
same candidates and full denominators.

The official metric pools capped per-session hits and denominators within each
objective, then applies weights **0.10 clicks / 0.30 carts / 0.60 orders**. Across
folds, pool counts rather than equally average fold scores. NDCG/MRR/hit-rate
average over labeled queries. Candidate ceilings and fit/evaluation durations
are separate diagnostics, not substitutes for ranked quality or serving latency.

The frozen cache does not preserve future-label timestamps or certified upstream
retriever fit provenance. Its window has already been explored. The existing
neural checkpoint was selected on Fold 0. New ranker inner partitions do not
retroactively certify that upstream fit or create an untouched temporal test.
The current baseline therefore does not include two-tower scores as independent
features. These limits remain visible in the report and notebooks.

## Notebook publication

CI executes all canonical analytical notebooks and proves reuse. For pushes to
`results/` branches only, a publication job waits for every quality job, executes
and verifies the notebooks in the isolated kernel, and copies outputs back to
the same filenames. The publisher verifies source cells, input/runtime identity,
receipt checksums, complete execution and absence of notebook error/warning
outputs before any source replacement. It writes `notebooks/execution.json`
last, outside the input evidence tree to avoid self-invalidating identities.

Only canonical notebook paths and that execution receipt may be staged. A
concurrent remote branch advance aborts the push; there is no force push. A
results PR subsequently validates the exact rendered commit before merge. This
retains employer-viewable outputs in Git, not only a 30-day CI download.

## Next research decisions

The first baseline establishes a measured comparison, not feature exhaustion.
Thirty features were used; broad training-only screening and feature-family
ablations have not been performed. Candidate coverage is the immediate ceiling:
more ranking features cannot recover an item discarded by retrieval/compression.

Compare candidate budgets and sources together with broad, domain-informed
families: repeated intent and recency decay, action-conditioned sequential
transitions, score normalization and query-relative ranks, source interactions,
and historical item trends with verifiable availability cutoffs. About 45.15%
of the audited prefixes contain one event, so short-prefix diagnostics are
important alongside longer-sequence representations.

Generate candidates broadly, then stream training-only schema/finite/missingness,
constant/duplicate/redundancy and utility checks before constructing a selected
matrix. Record catalog definitions, fit cutoffs, candidate/retained/rejected
counts and reasons. Evaluate feature groups through controlled inner-selection
ablations and fixed outer comparisons with paired session uncertainty. Do not
claim utility from feature count or split/gain importance alone.

Certified neural-source fitting, an untouched temporal evaluation, full-test
candidate generation/prediction, submission-format validation and current Kaggle
acceptance remain separate milestones. No submission or final competition
performance is claimed by the completed baseline.
