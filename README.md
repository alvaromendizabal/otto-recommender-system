# OTTO Multi-Objective Recommender System

Session-based recommendations for the next **click**, **cart**, and **order**.
Co-visitation and Item2Vec feed a task-specific LambdaRank baseline; a separate
objective-conditioned two-tower experiment evaluates neural candidate discovery
and FAISS search.

**Measured baseline:** weighted Recall@20 **0.497317**, versus **0.373086** for
source-agreement/reciprocal-rank fusion on the same 100-candidate pool and 103,468
outer sessions. All three rankers and the results notebook completed in SageMaker
on 2026-09-08 UTC. This is **exploratory Fold 0 validation**, not an untouched
holdout, leaderboard result, or state-of-the-art claim. Broad feature research
and Kaggle submission remain unfinished.

## Start here

| Review | Executed notebook | Evidence |
|---|---|---|
| Ranking quality and next decision | [08 — Ranking evaluation](notebooks/08_ranking_evaluation.ipynb) | Matched scores, candidate ceilings, selection iterations and compute cost |
| Neural search trade-offs | [06 — ANN benchmark](notebooks/06_ann_benchmark.ipynb) | Exact/approximate fidelity, paired intervals, latency and source coverage |
| Feature and split integrity | [07 — Ranking features](notebooks/07_ranking_features.ipynb) | Observed-only features, complete query denominators, independent audit and resume proof |

No dataset download, AWS account or model run is needed to review the notebooks.
Supporting analyses cover [validation](notebooks/01_validation_protocol.ipynb),
[retrieval benchmarks](notebooks/02_retrieval_benchmarks.ipynb),
[candidate budgets](notebooks/03_candidate_frontier.ipynb),
[hard negatives](notebooks/04_hard_negative_quality.ipynb), and
[two-tower results](notebooks/05_two_tower_results.ipynb).

## Actual matched results

The official metric is `0.10*Recall@20(clicks) + 0.30*Recall@20(carts) +
0.60*Recall@20(orders)`. Pool per-session hits and true-item denominators capped
at 20 within each objective, then apply the weights. Queries with no candidates
are not dropped from the denominator.

| Objective | Matched baseline | LambdaRank | Candidate ceiling |
|---|---:|---:|---:|
| Clicks | 0.378017 | **0.520709** | 0.655426 |
| Carts | 0.267818 | **0.374504** | 0.463247 |
| Orders | 0.424899 | **0.554825** | 0.605341 |
| Weighted | 0.373086 | **0.497317** | 0.567721 |

The absolute improvement is **12.423 percentage points** on this matched
comparison. Candidate ceilings describe recoverable items, not achieved ranked
quality. The first model uses **30 features**: source evidence, source agreement,
observed session context and observed item repeat/recency signals. The schema's
three absent neural-source columns were excluded; no broad learned screening or
feature-family ablation is claimed.

[Measured report](reports/metrics/ranking_evaluation.json) ·
[Source receipts and verification scope](reports/metrics/ranking_evaluation_provenance.json)

The earlier all-fold revisit/time benchmark (0.549644) uses a different cohort
and candidate policy and is not a matched comparison with this result. Likewise,
the earlier **0.731544 → 0.741809** uncompressed ANN candidate-ceiling experiment
is not the coverage or ranked score of this compressed pool. Notebook 06 retains
those original measurements and their qualifications.

## Engineering evidence

The observed-feature cache covers **515,702 sessions**, **1,544,172 session/item
rows**, and **1,547,106 queries**, with 32 independently audited feature buckets.
Candidate materialization produced **51,570,200 rows per objective**, also in 32
resumable buckets. The uploaded restart logs reused all feature and candidate
buckets without recomputing them.

The ranker enforces disjoint fit/inner/outer session IDs, complete query groups,
explicit feature order, deterministic ties, inner-only checkpoint selection and
checksum-verified iteration recovery. Objective models and evaluation receipts
remain in S3. The completed run retained **3,145.965 seconds** of fitting; its
end-to-end pipeline took **8,102.398 seconds**, including candidate preparation
and publication. These are not serving-latency measurements.

UTC logs, stage/total timing and resource heartbeats expose progress. A duplicate
CLI request is rejected before shared-log or candidate writes; it does not delete
a lock or displace an active model. Tests include real process-lock ownership,
corruption, absent-workspace restoration and verified reuse. The code lives in
`src/otto_recsys/` and `gpu/two_tower/`; notebooks are analytical views.

## Observe without restarting

From an existing Studio checkout:

```bash
.venv/bin/python scripts/run_ranking.py --stage status
```

Add `--watch-seconds 300` for bounded read-only observation. This command does
not start training or call AWS. A lock filename alone does not imply an active
writer; the status code probes the kernel lock and verifies saved evidence.
The completed baseline does not need another training launch.

## Reproduce and publish notebook outputs

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
.venv/bin/python scripts/project_status.py
```

CI executes every canonical notebook in its pinned, isolated analysis kernel
and checks reuse. On `results/` branches only, after all quality jobs pass, a
restricted publication job verifies source cells, input/runtime fingerprints
and output checksums, then commits only canonical notebook outputs and
`notebooks/execution.json`. It refuses to overwrite a concurrently advanced
branch. A results PR must still pass checks before merging into `main`.
Executed notebooks are therefore versioned in Git rather than available only
in an expiring CI archive. No model training occurs during notebook replay.

## Next research phase

The 100-item pool limits this baseline to a weighted ceiling of 0.567721.
Additional ranking features cannot recover excluded candidates. Test candidate
budgets and source coverage together with broad, domain-informed feature
engineering: recency/repeat intent, action-conditioned transitions, source-score
normalization and interactions, and historical context with certified cutoffs.
Stream candidate features; screen only on training sessions; validate selected
families through controlled ablations. Hundreds or thousands of generated
features are not evidence of utility without that selection and evaluation.

The frozen label cache lacks future-label timestamps and certified upstream
retriever fit provenance. The existing neural checkpoint was selected on Fold 0.
The current results must remain exploratory. Paired ranker uncertainty, broad
feature discovery/screening, certified neural-source comparisons, an untouched
temporal evaluation, full-test prediction, submission validation and Kaggle
acceptance are separate uncompleted milestones.

[Ranking methodology and execution](docs/RANKING.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Durability](docs/DURABILITY.md) ·
[Managed training](docs/TWO_TOWER_PIPELINE.md)
