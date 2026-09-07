# OTTO Multi-Objective Recommender System

Session-based recommendations for the next **click**, **cart**, and **order**.
The project combines co-visitation, Item2Vec, an objective-conditioned two-tower
retriever, FAISS search, and a tested LambdaRank training engine.

**Status:** retrieval, ANN benchmarking, and observed ranking-feature preparation
are executed and audited. Full-data learned ranking and a Kaggle submission are
not yet complete. The existing neural Fold 0 results are exploratory because
that fold also selected the checkpoint.

## Start here

| Review | Executed notebook | What to look for |
|---|---|---|
| Recommendation quality | [02 — Retrieval benchmarks](notebooks/02_retrieval_benchmarks.ipynb) | Ordered top-20 results and retrieval ablations |
| Neural search trade-offs | [06 — ANN benchmark](notebooks/06_ann_benchmark.ipynb) | Exact versus approximate search, confidence intervals, measured latency |
| Feature and split integrity | [07 — Ranking features](notebooks/07_ranking_features.ipynb) | Observed-only features, nested session assignments, full-data audit, resume proof |

These notebooks read compact committed evidence; reviewing them requires no
AWS account, dataset download, or training job. Supporting notebooks cover the
[validation protocol](notebooks/01_validation_protocol.ipynb),
[candidate frontier](notebooks/03_candidate_frontier.ipynb),
[hard-negative quality](notebooks/04_hard_negative_quality.ipynb), and
[two-tower results](notebooks/05_two_tower_results.ipynb).

## Measured evidence

The official metric is:

```text
0.10 × Recall@20(clicks) + 0.30 × Recall@20(carts) + 0.60 × Recall@20(orders)
```

Recall aggregates session-level positive hits and true-item denominators capped
at 20 within each objective. A larger candidate pool's ideal top-20 ceiling is
reported separately from the quality of an actual ordered top-20 list.

| Measurement | Result | Scope |
|---|---:|---|
| Revisit + time co-visitation ordered Recall@20 | **0.549644** | Earlier all-fold retrieval benchmark |
| Exact neural ordered Recall@20 | **0.218670** | Exploratory Fold 0; 103,468 sessions |
| ANN neural ordered Recall@20 | **0.217494** | Same Fold 0 cohort |
| Baseline → baseline + ANN candidate ceiling | **0.731544 → 0.741809** | Same Fold 0 cohort; neural K=800 |
| Incremental candidate ceiling, paired 95% interval | **+0.928 to +1.117 percentage points** | Session bootstrap; not a final ranking gain |
| Warm CPU ANN search + reranking speedup | **5.35–5.63×** | Matched tuning queries; excludes encoder, network, and loading |

The earlier all-fold benchmark and Fold 0 are different cohorts. Their scores
must not be treated as a matched model comparison. The neural model contributes
additional candidates; it has not been shown to outperform the ranking baseline.

![ANN quality and search cost](reports/figures/two_tower_ann_quality.png)

Source reports and independent audits are linked from notebooks 02 and 06.
Detailed qualifications and execution history remain in
[the ANN benchmark](docs/ANN_BENCHMARK.md) and
[the retrieval evaluation](docs/FOLD_EVALUATION.md).

## Engineering evidence

Observed ranking features cover **515,702 sessions**, **1,544,172 session/item
rows**, and **1,547,106 evaluation queries**. All **32** feature buckets were
reused on a verified local restart. The independent full-data audit found zero
session, item, query, or duplicate-key mismatches. Features, receipts, and logs
are durable in S3; notebook 07 contains the committed evidence.

The [LambdaRank engine](src/otto_recsys/ranking/lambdarank.py) enforces disjoint
fit/inner/outer session IDs, complete query denominators, deterministic ties,
explicit model feature order, and inner-only early stopping. It checkpoints
model state and selection history atomically, verifies resume identity, and
falls back from corrupt checkpoints. The [engine tests](tests/test_lambdarank.py)
cover interruption equivalence, checkpoint corruption, concurrent writers,
leakage-column rejection, and metric edge cases. These are synthetic engineering
tests, **not** full-data OTTO ranker results. S3 restoration/publication for this
engine still requires its candidate/training orchestration integration.

Production logic lives in `src/otto_recsys/` and `gpu/two_tower/`; notebooks are
analytical views. Existing workflows use UTC logs, stage and total timings,
resource heartbeats, content hashes, and verified resumable artifacts. Source,
tests, notebooks, and compact results are versioned in GitHub; large data and
models remain in S3. Canonical files are updated in place.

## Reproduce the checks

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
.venv/bin/python -m pytest -q tests/test_lambdarank.py -W error
uv pip check --python .venv/bin/python
git diff --check
.venv/bin/python scripts/project_status.py
```

These commands check the code and published milestones; they do not start paid
training. The status command does not query live AWS jobs. Existing artifacts
should be restored and verified rather than recomputed because a report changed.

## Remaining modeling work

Materialize complete, label-blind baseline candidate features; connect the
ranker engine to durable per-fit orchestration; then execute matched baseline
and neural-source experiments under the documented split/provenance rules.
Publish the actual ranking results and ablations in an executed notebook before
fitting full-test predictions and validating a submission file. An accepted
Kaggle submission is a separate milestone from generating that file.

The frozen labels cache does not retain future-label timestamps, and the current
neural checkpoint was selected on Fold 0. Neither the existing fold nor a new
nested session assignment is an untouched temporal test set. Those limitations
must remain visible in subsequent results.

[Ranking methodology and integration](docs/RANKING.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Durability](docs/DURABILITY.md) ·
[Managed training](docs/TWO_TOWER_PIPELINE.md)
