# OTTO Multi-Objective Recommender System

Session-based recommendations for the next **click**, **cart**, and **order**.
The project combines co-visitation, Item2Vec, an objective-conditioned two-tower
retriever, FAISS search, and a resumable LambdaRank pipeline.

**Status:** retrieval, ANN benchmarking, and observed ranking-feature preparation
are executed and audited. Candidate materialization, ranker training/evaluation,
and durable results-notebook publication are implemented. Full-data learned
ranking and a Kaggle submission are not yet complete. The existing neural Fold 0
results are exploratory because that fold also selected the checkpoint.

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
explicit model feature order, and inner-only early stopping. The integrated
[pipeline](scripts/run_ranking.py) streams label-blind candidate features,
trains objective-specific rankers, and compares them with a matched baseline.
It restores checksum-verified S3 buckets, iteration snapshots and completed
objective models. Local locks protect each workspace; use one writer per remote
experiment namespace. Existing observed-feature contract dependencies remain
unchanged, so this implementation does not invalidate completed feature work.

The [engine tests](tests/test_lambdarank.py) and
[integration tests](tests/test_ranking_pipeline.py) cover actual Parquet, DuckDB,
Gensim, FAISS and LightGBM operations, corruption, missing candidates, label-blind
membership, memory guards and fresh-workspace recovery. Synthetic test scores
are **not** full-data OTTO ranking results.

Production logic lives in `src/otto_recsys/` and `gpu/two_tower/`; notebooks are
analytical views. UTC logs, stage/total timings, resource heartbeats and verified
receipts expose progress. Source, tests, notebooks and compact results are
versioned in GitHub; large data and models remain in S3. Canonical files are
updated in place.

## Execute the next ranking experiment

From the existing SageMaker Studio CPU workspace with the frozen inputs:

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_ranking.py \
  --checkpoint-uri s3://otto-recsys-560403859723-us-west-2/ranking \
  --publish-report --execute-notebooks
```

This runs candidate preparation, three rankers and complete-query evaluation.
After measured results exist, it generates and executes the canonical
`notebooks/08_ranking_evaluation.ipynb`, saves the compact report, and uploads
the executed notebook with the model run. It does not start a separate AWS job,
instance or endpoint; ordinary running-workspace and S3 charges still apply.
The default is an exploratory Fold 0 run with a fixed 100-candidate baseline
budget, not an optimized or certified neural-source experiment.

Repeating the command reuses verified completed work. An interruption can repeat
the active bucket, uncommitted training interval or unfinished evaluation, not
all completed stages. The final ranking report/notebook are local and in S3;
they must be reviewed and committed through a results PR after execution.
See [the ranking runbook](docs/RANKING.md) for inputs, memory estimates,
checkpoint paths, split assumptions and remaining experiments.

## Reproduce the checks

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
.venv/bin/python -m pytest -q tests/test_lambdarank.py tests/test_ranking_pipeline.py -W error
uv pip check --python .venv/bin/python
git diff --check
.venv/bin/python scripts/project_status.py
```

These commands check the code and published milestones; they do not start
full-data training. The status command does not query live AWS jobs. Existing
artifacts should be restored and verified rather than recomputed because a
report changed.

### Replay analytical notebooks only

From the repository root in SageMaker Studio or Linux, use the separate analysis
kernel rather than changing the model-training environment:

```bash
uv venv "$HOME/.venvs/otto-notebooks" --python 3.12.13 --no-project
uv pip install --python "$HOME/.venvs/otto-notebooks/bin/python" -r notebooks/requirements.txt
"$HOME/.venvs/otto-notebooks/bin/python" scripts/execute_notebooks.py
```

The last command executes the available analytical notebooks with their committed
evidence, without fetching training data or starting cloud jobs. Results retain
canonical filenames under `artifacts/notebooks/`; committed originals remain
unchanged. Repeating the last command verifies and reuses successful outputs.
UTC logs, per-notebook receipts and the final manifest are in the same directory.
CI executes the notebooks and their reuse check on every change. Private filesystem
sockets keep kernels off TCP; notebook and kernel warnings fail the execution
gate rather than being hidden. CI output archives expire after 30 days; canonical
notebooks and compact evidence remain versioned in the repository.

## Remaining modeling work

Execute the integrated baseline on actual OTTO data. Then measure paired session
confidence intervals, candidate-budget and feature ablations, and certified
neural-source experiments under the documented split/provenance rules. Publish
actual ranking results before full-test prediction and submission validation.
An accepted Kaggle submission is distinct from generating a submission file.

The frozen labels cache does not retain future-label timestamps, and the current
neural checkpoint was selected on Fold 0. Neither the existing fold nor a new
nested session assignment is an untouched temporal test set. Those limitations
must remain visible in subsequent results.

[Ranking methodology and integration](docs/RANKING.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) ·
[Durability](docs/DURABILITY.md) ·
[Managed training](docs/TWO_TOWER_PIPELINE.md)
