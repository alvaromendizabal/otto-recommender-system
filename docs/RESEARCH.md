# Controlled feature research

The completed 30-feature Fold 0 ranker is an exploratory result. This study
measures a broader feature space using explicitly chronological inputs. Its
protocol is frozen in `configs/research.toml` before feature screening or model
selection. Original OTTO data has appeared in earlier experiments; this study
does **not** claim that the underlying dataset has never been inspected.

## Availability and evaluation

| Component | Available data, UTC | Selection rule |
|---|---|---|
| Historical retrieval and item statistics | Events before 2022-08-20 22:00 | No query labels |
| Ranker fitting and feature screening | Sessions starting August 20 22:00–August 23 22:00 | 100,000 deterministic session hashes |
| Model and candidate-budget selection | Sessions starting August 23 22:00–August 24 22:00 | 20,000 deterministic session hashes |
| Reserved temporal evaluation | Sessions starting August 24 22:00–August 26 22:00 | All eligible sessions |

Intervals are left inclusive and right exclusive. A session belongs to the
interval containing its first event; its events are clipped at that interval's
end. Sessions with fewer than two remaining events cannot form a prefix/future
query. Every eligible session receives a deterministic, label-independent
prefix cut. The next click and unique future carts/orders form the labels.
Future timestamps and event indices are retained and audited. Catalog filtering
never removes unseen ground-truth items. The complete query ledger preserves
zero-candidate queries and the official capped denominators.

All retrievers are refitted on the historical interval. Older co-visitation,
Item2Vec and neural artifacts are not inputs to this controlled study. Existing
neural experiments retain their original measured scope in notebook 06.

## Resumable execution

After obtaining the official data and running the repository's conversion
workflow, the research entry point accepts a directory of `part-*.parquet`
files with the canonical five event columns:

```bash
.venv/bin/python scripts/run_research.py --stage prepare --source data/processed/train
.venv/bin/python scripts/run_research.py --stage retrieval
```

Preparation hashes every source partition and retains the conversion manifest's
raw-source identity. This is not a claim of independently replaying the full
raw-to-Parquet conversion. Completed outputs and graph partitions have atomic
SHA-256 receipts. Changed protocols or fitted-history contracts are rejected;
corrupt partitions are rebuilt while intact completed partitions are reused.
A kernel lock protects each workspace. UTC heartbeats report elapsed time,
memory, CPU and current stage every 15 seconds.

The historical graph retains at most 30 events per session and pairs within
three positions and 24 hours. Time decay and position distance weight pairs;
per-session maxima limit repeat inflation. Three channels emphasize general
transitions, cart intent and purchase intent. Each source item retains the union
of its best 40 neighbors per channel. Item counts use seven historical windows
plus the entire available history. All windows end at the fixed history cutoff.

## Completion contract

The implemented catalog contains **1,482 candidate features**:

| Family | Candidate features | Mechanism |
|---|---:|---|
| Historical graph affinity | 1,152 | Three action channels, four observed action filters, six prefix lengths, four weighting rules and four aggregates |
| Candidate repeat intent | 160 | Recurrence, observed position, spans, ages and time decay |
| Historical item context | 99 | Counts, window shares, rate trends, action ratios and availability |
| Retrieval evidence | 36 | Source scores/ranks, agreement, dispersion and normalized interactions |
| Session context | 24 | Activity, diversity, duration, gaps and cyclical time |
| Intent interactions | 11 | Candidate affinity combined with session intent and historical conversion |

The first broad matrix contains 514,013 candidate rows from 8,448 fitting
sessions. The 500,000-row screening budget is exceeded only to preserve complete
query batches. All discovered positives and a deterministic mixture of hard and
random negatives are retained for fitting. Evaluation uses complete candidate
pools. Normalized features are computed before fitting rows are sampled.

Quality screening rejects constant, near-constant, duplicated and target-equivalent
columns. Three session-grouped fitting folds train separate objective pilots;
every eligible feature is considered. Weighted gain and fold stability guide
selection, followed by correlation pruning and a cap of 128 retained columns.
All rejection reasons and pilot diagnostics are recorded. These pilot diagnostics
use sampled fitting negatives and are not final ranking results.

```bash
.venv/bin/python scripts/run_research.py --stage features
.venv/bin/python scripts/run_research.py --stage screen
```

The research phase closes when the repository contains measured candidate-budget
coverage, a generated feature catalog, training-only screening with rejection
reasons, matched family ablations, a sealed final model choice, complete temporal
evaluation with paired uncertainty, cost/latency measurements, and an executed
analysis explaining both gains and limitations. Feature count alone is not a
quality result. No unfinished experiment is presented as measured evidence.

The design draws on the official [OTTO task and evaluation specification](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md)
and the documented [third-place feature approach](https://github.com/TheoViel/kaggle_otto_rs).
Reported competition scores use different cohorts and are not direct comparisons
with this study's temporal evaluation.
