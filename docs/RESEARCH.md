# Controlled feature research

The completed 30-feature Fold 0 ranker is an exploratory result. This study
measures a broader feature space using explicitly chronological inputs. Its
protocol is frozen in `configs/research.toml` before feature screening or model
selection. Original OTTO data has appeared in earlier experiments; this study
does **not** claim that the underlying dataset has never been inspected.

## Measured outcome

[Notebook 09](../notebooks/09_controlled_feature_study.ipynb) contains the executed
analysis. Eight configurations produced 24 native models. Selection chose the
102-feature `without_source` variant for clicks, carts and orders. Graph and intent
interactions still encode retrieval information; this ablation isolates the direct
source-feature family.

On all 432,492 reserved evaluation sessions, weighted Recall@20 is **0.584392** versus
**0.564904** for the 28-feature core and **0.535244** for fixed fusion. The paired 95%
gain intervals are [0.018399, 0.020645] and [0.046972, 0.051363], respectively. Fusion
still wins click and cart recall; orders explain the weighted gain. The evaluation
and source audit are published under [reports/research](../reports/research).

The independent auditor reconstructs all 788,883 labels and every observed prefix
from the 217 original Parquet event partitions, with zero differences in either
direction. It verifies all 24 native models, recomputes the scores from complete
query counts, and matches 4,608 sampled model/candidate checks. Native TreeSHAP,
family permutation and matched feature costs use selection queries after the model
is frozen. The selected schema's warm feature-computation p95 is 6.42 ms, versus
22.10 ms for the broad catalog; this is not an online serving measurement.

## Availability and evaluation

| Component | Available data, UTC | Selection rule |
|---|---|---|
| Historical retrieval and item statistics | Events before 2022-08-20 22:00 | No query labels |
| Ranker fitting and feature screening | Sessions starting August 20 22:00–August 23 22:00 | 100,000 deterministic session hashes |
| Model selection and candidate coverage diagnostics | Sessions starting August 23 22:00–August 24 22:00 | 20,000 deterministic session hashes |
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
use pointwise binary LightGBM models and sampled fitting negatives; they are
inexpensive training-only utility diagnostics, not final ranking results.

The completed screen retained 128 features and rejected 1,354: 54 constants,
5 near-constants, 146 exact duplicates, 31 highly correlated columns and 1,118
columns below the retention budget. The retained families contain 57 historical,
26 source, 19 context, 17 repeat, 5 graph and 4 interaction features. Selection
considers every eligible column; a predeclared compact core remains available
as a matched control. The full fitting cache has 6,083,582 rows over 100,000
sessions. The selection cache contains every one of the 400 candidates for each
of 20,000 sessions: 8,000,000 rows.

```bash
.venv/bin/python scripts/run_research.py --stage features
.venv/bin/python scripts/run_research.py --stage screen
.venv/bin/python scripts/run_research.py --stage model_features --role fit
.venv/bin/python scripts/run_research.py --stage model_features --role selection
.venv/bin/python scripts/run_research.py --stage ablate
.venv/bin/python scripts/run_research.py --stage evaluate
```

## Matched comparisons and final evaluation

Eight feature configurations use the same fitting queries, candidates,
negative sample, LightGBM settings and stopping rule: the 28-feature core,
all 128 retained features, and six leave-one-family-out ablations. Each fits
separate click, cart and order LambdaRank models. The training limit is 400
boosting iterations with 40-iteration patience. Complete selection Recall@20
is evaluated at iteration 1 and every 5 iterations thereafter. The first
measured maximum wins ties. Native models and their full experiment contracts
are checkpointed every 25 iterations and at completion.

Model selection chooses the best measured selection Recall@20 separately for
each objective, breaking exact ties by fewer features and then the variant
name. An immutable seal records these models, native model SHA-256 digests,
feature order, retrieval identity and the corpus identity before evaluation
labels are opened. The candidate budget is fixed at 400 for these comparisons;
100/200/400 candidate ceilings characterize coverage and are not separately
trained budget comparisons.

Reserved evaluation streams all eligible queries without negative sampling.
It reports the official weighted Recall@20, per-objective recall, NDCG@20,
MRR@20, hit rate, candidate ceilings and observed-prefix slices. Paired session
bootstrap intervals compare the selected models against the core and fixed
source fusion. They quantify sampling variation conditional on the frozen
models and this cohort; they do not quantify training-seed variation or remove
the possibility of temporal drift. Native checkpoint recovery and missing
evaluation-part recovery are exercised by tests on actual trained models.

## Managed execution

`scripts/processing_research.py` verifies every staged corpus object, split
input bundle, assembled archive and source bundle before installing the locked
Python stack. `otto_recsys.cloud.research_job` restores checkpoints through the
AWS execution role, runs the ablations, seals the choice and runs evaluation.
The selected execution configuration and source commit are recorded in the
job status. Each upload includes an expected bucket owner and SHA-256 metadata;
each restored file is independently hashed. Data arrives before its receipt.
An interrupted job resumes under the identical source and experiment contract.
SageMaker's job name and maximum runtime bound the managed execution; no
persistent serving endpoint is required.

The research completion evidence is now published: candidate-budget coverage, the
full feature catalog and rejection reasons, matched ablations, the model-selection
seal, complete temporal evaluation with paired uncertainty, interpretation, measured
feature cost and an independently audited analytical notebook. The separate batch
inference workflow is documented in [INFERENCE.md](INFERENCE.md).

The design draws on the official [OTTO task and evaluation specification](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md)
and the documented [third-place feature approach](https://github.com/TheoViel/kaggle_otto_rs).
Reported competition scores use different cohorts and are not direct comparisons
with this study's temporal evaluation.
