# Model card: controlled OTTO ranker

## Purpose and intended use

Rank candidate products for the next click, cart and order in anonymous OTTO shopping
sessions. This is an offline research and batch-inference demonstration for an employer
portfolio. Outputs are ordered product IDs, not calibrated purchase probabilities.
No online deployment, revenue lift, personalization across identified users, or Kaggle
leaderboard result is claimed.

## Model and inputs

Three native LightGBM 4.7.0 LambdaRank models use the same selected 102-feature schema.
Click, cart and order checkpoints contain 40, 60 and 95 boosting iterations respectively.
The experiment uses learning rate 0.05, 31 leaves, minimum leaf size 100, deterministic
CPU training and a fixed seed of 20260908. Complete parameter contracts accompany each
native checkpoint.

Candidate generation combines three historical co-visitation channels, observed-session
revisits and historical popularity into a common pool of up to 400 products. The selected
schema contains 57 historical-context, 19 session-context, 17 recurrence, 5 graph-affinity
and 4 intent-interaction features. Direct source-family columns were excluded after matched
selection experiments. Graph and intent interactions still encode retrieval information. The catalog defines all formulas and their availability in
[feature_catalog.csv](../reports/research/feature_catalog.csv).

Raw product and session IDs locate observations and candidates; they are not numeric
model features. Features use the fixed permitted history and observed prefix. Unseen
items and missing graph edges receive defined fallback values. Candidate generation
cannot recover every future product; those missed targets remain in evaluation.

## Training and selection

The source conversion covers 216,716,096 events in 12,899,779 training sessions.
All 217 original Parquet partitions were independently hashed; raw JSONL source identity
is retained, but the original conversion was not independently replayed.

| Role | Availability, UTC | Query sample |
|---|---|---:|
| Historical retrieval | Before August 20, 2022, 22:00 | All permitted history |
| Ranker fitting | August 20–23, 22:00 boundaries | 100,000 sessions |
| Model selection | August 23–24, 22:00 boundaries | 20,000 sessions |
| Reserved evaluation | August 24–26, 22:00 boundaries | All 432,492 eligible sessions |

Sessions are disjoint across roles and assigned by their first event. Exclusive period
ends clip future observations. Prefix cuts and sampling do not consult target labels.
Targets are the next click and unique future cart/order items.

The broad screening sample has 514,013 candidate rows from 8,448 fitting sessions.
Three session-grouped folds and three binary objective pilots provide gain/stability
diagnostics. Quality, redundancy and capacity screening reduce 1,482 formulas to 128.
Eight matched configurations fit three rankers each. All use the same fitting candidate
sample: discovered positives plus 30 hard and 30 random negatives. Selection retains
all 400 candidates per query. Models maximize selection Recall@20, with fewer features
then variant name resolving exact ties. The source-family ablation wins all tasks.

The model-selection seal was persisted before the evaluation labels were opened. The
original OTTO dataset had earlier exploratory exposure; this newly reserved temporal
study does not claim never-inspected data. Earlier Item2Vec and neural checkpoints are
not inputs to the controlled comparison.

## Evaluation and tradeoffs

| Metric | Fusion | Compact ranker | Selected ranker |
|---|---:|---:|---:|
| Click Recall@20 | 0.526781 | 0.457558 | 0.505645 |
| Cart Recall@20 | 0.430211 | 0.408325 | 0.427919 |
| Order Recall@20 | 0.589171 | 0.661085 | 0.675753 |
| Weighted Recall@20 | 0.535244 | 0.564904 | 0.584392 |

The selected model improves weighted recall by 0.049148 over fusion and 0.019488 over
the compact control. The respective paired 95% gain intervals are [0.046972, 0.051363]
and [0.018399, 0.020645], from 1,000 session bootstrap replicates. These quantify query
sampling uncertainty conditional on the frozen model and temporal cohort, not training
or hyperparameter-search uncertainty.

Fusion wins click and cart recall. Orders carry 60% of the official metric and explain
the aggregate gain. The 2–5-event prefix slice slightly loses to fusion. No hybrid was
chosen after observing these evaluation results. The 400-candidate weighted ceiling
is 0.686951, leaving both retrieval and ranking headroom.

Native TreeSHAP and whole-query family permutation are diagnostic explanations, not
causal effects. The warm feature-computation benchmark measures 64 queries over two
passes of the same 32 prefixes: selected p50/p95 5.83/6.42 ms versus 17.21/22.10 ms for
the full catalog. It excludes model scoring, network, storage and service overhead.

## Verification and lineage

The independent auditor reconstructed every observed prefix and all 788,883 target
records across fit, selection and evaluation from the original Parquet event partitions.
Both directions of the comparisons contain zero differences. Complete per-query metric
counts reproduce every published score, all 24 native model digests match, and 4,608
sampled prediction/candidate checks have zero mismatches. Another 6,912 sampled
NDCG, reciprocal-rank and hit-rate checks match independent arithmetic.

| Artifact | Identity / evidence |
|---|---|
| Temporal corpus | `55ad451e895863af311e4a917a6fe0d4ab9165ad6b406fffb066d25bf0af4754` |
| Selection seal | `0b77504da3e1be6d41fec6e9fa8395106bd4d32741e60a913d87874d83b6af4e` |
| Evaluation | `adff5d8966b06ba568de08e84646061c42a2d87899fc5d1ba1577f1783cc1114` |
| Native models and feature order | [Evaluation seal](../reports/research/evaluation_seal.json) |
| Independent audit | [Audit report](../reports/research/audit.json) |
| Full evidence hashes | [Publication manifest](../reports/research/manifest.json) |

Competition inference uses the frozen ranking weights with a separately identified
historical graph refreshed from all permitted training events. Training history must
precede test observations and have disjoint session IDs. This operational refresh does
not change or retroactively replace the research evaluation.

Full Notebook 10 inference completed for 1,671,803 competition sessions and 5,015,409
rows. The default notebook exactly replays eight of those sessions using the native
models. [Execution and coverage evidence](../reports/research/competition_cloud_verification.json)
records all part receipts and the final output identity.

## Limits and appropriate conclusions

One fixed training seed, one newly reserved temporal cohort and a bounded 100,000-session
fitting sample support a specific empirical comparison. They do not establish robustness
to every period, seed, market or catalog. The dataset is historical and anonymized;
protected-group fairness and individual-level outcomes cannot be evaluated from these
inputs. The task is offline recommendation, with no online exploration or exposure-bias
correction. The multi-objective metric is a prescribed weighted aggregate, not a learned
Pareto frontier or a business-utility estimate.

Additional seeds, further temporal cohorts, certified neural retrieval under the same
cutoffs and an online experiment would extend the evidence. Those are future research
questions, not prerequisites for reproducing the completed experiment.
