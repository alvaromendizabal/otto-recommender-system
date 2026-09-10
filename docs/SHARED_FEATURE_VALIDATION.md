# Frozen shared features: larger validation and conditional ablations

## Hypothesis and source evidence

The prior pilot improved weighted Recall@20 from 0.578084 to 0.604069 using 32 shared additions. Separate per-task selection did not beat the shared control. This study asks whether the frozen shared representation retains a useful gain with more training support. It does not rerun the failed per-task selector or rescreen features.

The [official OTTO task](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md) distinguishes the next clicked item from sets of future cart/order items, weighted 0.1/0.3/0.6. The [third-place implementation](https://github.com/TheoViel/kaggle_otto_rs) motivates candidate-to-session similarities and action/recency weighting. These sources motivate the representations; they do not validate our chosen columns, episode thresholds or normalization.

## Fixed comparison and decision

[Exact configuration](../configs/shared_feature_validation.json). Preserve the original 102 baseline columns and the exact 32 shared additions, verified against the prior fitting-only screening file and study identity. Use 50 fitting and 20 selection partitions: 12,800 / 5,120 complete sessions. Systematic spacing excludes potentially partial tails. The larger samples need not contain every smaller-pilot partition; only matched arms within this study are directly comparable.

Both arms use identical candidates, targets, fitting negatives, seed 20260910, four CPU threads and task-specific LambdaRank settings: at most 150 rounds, 30-round patience, metric checks every five rounds. Selection keeps all 400 candidates per query and denominators include targets missed by retrieval. No new features, screening, candidate expansion or final-evaluation model selection occurs in this comparison.

Advance to ablations only if shared minus baseline weighted recall is at least 0.001 and order recall does not decline. A failed point gate stops this escalation; a passing gate justifies another development experiment, not promotion. All successful controls and statistics are checkpointed before the result is reviewed.

## Prespecified conditional block removals

A second, separately launched stage requires the completed matching validation result and unchanged model checkpoints. It reuses all six controls and fits only the five block removals (15 models).

| Removed block | Columns | Domain rationale | Information and leakage boundary |
|---|---:|---|---|
| Funnel/state | 11 | Unfinished baskets, repeated attention, transitions and concentration | Observed session prefix only; no future cart/order events |
| Episodes | 6 | Shopping bursts, recent activity and recency across gaps | Prefix-only 30-minute/two-hour gap heuristics; thresholds are not externally validated |
| Raw graph affinities | 3 | Strength of candidate relationships to recently observed items | Previously certified historical edges; observed prefix query pools |
| Row-normalized graph | 10 | Compare affinities relative to outgoing graph mass | Historical retained/pruned row mass, not full transition probabilities |
| Degree-normalized graph | 2 | Adjust candidate relationships for popularity/connectivity | Historical retained graph degrees; not complete catalog exposure probabilities |

All removals retain every baseline column and the order of all remaining columns. Report full shared minus dropped-block differences with paired session-bootstrap intervals. Five related comparisons on repeatedly used development data are exploratory: these intervals do not correct for research selection or establish simultaneous family significance. A winning removal is a candidate for later confirmation, not an automatically promoted schema.

Query-time quartiles use observed prefix timestamps with session-ID tie breaking. Prefix slices are 1, 2–5 and 6+ events. Show counts, task denominators and weighted results; slices with any unsupported objective are marked insufficient. These slices describe stability within the cohort and do not replace temporal retraining across folds/seeds. Correlated blocks may substitute for one another; a marginal removal does not prove a feature family is universally useless.

## Execution and durable evidence

Run the synthetic two-stage tests and frozen-schema guards before paid compute. Verify all source/input hashes and schemas. Each managed job has a 900-second ceiling using the previously proven CPU image. The earlier same-day instance rate implies at most $0.8568 instance compute per stage, excluding storage, requests, transfer and logs. The ablation stage is launched only after inspection of the first result.

Publish contracts, model checkpoints, UTC heartbeats, arm counters, per-session statistics and phase-specific results to the owned S3 prefix. A second pass disables training and must reproduce every model hash and metric. Independently recalculate pooled metric arithmetic and verify native feature order. Successful validation remains separately saved when ablations append results. A changed schema, source, input or model checkpoint must fail before replacement fitting.

Feature engineering remains open. The accepted Kaggle score is unchanged, and this reused development cohort cannot establish parity with the historical winning private score.

## Reproduction

With the checksum-verified inputs restored, run `uv run python scripts/run_shared_feature_validation.py --phase validation`. Inspect the gate before running the same command with `--phase ablation` against the same output directory. The CLI enforces a 900-second deadline; the managed bootstrap additionally verifies zero-training replay and uploads checkpoints.

## Completed larger comparison

The frozen shortlist scores **0.5950734445** against the matched baseline's
**0.5898696552**: **+0.5204 percentage points**, with a descriptive paired 95%
interval **[-0.4007, +1.5958] points**. Orders gain nine hits (559 versus 550);
clicks and carts each lose six. All six native models, independent pooled metric
arithmetic and zero-training replay passed audit.

The prespecified point gate passes, so it authorizes the five block ablations,
not promotion. Query-time gains are +1.025, +1.311, -1.755 and +1.073 points:
the third quartile underperforms. All three prefix-length group point changes
are positive, but this does not establish temporal or training stability.

The fixed candidate ceiling is 0.692289 on these development sessions. The
0.097215 gap from the shared ranker is recoverable ranking headroom only in
principle, not an expected feature gain. None of these development quantities
can be subtracted from the historical Kaggle winning score to estimate the
competition gap. The accepted private-score gap remains 0.03661.

## Primary-source review and remaining representation gaps

The [winning implementation's feature join](https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/blob/main/codes/otto/scripts_prepare_2nd/prepare_feature.py)
combines multiple graph and learned candidate sources, joins item statistics to
session-hour statistics, and computes within-session ranks of item counts, rates
and demand features. Our original source ranks are already tested; they are not
new work. Query-relative historical demand ranks and genuinely query-time demand
updates remain distinct open hypotheses. The current dataset builder computes
features on the complete candidate pool before taking fitting negatives. New
relative features must use that path: recomputing ranks on the already sampled
fitting cache would make fitting and selection representations inconsistent.
Time-indexed popularity must use events available at each query; the names of
historical competition windows are not evidence that they meet this project's
strict cutoff contract.

The [winner's action-conditioned encoder](https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/blob/main/codes/otto/scripts_nn/emb_model_v42.py)
adds target-action embeddings to session representations and selects difficult
negative scores during training. Its candidate scores enter the later ranker.
This motivates jointly testing representation complementarity and ranking value;
a new retriever's recall alone is not sufficient. We already have an objective-
conditioned two-tower implementation, so reproducing that concept without a
controlled change would duplicate work. It still needs a current-window,
feature-aware comparison under the accepted protocol.

[TRON](https://arxiv.org/html/2307.14906v2), from OTTO researchers, combines efficient
uniform/in-batch sampling, top-k hard-negative updates and sampled softmax. Its
reported offline setup uses clicks, minimum-support filtering and next-item
metrics, so those results do not establish a gain on this competition's weighted
three-action target. Our inference is narrower: negative sampling and temporal
sequence representation deserve a controlled experiment after cheaper feature
hypotheses, with matched data and full-query weighted Recall@20 evaluation.

The [winner's reproduction notes](https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/blob/main/codes/readme.txt)
describe substantially larger hardware and multi-day builds. These are historical
cost evidence, not a prescription for this project. Reuse candidate caches, test
one representation change at a time, and require measurable marginal value before
scaling or expanding an ensemble. The 3.661-point private-score gap cannot yet be
reliably apportioned among features, retrieval, data support and ensembling.

## Completed block ablations

| Removed addition block | Columns removed | Weighted Recall@20 | Full minus removal (pp) | Descriptive 95% interval (pp) |
|---|---:|---:|---:|---|
| funnel | 11 | 0.592273 | +0.280 | [-0.328, +0.939] |
| episode | 6 | 0.592745 | +0.233 | [-0.444, +0.953] |
| graph_raw | 3 | 0.587329 | +0.774 | [-0.097, +1.658] |
| graph_row | 10 | 0.592504 | +0.257 | [-0.533, +1.103] |
| graph_degree | 2 | 0.594788 | +0.029 | [-0.600, +0.723] |

Every removal reduces the point estimate; none of these five descriptive intervals
excludes zero. The raw graph block has the largest measured contribution and the
degree-normalization block the smallest. These are conditional removals from the
32 additions, not ablations of all graph or sequence features in the system.
The full shortlist remains the frozen challenger. There is no evidence here to
justify repeatedly tuning block combinations or promoting any individual family.

The six control models and every control metric are identical across stages. All
21 native models pass schema/hash checks; pooled metrics are independently rebuilt
from per-session hit/denominator artifacts. Complete restart performs zero training
calls and reproduces every arm. All models, per-session statistics, contracts and
heartbeats are saved under the source-addressed S3 checkpoint URI in the run receipt.

The [next temporal protocol](../configs/shared_feature_confirmation.json) freezes
early/middle window comparisons and the full schema. It is a design and preflight
contract, not an executed replication. Start with artifact checks and a 256-query
feature smoke, then 1,024 sessions to measure cost. Each later managed stage keeps
the 900-second cap. Reference-window graphs cannot be reused at earlier cutoffs.
This is worth additional compute because one current query-time quartile regresses
and all family intervals span zero; transfer evidence is more informative now than
further tuning on the same cohort. Broader domain-informed feature work remains open.
