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
