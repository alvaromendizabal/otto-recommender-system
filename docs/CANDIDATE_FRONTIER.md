# Candidate-frontier library: feature-research checkpoint

**Status: independent retrieval library implemented and locally tested; no real-data frontier experiment executed.** The execution runner is not included in this change: its GitHub publication was blocked by the tool safety layer. It has not been retried through another route. AWS actions were also not exposed by current connector discovery, so no live workspace synchronization or compute-state claim is made here.

## What is implemented

`src/otto_recsys/research/candidate_frontier.py` preserves the exact current baseline candidate prefix, then builds one-hop or bounded two-hop expansions from existing baseline, symmetric and forward historical graph objects. The five arms are baseline 400, one-hop 800/1200, and two-hop 800/1200. The 800-item arm is an exact prefix of its corresponding 1200-item arm; the two different hop families are not required to nest in one another.

The library generates the baseline once for all arms, builds each 1200-item expansion once, handles unseen item IDs without accidentally looking up another row, rejects bad graph evidence, and uses deterministic score/item-ID tie breaking. It accepts no target labels for candidate generation. `candidate_ceiling` is an explicitly separate evaluator which rejects duplicate labels, multiple next-click targets and denominators that do not equal `min(20, full distinct truth count)`. Missing targets remain in the denominator. The baseline identity-only control explicitly marks source evidence unavailable rather than presenting placeholder zeros as measured features.

Twenty independent fixture tests cover these behaviors, exact baseline preservation, real 800/1200 budgets, two-hop-only discovery, nested scores/ranks, deterministic replay, input nonmutation, empty graphs, cutoff mismatches, resource limits and integer overflow. They passed locally with Python warnings treated as errors. The six additional local runner/checkpoint tests are **not** part of the published test count or this library PR. Full repository CI remains required; local tests use Python 3.13.5, NumPy 2.3.5 and SciPy 1.17.0 rather than the full locked AWS environment.

The prior experimental branch at `d82ff8131ea20623f5e963cc0c5b742bb3c813f6` contained an invalid starred list-comprehension expression and an eight-candidate fixture that contradicted its 400-minimum contract. This independent library checkpoint corrects those defects without merging that branch's separate PR #50 evidence. No existing canonical notebook, model, feature formula, CI gate or production candidate budget is changed.

## Why this hypothesis is worth testing

The [first-place solution](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution) reported approximately 1200 candidates, multiple action/time-weighted covisitation matrices, repeated graph expansion and target-conditioned neural candidates. It also used ranked popularity, interaction features and ensembles. The [final private leaderboard](https://www.kaggle.com/competitions/otto-recommender-system/leaderboard) records 0.60503 for the winner. That is a competition comparator, not directly comparable to retrospective fitting or selection scores.

Count alone is not the objective: [the third-place author's component](https://github.com/TheoViel/kaggle_otto_rs) reported about 80 candidates and 744 weighted candidate/session similarity features. A 400-candidate cap is therefore a possible recall limitation to measure, not proof that it is the largest or sole competitive gap. A candidate oracle above the achieved ranker score also leaves substantial representation/ranking work. A negative ranker-only Item2Vec ablation does not establish that Item2Vec retrieval is useless.

## Prior completed evidence, not a new run

PR #52 is merged at `1d9681b3961b4aacf64aa1ff8baaa143f6a97cd2`. Its `reports/research/feature_value_pilot.json` and executed research notebook document 54 models on two fitting-only forward folds, a six-hour query embargo and censored training targets before negative sampling.

Pooled weighted Recall@20 was 0.470574 for the baseline, 0.484737 for shared features, 0.486757 for shared plus 12 relative-demand features, and 0.490712 after dropping the row-normalized graph block. Important gain intervals include zero. Relative demand had mixed fold signs. Dropping row-normalization improved both fold point estimates but remains a confirmation hypothesis, not permission for automatic feature deletion. The earlier 0.537570 three-fold OOF result uses a different protocol and cannot be ranked against these forward-fold scores.

## Next real-data gate: preregistered, not executed

The included configuration specifies 256 fitting sessions from seed 20260911; it does not launch or execute them. Require the certified Aug-16 exclusive cutoff `1660687200000`, verify all graph and corpus manifests, and compare every baseline candidate ID/order against the existing `early-scale-29439ec2` checkpoints. Do not rebuild the certified graphs or rerun the completed 54-model pilot.

The planned study must retain full targets in its denominators, report all five arms, and measure oracle candidate coverage separately from achieved ranking. Fixed source weights, RRF offset 20 and at most 64 bridge items per source are not to be tuned against smoke labels. The library enforces a 250000-item discovery cap by failing, not silently truncating. The configuration's 300-second work limit, durable per-session checkpoints, source-hash verification, external app-lifetime cap and verified shutdown remain **runner/launch requirements**, not claims that the library alone enforces cloud execution safety.

A best expanded weighted candidate-ceiling gain of at least +0.005 permits only an unchanged, separately committed 1024-session fitting-only replication. It does not authorize final-model selection, feature retention, selection/evaluation access or a leaderboard claim. Reused fitting data and choosing the best of four expanded arms are exploratory, not independent confirmation.

## Feature engineering remains open

Continue testing richer source/candidate-session aggregation, action-pair and time-window covisitation, neural/task-conditioned retrieval, shopping episode/funnel signals, graph normalization, and the separate 75-feature strict rolling as-of demand/support family. The tested 12 fixed-cutoff relative-demand features are not a substitute for that rolling family. Screen on fitting data, use matched add/drop comparisons, and require temporal stability before retention. Use existing ANN/Item2Vec evidence before paying to retrain embeddings.

Anonymous item IDs do not supply product categories, prices or customer identity; external metadata needs an actual permitted mapping. The strongest attainable comparable competition score remains the target, but beating it cannot be guaranteed in advance. Neither a large feature count nor successful code publication closes feature engineering.
