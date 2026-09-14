# OTTO feature research — evidence before feature count

## Current source-derived finding

The user's verified Round 01 evidence reports 0.484736883 for the 134-feature control and 0.494229507 with 24 timing additions: +0.009492624. Support-only and support-plus-timing regressed. Timing improved both forward folds, but its descriptive interval [-0.003769246, 0.026560197] includes zero. It added five click hits, one cart hit and one order hit; the one order supplies about 69% of the pooled weighted gain. These facts motivate replication rather than indiscriminate expansion.

The new kernel receipt reports execution counts 1 and 2 with Idle replies, followed by all seven emitted chart specifications and NOTEBOOK_REVIEW_COMPLETE. Browser pixels were not independently checked. The ML interpreter stayed separate from the notebook interface, with no installations or refits.

## This round's preregistered inference question

Does the exact frozen timing representation improve a **matched newly trained control** across 4096 different fitting-session IDs? The old 1024 sessions are excluded from both new model training and new validation. Eight old sessions are used solely for numeric compatibility checks, never for fit or score. The new session pool is selected by a fixed ID hash, before inspecting target counts. It is still drawn from a previously explored training dataset.

Do not change windows, thresholds, graphs, normalization, candidate count, tuning parameters or the 24 formulas based on this result. Two arms, two forward folds, three objectives: 12 new models. Each component of the 24-feature block was selected after the prior experiment's secondary comparison; this replication tests the block, not 24 individually selected columns.

## Representation and information availability

For each symmetric/forward graph and time/cart/order channel, build two observed-item pools: the latest 20 distinct items and the latest 20 items with an observed cart event after their last observed order. Exclude candidate self-edges. The two timing summaries are:

- Affinity-weighted average of log(1 + age in hours) across supporting items.
- Log(1 + age in hours) of the strongest supporting item; tied affinities follow most-recent observed-item order.

Age refers to the item's latest **observed occurrence**, not necessarily its cart timestamp. No future events, labels, actual cart inventory or abandonment status are inputs. Zero can mean no support or zero elapsed time under the original formula; do not silently redesign that convention for this replication. Graphs come from the certified exclusive cutoff 2022-08-16 22:00 UTC.

## External domain evidence, distinct from our measurements

The third-place author's published OTTO component describes candidate-to-session similarity scores aggregated with item position, timestamp and action-type information. This supports the *class of hypothesis*, not proof that our 24 equations have been used by winners or will improve this project. It also demonstrates why candidate counts and feature counts alone are inadequate success criteria.

Primary source: https://github.com/TheoViel/kaggle_otto_rs (Feature engineering section).

Plotly's documented JupyterLab renderer uses the Plotly MIME bundle. Both new saved-result notebooks use explicit raw MIME display; they do not invoke browser renderers or repeat inline HTML initialization.

Primary source: https://plotly.com/python/renderers/

## Metric, leakage and model controls

Pooled hits@20 / pooled min(20, full distinct target count), combined as 0.1 clicks + 0.3 carts + 0.6 orders. Ground-truth products absent from candidate retrieval remain in the denominator. Query-time chronological folds use a six-hour embargo. Training labels are censored at the corresponding validation-start cutoff before negative sampling; saved target first occurrences do not certify completed shopping sessions. This is a retrospective fitting diagnostic, not an online backtest.

The candidate pool stays at 400 and the shared baseline stays at 134 columns. LambdaRank parameters, seeds, deterministic mode, 150 boosting rounds, objective-specific training, 60-negative budget and negative sampling seed stay fixed. A holdout score is never used for early stopping or feature selection. Training-only support/correlation diagnostics cannot automatically delete columns.

New controls are necessary because the session cohort changes. Reusing the old 0.484736883 control score on new sessions would confound feature value with cohort difficulty. Old 54-model and 18-model experiments are not rerun.

## Decision

Report all objectives, both folds, the exact net number of recovered hits, descriptive paired intervals stratified by fold, and sensitivity to a single order hit. A weighted gain >=0.003 with nonnegative signs on both folds and no pooled order-recall loss permits *later cross-window confirmation*, not production promotion. A confidence interval spanning zero remains unresolved. A negative or inconsistent result is kept as negative evidence; do not tune a variant until it passes the same reused data.

The 0.60503 historical private score is the research target supplied by the user. The accepted 0.56842 private submission is not replaced by this experiment. Internal scores above 0.60503 do not establish a leaderboard record because their data/protocol differ.

## Following bounded rounds (not launched or implemented by this package)

1. **Recent-support mass.** After interpreting frozen timing replication, engineer the fractions of candidate affinity contributed by observed items in the last 5, 30 and 120 minutes. With two graph families, three channels and two pools, this is a proposed 36-feature family. These fractions carry distribution information not recoverable from a single average age. Missing-support indicators require a separately documented design, not retrospective alteration of the timing baseline.
2. **Strict rolling as-of demand/support.** Investigate the existing 75-feature family with snapshots strictly before each permitted query time, action-specific count/rate changes and relative demand. The earlier 12 fixed-cutoff relative features are not this family. Reuse available events; do not allow validation outcomes into transforms or fitted encodings.
3. **Action-pair and shopping-episode relations.** Test click→cart, cart→order, return-visit and short-versus-long-gap interactions, retaining the observed-state interpretation. Add/drop each plausible block under matching temporal controls rather than generating unmotivated cross-products.
4. **Latent candidate–session similarity and cross-window confirmation.** Audit existing Item2Vec/two-tower/ANN studies first; do not retrain a generic embedding merely because a new run is possible. Test changed hypotheses, source-specific aggregates, and objective conditioning against the same candidates before later considering wider candidate sets or ensembles.

Each round requires hypothesis, availability contract, unit tests, bounded pilot, observed feature-value comparison, diagnosis, preservation and report. Algorithm choice is held fixed here to isolate representation; this is not evidence that algorithms or candidate retrieval play no role in the remaining competition gap. Feature engineering remains open.
