# Feature-value pilot: measured ranking effects, not feature count

## Research target and mechanism

The top competition performance remains a research target, not a guaranteed outcome. A passing engineering check does not establish predictive improvement. Feature engineering remains open; ranking fits below are controlled feature experiments, not a move to final training. A score above the historical record must be established on a comparable competition evaluation, never inferred from this small fitting pilot.

The first-place author's write-up reports multi-window popularity ranks, action-conditioned neural similarities, multiple co-visitation views and roughly 1,200 candidates. About 200 candidate features were narrowed to roughly 100 per objective. [Primary solution](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution). The final private winner score is 0.60503. [Official leaderboard](https://www.kaggle.com/competitions/otto-recommender-system/leaderboard).

The certified representation has 400 candidates and 102 baseline plus 32 shared additions. For six count sources (6-hour clicks; 72-hour carts/orders; 168-hour carts; 336-hour total activity; 1-hour total activity), compute a tie-aware popularity percentile and fraction of candidate-set activity. Zero count yields zero evidence. This adds 12 interpretable features. The transform accepts no labels and runs on complete candidates before target-dependent negative sampling. It is invariant to candidate permutation and positive scaling of counts.

These are fixed historical snapshot features, not the separately proposed 75 rolling as-of demand/support features. The latter still require snapshot construction and tests. No formulas are claimed novel merely because they add columns.

## Frozen experiment and leakage contract

Reuse the four verified 256-session feature partitions and denominator ledger from `early-scale-29439ec2`. Do not reconstruct graphs. Within those fitting sessions, order by query timestamp; use forward 512-to-256 and 768-to-256 splits. Exclude training queries within six hours of validation start, including ties. Censor every training positive at the exclusive validation cutoff before computing fitting negatives. Saved cart/order first-occurrence timestamps and the next-click timestamp recover these pre-cutoff positive sets; they do not identify session completion. Never infer complete outcome availability from the maximum first-occurrence timestamp. Training negatives have a shorter declared horizon than the original full-period labels. This retrospective within-fitting comparison is not a deployment-faithful online backtest.

No selection or evaluation rows enter the experiment. Compare baseline, shared, baseline plus relative demand, shared plus relative demand, and five leave-one-family-out shared variants: funnel, episode, raw graph, row-normalized graph, degree-normalized graph. Nine arms, three objectives, two folds: at most 54 small native rankers. Use 150 fixed rounds, 15 leaves, four threads and no validation-based early stopping or hyperparameter search. Candidates, time-censored labels, negative-sampling policy and seed match across arms. [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

Compute the [competition pooled weighted Recall@20](https://www.kaggle.com/competitions/otto-recommender-system/overview/evaluation), not a mean of per-session recalls. Also compute the candidate-oracle ceiling on the same cohort. Report fold support, embargo exclusions, label censoring, objective recalls, paired unadjusted descriptive intervals, resource usage and native reload parity. Small reused fitting results cannot establish a leaderboard record. Nonnegative gain on both folds plus positive pooled gain permits expanded fitting-only replication, never automatic promotion.

## Durable execution and open gaps

Use immutable source, contract, fold membership, native models and per-model receipts. Each completed arm writes session-level hit statistics. A rerun must reload verified models, make zero fitting calls and reproduce metrics. The outer worker provides a hard limit, UTC heartbeats and private S3 checkpoints. Commit source and measured results; align the existing AWS checkout to the final merged SHA without resetting local work.

After this pilot, investigate the measured gaps in ranking within the fixed frontier, missing candidates, action-conditioned sequence/embedding signals, multi-hop co-visitation, and current as-of demand. Prioritize official-metric contribution, support, uncertainty and compute cost. Neither feature count nor one favorable small fold closes the research gate.
