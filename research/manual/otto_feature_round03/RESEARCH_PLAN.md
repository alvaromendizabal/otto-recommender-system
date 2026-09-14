# OTTO Round 03 — recent affinity mass

## Status and rationale

Not executed on real AWS data. Round 02's frozen 24 timing features regressed on both time-fold point estimates. Keep that negative finding; do not promote, resubmit, or scale that unchanged configuration. More feature columns are not automatically better.

The next hypothesis was listed before this result: summarize the **distribution of supporting observation ages**, not just mean/strongest-support age. A primary implementation account by the third-place author describes candidate–session item similarities, timestamp/type/position weighting and aggregation: https://github.com/TheoViel/kaggle_otto_rs . This supports the broad representation strategy, not these exact equations or window values. This is an authored testable extension, not a claimed winner formula.

OTTO's official metric and timestamp-truncated task: https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md . Preserve complete capped target denominators, including targets absent from retrieval. The test set is not accessed in this experiment.

## Exact 36 formulas

For candidate c, a certified historical graph G, observed seed pool S, and window w:

`recent_mass(c,w) = sum(G[s,c] for s in S, s != c, age(s) <= w) / sum(G[s,c] for s in S, s != c)`.

Windows are fixed **5, 30, 120 minutes**, left-inclusive at zero and inclusive at the upper boundary. Pool members are the most recent 20 distinct observed products, or the most recent 20 distinct products with a cart observation after their latest observed order. Ages use the member's last observed event. The latter is an observed-action state, NOT an observed live shopping cart or known abandonment.

2 graph families (symmetric/forward) × 3 graph channels (time/cart/order) × 2 pools × 3 windows = **36 columns**. The graph edges are retained historical affinities, not complete behavioral probabilities. Self-affinity is excluded. Unknown IDs have no graph edge; no lookup may alias them to another item. Ratios are invariant to positive rescaling of all affinities in one pool. Short-window mass cannot exceed longer-window mass.

**Missing-aware primary:** no positive other-item edge mass is −1. A legitimate zero means support exists but all of it is older than the window. A legitimate one means all support falls inside the window.

**Controlled ablation:** use the same 36 formulas and replace only −1 with zero. This isolates whether distinguishing absence helps. It is not an established diagnosis of Round 02's failure: that failure could reflect sampling noise, redundancy, representation mismatch, capacity interaction, or several factors. No new columns are chosen after observing validation scores.

## Hypothesis, controls and metrics

Primary: 134 shared-control columns plus 36 missing-aware fractions versus the saved 134 shared control. Secondary: the identical 170-column configuration with missing values filled with zero. A comparison between the two new configurations is an encoding ablation. Do not select whichever wins and retrospectively call it primary.

Reuse the exact Round 02 4,096 sessions and 400-candidate arrays, two forward folds, six-hour query embargo, and targets censored at validation-start before negative sampling. Controls are reloaded with exact model/input/target/group/feature-order contracts and must reproduce all saved per-session top-20 hits. No new control fits, control feature materialization, algorithm tuning or graph construction. Native reload parity is required for every new model.

Six new models per challenger: two folds × three objectives. Maximum **12 new experiment fits**. Unit tests include a separate tiny synthetic native-model fixture and are not competition experiments.

Screen feature support and correlations using fixed samples of each fold's training sessions only. Support/missing fractions and correlations are diagnostics, not automatic feature retention. Keep all 36 formulas in each arm for the prespecified group comparison. Train and score on identical rows/settings apart from the representation.

Report pooled weighted Recall@20 (0.1 clicks + 0.3 carts + 0.6 orders), action recalls, net hit changes, both fold gains, and 2,000 within-fold paired bootstrap intervals. Intervals are descriptive and unadjusted for the overall history of experiments. The folds have overlapping expanding training sets. Candidate availability and the ranking algorithm are held fixed to isolate this representation; no conclusion is made that either is otherwise optimal.

## Budget and failure behavior

No download, install, cloud-resource write, Git push or submission. Existing raw files, graphs, models, results and Git checkout are read-only. The source SHA and data/runtime receipts must match first.

Features: 300-second internal budget, 320-second outer process cap; 64-session immutable chunks, first 16 sessions repeated for numerical parity. Screen: 240-second internal, 260-second outer cap, individual immutable model/metric receipts. Report: 60-second process cap. All stages emit 15-second heartbeats; the launcher terminates the child process group on timeout. Pause near a unit boundary; do not automatically retry or silently change the cohort.

Maximum worker RSS is 26 GiB. Require 4 GiB free disk space; stop rather than delete evidence to obtain it. The timeout does NOT stop the chargeable SageMaker application. The owner stops the app after downloading the return ZIP, never deletes the space.

## Decision gates

Primary pooled gain >=0.003, both fold point estimates nonnegative, and no pooled order decline permits planning an unchanged fresh-fitting-cohort and different-cutoff historical-window confirmation. An interval spanning zero stays uncertain. No automated feature promotion, final model, Kaggle submission, or claim of beating the target.

If this primary configuration fails, record the negative result and stop further ad hoc edits to these windows/weights on this cohort. Move to the next distinct information source rather than repeatedly tune recency summaries. The secondary arm remains an ablation, not a loophole in the primary gate.

## Following research rounds (not executed here)

1. **Strict rolling as-of demand/support:** use the already identified 75-feature family, construct snapshot inputs strictly before each permitted query time, and test training-only additions/ablations. Counts at the fixed Aug-16 snapshot are not rolling query-time demand. Plan an explicit construction budget before scanning the raw training data.
2. **Action-pair and direction-specific covisitation:** differentiate click→cart, cart→order and gap-defined episode transitions under correct historical cutoffs; test candidate and ranking effects separately.
3. **Latent candidate–session affinities:** audit existing item/task representations and ANN studies before any retraining; use genuinely changed aggregation/retrieval hypotheses rather than duplicate generic Word2Vec experiments.
4. **Temporal confirmation:** use another valid historical cutoff with paired controls; do not reuse later-cutoff graphs on earlier sessions. Improve uncertainty and validate a full competition-input path before spending on a new submission.

The historical private target is 0.60503; the last accepted private score recorded earlier is 0.56842. This package neither produces nor claims a new comparable submission score. A large formula catalog, successful software test, or a fitting score near 0.60503 does not close feature engineering.
