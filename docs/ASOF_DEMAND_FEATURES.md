# As-of demand and candidate-relative support features

## Research hypothesis

Strong OTTO systems repeatedly exploit item/session demand, action-specific counts, recency, and ranks in addition to co-visitation. The third-place solution describes hundreds of candidate-session/item-item features and a public feature-engineering solution records item statistics by hour/weekday, occurrence rates, action-type distributions and ranks. The first-place implementation is retained as a primary comparison target in the project research inventory. These patterns motivate a **distinct** family from the already-tested retrieval-source ranks: query-time, strictly historical demand/support evaluated relative to the *full candidate frontier*.

This family is not a claim that popularity alone wins. Earlier project work already showed that broader retrieval can improve coverage while harming final ranking, and static/reference-window graph variants can regress. The hypothesis is narrower: **recent demand drift and full-candidate relative support may help the ranker decide which retrieved items deserve top-20 positions.**

Sources:
- OTTO dataset/task: https://github.com/otto-de/recsys-dataset
- First-place implementation: https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System
- Third-place solution: https://github.com/TheoViel/kaggle_otto_rs
- Public solution describing item/session/hour statistics, action distributions and ranks: https://github.com/nlztrk/OTTO-Multi-Objective-Recommender-System

## Leakage and availability contract

`DemandSnapshot` accepts only item/action counts for events **strictly before** `cutoff_ts`. The query must occur at or after that cutoff. The transform accepts no labels or future suffixes. Candidate-relative percentiles and mass shares are computed across the complete candidate set before any target-dependent fitting-negative subsampling.

Nested 1/6/24/72/168-hour counts must be monotone as windows widen. Last-event timestamps must be strictly before the snapshot cutoff. Unknown candidates receive zero counts and explicit unseen recency. Action-mix values use priors estimated from the same certified historical snapshot and are descriptive intensity statistics—not causal click-to-cart or cart-to-order conversion estimates.

A production snapshot builder is intentionally not bundled into the transform. That separation forces the raw historical source/cutoff identity to be proven before a snapshot can be used.

## Implemented candidate family

The initial catalog contains **75 interpretable candidates**:

- action-specific log counts over 1/6/24/72/168 hours;
- tie-aware within-query demand percentiles;
- within-query candidate demand mass shares;
- short-vs-long per-hour log-rate drift for 1-to-24, 6-to-24 and 24-to-168 hours;
- the same short-vs-long changes in candidate-relative percentile;
- support-smoothed action mix over 24 and 168 hours;
- action-specific last-seen availability and log age.

These are candidate features, not 75 retained features. Large catalogs are not evidence of value.

## Evidence plan

1. **Fixture/unit gate.** Temporal cutoffs, monotone windows, unknown items, candidate uniqueness, tied ranks, deterministic replay, action-mix normalization, finite output and no input mutation.
2. **256-query smoke.** Build a snapshot only from preflight-certified history, measure wall time/peak memory, and verify deterministic complete-candidate matrices.
3. **Fitting-only screening.** Report support/nonconstant rates and redundancy against the frozen baseline/shared representation. Any supervised screening uses fitting sessions only.
4. **Controlled family ablation.** Same candidate set, negatives, model capacity and seeds; compare baseline/shared versus shared+as-of, then drop subgroups if justified.
5. **Temporal stability.** Repeat the frozen family on a second fitting window before promotion. Selection/evaluation labels may measure a frozen hypothesis but may not redefine formulas.

Use the official pooled weighted Recall@20 and action-level hits/denominators. A positive coverage change without achieved ranking improvement is not success.

## Stop conditions

Stop rather than scale if the snapshot cannot prove its temporal cutoff, formulas duplicate existing information without incremental fitting evidence, signs are unstable across fitting folds/windows, or construction throughput/memory is incompatible with the bounded confirmation plan. Do not rescue a failed result by changing formulas on the inspected selection cohort.

## Current status

**Implemented and unit-tested; not screened on real OTTO fitting data, not ablated, and not promoted.** Correct-cutoff historical graph/feature reconstruction remains a separate open blocker for the frozen 102/134 confirmation.
