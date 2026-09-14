# Feature-first research plan and Round 01 preregistration

## Basis and limits

The returned workspace report establishes source/data readiness, not predictive improvement. GitHub `main` was read at `2638faa34afa427ed4ac4e92bba04deda57c688e`. The repository's prior feature-value pilot is completed: 54 models, two forward folds inside fitting data, and 0.48473688289545064 for the 134-feature shared arm. Its model settings, fold identities, input hashes and saved per-session hits provide the fixed control here. The existing three-fold score and competition scores use different protocols and must not be compared numerically as gains.

The final historical private target is 0.60503; the latest valid submitted score recorded in project evidence is 0.56842. We have not demonstrated a better valid submission. Future-contaminated postcompetition test events must never be used in research, features or inference intended for that comparison.

## Domain research → hypothesis

The first-place author's account uses multiple action/time-weighted covisitation signals, multistep candidates and target-conditioned neural representations. The third-place component explicitly reports 744 features formed by item–item similarity, weighting by position/timestamp/action, and aggregation over the observed session. These support investing in richer candidate–session representations rather than assuming a particular model algorithm is the sole missing ingredient. They do NOT prove the 48 proposed formulas will improve our model.

Inspected existing source already has sum/max/mean/std graph aggregates, historical demand, repeated-item features, funnel transitions, pending-cart indicators, and raw/row/degree-normalized graph mean/max over several pools. Rather than duplicating those families, Round 01 asks whether the model lacks the **breadth, concentration and timing of support from OTHER distinct observed products**, particularly products with a cart event after their latest observed order. This is a concrete representational hypothesis, not an exhaustive claim about every repository formula.

## Forty-eight explicitly defined columns

Two historical graph families (symmetric and forward) × three channels (time, cart, order) × two observed-prefix pools × four summaries = 48.

Pools use at most 20 distinct items, ordered by last observation:
- `recent_unique`: the 20 most recently observed distinct items.
- `pending_cart`: the 20 most recently observed distinct items whose latest observed cart is after their latest observed order. A later click does not clear this state, a later order does, and a later cart reopens it. This is NOT actual cart inventory or inferred abandonment; cart-removal data are absent.

For each candidate, exclude that candidate's own item from the seed pool. Unknown catalogue seeds have zero affinity but remain in the eligible denominator. Positive support describes retained certified graph edges, not the whole latent product graph.

For affinity vector w across eligible seeds and query-time age a in hours:
1. Coverage: count(w>0) / number of eligible other-item seeds; zero when none.
2. Evidence entropy: -sum(p log p) / log(max(2, eligible seed count)), where p=w/sum(w); zero when there is no positive mass.
3. Affinity-weighted mean of log(1+a).
4. log(1+a) for the strongest supporting seed, with exact ties broken in favor of the most recently observed seed; zero with no support.

Coverage/entropy form the 24-column support group. The two age summaries form the 24-column timing group. A no-support zero age is not interpreted as evidence of a recent interaction: support columns are available, and the timing-only ablation tests whether those summaries work without them.

## Availability and leakage

All graphs must match the certified **2022-08-16 22:00 UTC exclusive** cutoff. Feature generation only reads observed prefixes, existing candidate IDs and those graphs. It does not consume labels, selection/evaluation roles, the official test, or the contaminated full test. Full target ledgers are loaded only in the supervised screen. Target timestamps must follow the observed prefix; tied timestamps require a later event index. Training targets are censored at the fold cutoff BEFORE sampling negatives.

This uses the previously inspected 1,024 fitting sessions and two existing forward folds. It is not a new untouched holdout or a deployment-faithful online backtest. Validation horizons remain those of the prior matched pilot so that the control is comparable. Another independent temporal replication and stronger order support are necessary before retention.

## Controlled experiment

Hold fixed: 400 complete candidates/session, 1,024 fitting sessions, training/validation identities, six-hour embargo, training-negative policy (60 negatives, seed 20260908), LightGBM parameters, 150 rounds, model seed 20260911, and full-target pooled Recall@20.

Reuse the six saved 134-feature objective/fold models' validated per-session hit counts. Recompute their aggregate metrics after verifying the source and exact fold IDs. Do NOT refit them. New arms:
- shared + support (158 columns)
- shared + timing (158 columns)
- shared + both (182 columns)

Three arms × two folds × three objectives = at most 18 new models. The two group-only arms also supply matched group-drop comparisons against the full addition. Primary comparison is all 48 against control; group results are secondary and descriptive. The original 102-feature score is reference history, not the control for this incremental test.

Before fitting, report training-only support, variance and correlation against the 134 existing columns on a deterministic sample of up to 64 training sessions/fold. An entirely constant group stops before wasted fits. Individual flags do not automatically select/remove features. Full validation candidate pools and full capped target denominators remain intact.

Native model reload must exactly reproduce predictions. Matrices/checkpoints carry code, protocol and input hashes. A successful rerun reuses completed models; differing contracts stop rather than overwrite.

## Decision gate

For a new arm: negative/zero pooled gain stops that exact variant; a negative time-fold or orders change requires diagnosis before scale; a stable positive gain below 0.003 is lower priority. Gain ≥0.003, nonnegative folds and no order decline justify only a larger fitting-only replication proposal. An interval spanning zero is explicitly uncertain. No feature is promoted automatically. Bootstrap samples paired validation sessions within folds; intervals are descriptive and unadjusted for multiple comparisons. There are only 92 pooled order-denominator units, so small changes can be dominated by a few orders.

A larger follow-up must be preregistered after inspecting these outcomes; do not change feature formulas, fusion weights or model settings repeatedly against this same small cohort. Preserved negative results are useful scientific outcomes.

## Subsequent rounds, not executed or silently bundled here

| Round | High-value question | Required evidence before retention |
|---|---|---|
| 01 — this package | Do other-item support breadth and evidence age add information? | Matched add/drop effects and time-fold signs |
| 02 — rolling demand/support | Do strict as-of 1/6/24/72/168-hour action counts, relative ranks and demand drift improve over a stale fixed snapshot? | Timestamp/source audit, current-query exclusion, only-past availability, fitting-only ablation and time stability |
| 03 — action-conditioned structure | Do click→cart, cart→order and directional/time-window graph variants represent intent beyond generic similarity? | Certified historical graphs; source/target action ablations; candidate coverage distinguished from ranking gain |
| 04 — latent similarities | Do existing task-conditioned or item representations help under candidate/session weighting rather than another generic embedding fit? | Reuse prior checkpoints where cutoffs permit; isolate candidate and ranker effects; evaluate drift and incremental utility |
| Confirmation | Which positive families survive more fitting sessions, another time window and seeds? | No selection/test tuning; paired estimates, practical cost and final production-compatible inference |

Keep the separate candidate-coverage/frontier track open; this package does not include or repackage its previously blocked runner. Candidate pool size is not by itself evidence of quality. Ensembling is a later additive question, not a substitute for valid representation and data.

## Primary sources

- Official task and evaluation: https://www.kaggle.com/competitions/otto-recommender-system
- Final leaderboard: https://www.kaggle.com/competitions/otto-recommender-system/leaderboard
- First-place writeup: https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution
- Third-place author's implementation: https://github.com/TheoViel/kaggle_otto_rs
- Session-based Recommendation with Graph Neural Networks (SR-GNN): https://arxiv.org/abs/1811.00855 — motivation for combining structural context with current interest, not validation of these handcrafted formulas.
- Existing project definitions inspected: `src/otto_recsys/research/features.py`, `domain_features.py`, `graph_signals.py`; previous controlled protocol and helpers: `scripts/run_feature_value_pilot.py` at commit `2638faa...`.
