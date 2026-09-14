# OTTO Round 06 — Repeat-versus-discovery affinity context

## Decision from Round 05

Freeze the negative result. The 134-feature shared control scored **0.5692312343**; 27 typed-transition additions scored **0.5564271438** (−0.0128040905); nine collapsed-action additions scored **0.5628419674** (−0.0063892669). Both lost on both chronological folds. Typed transitions lost 19 click hits, five cart hits and five order hits. Its descriptive paired 95% interval versus control is [−0.0239971994, −0.0027569301].

The uploaded ZIP's 66 manifest-listed files match their hashes. Pooled/fold metrics and all three 2,000-sample intervals were independently recomputed from per-session integers. Four uploaded notebooks contain no saved errors. Large native models, feature arrays and transition index shards were not uploaded; their execution cannot be independently replayed here. Do not rerun Round05 indexing or training.

This is repeatedly inspected fitting-cohort evidence, not independent validation or a Kaggle score. The negative result applies to the particular adjacent-pair features, fixed historical tail, sample, candidates and model settings. It does not prove all action-pair representations useless. Sparse purchase-type support was observed in 64 sampled training sessions per fold; this is not proof of the cause of the loss.

## Research overlap check — do not repeat failed demand ranks

An initial candidate-relative-demand idea was rejected during preparation: Round04 already tested candidate percentiles and mass shares of clicks/cart/order demand in rolling and fixed-snapshot arms. No new demand percentile or demand-mass feature is in this package. Round02 timing, Round03 recent-mass, Round04 demand and Round05 transitions are excluded. Previous older handoff descriptions of demand ranks as untested are superseded by the Round04 source and feature contract.

## The one question

Does comparing graph affinity **within products already seen versus not seen in the observed prefix** help the frozen ranker, beyond comparing affinity globally across the candidate list?

A repeated product competes with other repeated products for a potentially different reason than an unseen related product. This is a hypothesis, not a guaranteed mechanism. Repetition flags, graph scores and source ranks already exist in the project; they are not claimed as new. The new representation is the conditioning of six selected graph-affinity rank/share calculations on complete-prefix repeat status.

### Primary sources and limits of transfer

- RepeatNet, Ren et al., arXiv:1812.02646: https://arxiv.org/abs/1812.02646 — explicit repeat/explore distinction in session recommendation. We use the distinction as motivation, not its neural model or its reported improvements. This is not a RepeatNet reproduction.
- OTTO first-place implementation, `AddFeats.add`: https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/blob/main/codes/otto/scripts_prepare_2nd/prepare_feature.py — item statistics ranked within candidate sessions after joining candidates. It motivates contextual features, but does not validate this repeat-stratified graph formula. Its demand ranks were already represented in Round04.
- Project source at verified commit: https://github.com/alvaromendizabal/otto-recommender-system/blob/2638faa34afa427ed4ac4e92bba04deda57c688e/src/otto_recsys/research/domain_features.py — `full_count_share = positions.size / prefix.aid.size` on complete observed prefix. Thus `> 0` is a valid observed-membership flag, not an inference of purchase or abandonment.
- Project original graph features: https://github.com/alvaromendizabal/otto-recommender-system/blob/2638faa34afa427ed4ac4e92bba04deda57c688e/src/otto_recsys/research/features.py — historical and observed-prefix graph aggregation definitions.

## Frozen feature specification

The six nonnegative cached inputs are:

- `graph_time_all_n1_uniform_sum`
- `graph_cart_all_n5_uniform_max`
- `domain_norm_symmetric_time_row_recent_unique_mean`
- `domain_norm_forward_time_row_recent_unique_mean`
- `domain_norm_symmetric_order_row_purchase_unique_mean`
- `domain_norm_symmetric_order_row_last_mean`

Let G_i contain all candidate products with the same complete-prefix seen status as candidate i. For each affinity x:

1. If x_i > 0, average its 1-based ascending rank among G_i and divide by |G_i|. A zero maps to zero, including an all-zero group. Ties use the same midrank, never item ID.
2. Use x_i / sum(x_j for j in G_i); zero when the group has no affinity mass.

The primary has 6 × 2 = **12 features**. The ablation applies precisely these operations over all 400 candidates, ignoring seen/unseen status: another **12 features**. Each is added to the same 134-feature control, yielding 146 columns. The ablation removes conditioning information; equal width does not make it an equal-complexity placebo. All-seen/all-unseen queries match the global ablation. A singleton positive group has rank/share of one. No group membership uses target actions. These values are context scores, not calibrated probabilities.

Ranks and shares are computed on the **complete 400-candidate pool before any target-dependent sampling**, with exact candidate-permutation tests. Six inputs are frozen a priori for this screen; no new feature selection is performed on validation data. Previous selection of the 134 controls and repeated inspection of this cohort limit generalization claims.

## Controlled experiment and gates

Reuse the exact 4,096 Round02 fitting sessions, candidates, chronological folds, six-hour embargo, negative-sampling seed 20260908 and 60-negative budget. Censor training targets before the fold cutoff before negative sampling. Use complete evaluation denominators, including retrieval misses. The earlier history cutoff remains 2022-08-16 22:00 UTC (1660687200000 milliseconds).

Replay all six saved native control models with exact feature order and validate their per-session hits before any new challenger model. At most 12 real challenger models: two arms × two folds × three objectives. Fixed LightGBM LambdaRank, 150 rounds, 15 leaves and unchanged saved parameters; no algorithm search or ensemble. The test stage also uses temporary tiny synthetic models and never private experiment data.

Predeclared primary advancement gate: pooled gain ≥0.003, neither fold negative, and no pooled order-recall loss. A pass only proposes frozen confirmation on different sessions and a different historical window; it does not promote the features. The typed or global secondary comparison cannot rescue a primary loss. Bootstrap intervals are descriptive, conditional on models and reused cohorts, not adjusted for the entire adaptive research history.

The historical project target **0.60503 private leaderboard** is not comparable to this internal fitting score. The package does not generate or submit Kaggle predictions and cannot guarantee a winning score. Candidate and model capacity are held fixed, not proved optimal.

## Nine saved-result charts and additional diagnostics

Overall metric; descriptive paired intervals; per-fold gains; action recall; exact-hit contributions; how much repeat conditioning changes each feature; training-only redundancy; capped candidate oracle at20 versus achieved control; censored training-positive feature support. `context_diagnostics.json` also records training repeat/discovery candidate counts and positive exposure. Coverage decomposition is descriptive, not a causal attribution or a new achieved score.

All formulas stay in the screen. Diagnostics do not silently drop, rescale or select features. Before another experimental round, inspect evidence for absent targets, ineffective ranking of present targets, zero affinity support, and whether stratum conditioning actually changes values.

## Boundaries and checkpoints

| Stage | Useful-work cap | Outer process cap | New real model fits |
|---|---:|---:|---:|
| Tests | 60 seconds outer | 60 seconds | 0 (tiny synthetic fixtures only) |
| Features, including preflight | 180 seconds | 200 seconds | 0 |
| Control replay and screen | 240 seconds | 260 seconds | At most 12 |
| Report | 60 seconds outer | 60 seconds | 0 |

Limits are safety boundaries, not expected runtimes. No raw JSON scan, new index, historical graph rebuild, package installation, AWS API operation, S3 write, Git write or submission. Require 10 GiB free disk and stop the worker above 26 GiB RSS. Fifteen-second UTC heartbeats; 64-session feature checkpoints; one native model at a time with reload parity and immutable receipts. An orphan, checksum conflict, changed environment or source mismatch stops instead of overwriting.

A planned pause or failure is **not** authorization for an unchanged retry. Launcher receipts block that retry. Bundle and return evidence first. Completed matching artifacts are reused when a reviewed continuation is appropriate. Never run the notebook and terminal paths concurrently. `launch.py bundle` is read-only with respect to model/data artifacts and remains available after a stop.

## Storage and publication

New package/output folder: `~/otto_feature_round06`. All earlier project folders remain read-only. The reviewed GitHub main is `2638faa34afa427ed4ac4e92bba04deda57c688e`; this package is a manual attachment, **not a commit or AWS synchronization**. A non-clean or changed checkout stops; do not reset or discard work. Return ZIP includes source, logs, receipts, notebooks, small statistics and HTML, not native models or full arrays. Private checkpoints stay in the persistent space and are not automatically backed up to S3.

After saving and downloading results, stop the **otto-dev application** through Studio Running instances → Stop; do not delete the persistent space. Stopping only the notebook kernel or closing the browser does not stop the application. Storage can remain billable after stopping compute.

## Following rounds — decisions, not automatic jobs

1. **Mechanism diagnosis / continuation:** freeze this result. If positive, confirm on different sessions and history. If negative, inspect whether conditioning changed meaningful supported signals; do not tune these formulas repeatedly against these same labels.
2. **Candidate/representation compatibility:** use the actual capped candidate ceiling. Where targets are missing, audit the existing neural union and distinct retrieval sources. Where targets are present but missed, examine source-specific ranker use. Wider candidates and some graph/embedding variants previously lost; more coverage alone is not a promotion criterion.
3. **Local session and multi-intent evidence:** a bounded historical-session-neighbor or separated-intent representation is distinct from last-action adjacent transitions. Define availability, exclusions, support and a matched ablation before building it. Do not invent product categories, prices, brands or persistent users absent from the data.
4. **Temporal confirmation and scale:** only frozen, justified features advance to separate temporal/cohort checks. Scale and model/ensemble contributions remain hypotheses, not guaranteed explanations or substitute work for missing feature evidence.
