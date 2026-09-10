# Feature research: evidence and completion gate

**The feature-research gate is open.** The delivered baseline and its nine audited
temporal replications establish a useful system, not exhaustive coverage of the
important recommendation signals. Final training and project wrap-up remain blocked
by unresolved feature research. Development ranker fits are necessary to measure
feature utility and do not constitute final training.

The machine-readable inventory is [feature_research.json](../configs/feature_research.json).
It names 22 bounded areas: six have evidence for the original tested scope,
14 have unresolved questions, and two require unavailable data. These counts are
an inventory, not an estimate of percent complete. The inventory can expand when
error analysis or domain evidence reveals another high-value hypothesis.

## What the data and objective support

OTTO supplies anonymous session IDs, item IDs, timestamps and actions. Clicks target
the next clicked item; carts and orders target all distinct future items of those
types. Official weighted Recall@20 gives orders weight 0.6, carts 0.3 and clicks 0.1.
The denominator includes targets missed by candidate retrieval. Consequently, an
improved candidate ceiling can coexist with a worse final ranking.
[Official task and data specification](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md).

The baseline catalog contains 1,482 formulas, including 1,152 transformations of
short-range graph affinity. That concentration is a limitation: many correlated
formulas do not amount to many independent hypotheses. The original feature screen
used fitting data, configuration selection used a later cohort, and the frozen
evaluation has since been inspected. That evaluation must not be presented as an
untouched test for newly chosen features.

Historical action/count ratios are **event-rate proxies**, not calibrated purchase
probabilities. A timestamp gap is a gap between recorded events, not verified dwell
time. Anonymous item IDs do not encode trustworthy categories or language semantics,
and session IDs do not establish persistent customer identities.

## Evidence already obtained

| Study | Measured finding | Scope of conclusion |
|---|---|---|
| Original feature study | Selected 102-feature ranker: 0.584392 offline weighted Recall@20 | Improves matched compact controls on the published cohort |
| Nine temporal/seed cells | Every matched feature comparison is positive | Confirms the original procedure across the specified windows/seeds |
| Four Item2Vec feature arms | Best weighted pilot change is small; paired interval spans zero | No supported promotion; does not exclude embedding retrieval |
| Wider symmetric retrieval | 0.59416434 versus 0.59952343 baseline on selection | Higher coverage, worse final ranking; reject this replacement arm |
| Wider forward retrieval | 0.59499875 versus 0.59952343 baseline on selection | Higher coverage, worse final ranking; reject this replacement arm |
| Complementary graph features | All three additions scored below 0.59952343 baseline; best challenger 0.59644286 | No promotion; raw block additions do not exhaust utility-screened or normalized variants |

All selection scores above describe development data, not Kaggle scores. The accepted
competition baseline remains **0.56842 private / 0.56862 public**. Evidence paths and
checksums are recorded in the inventory. Rejected experiments remain part of the
research record and must not be silently replaced with successful-looking summaries.

## Remaining domain hypotheses

| Area | Why it could matter | Decisive next comparison |
|---|---|---|
| Complementary graph features | Raw additions were negative; useful subsets may still carry complementary information | Preserve candidates and negatives; compare fitting-only task-specific utility pruning and support-aware normalization |
| Demand drift and popularity ranks | A fixed historical snapshot becomes stale | Frozen counts versus strictly as-of updates, ranks and temporal interactions |
| Action sequence and episode context | Equal counts can describe different shopping progressions | Candidate last-action state, action transitions, repeat bursts, concentration and gap-defined episodes |
| Smoothed propensities and encodings | Rare-item count ratios are unstable | Support-aware shrinkage; any target-derived encoding uses earlier complete sessions only |
| Normalized graphs | Raw edge weights may favor universally popular products | Raw scores versus smoothed lift/PMI, row normalization, neighbor rank and support |
| Multi-hop discovery | Relevant alternatives may not be direct neighbors | Unique-positive coverage and final ranking for capped two-hop paths |
| Item2Vec retrieval | A weak ranking feature can still discover new candidates | Reuse saved vectors for an ANN union before retraining embeddings |
| Implicit matrix factorization | Latent collaborative structure differs from local sequence context | One bounded historical factorization and matched affinity/coverage comparison |
| Session neighbors | Similar historical baskets can supply complementary recommendations | Bounded overlap, inverse-frequency and order-aware historical session retrieval |
| Task-conditioned neural retrieval | Next-click and future purchase objectives differ | Current-cutoff training with appropriate multi-positive targets and fixed negative policy |
| Multiple session interests | One pooled vector can average unrelated intentions | Reuse existing vectors for distinct recent/cart/order query pools before a new encoder |
| Repeat versus new items | Revisit behavior differs from discovery behavior | Development-selected interactions/routing and disjoint error slices |
| Candidate frontier and consensus | Larger unions require a ranker able to distinguish their sources | Nested 400/800/1,200 unions, source overlap, full-query ranks, recall and cost |
| Objective-specific utility | Shared screening can suppress useful task or family signals | Independent arm screening, per-task feature sets and controlled training support |

These are hypotheses, not promised gains. The first-place implementation motivates
complementary retrieval and richer session/item relationships. The third-place authors
describe aggregating item-to-session scores with position, time and action weights,
including Word2Vec and factorization similarities. Their results do not transfer
automatically to this repository's cohort or resource budget.
[First-place code](https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System),
[third-place code](https://github.com/TheoViel/kaggle_otto_rs).

TRON motivates examining retrieval targets, negative sampling and loss construction
alongside the encoder. MiaSRec motivates frequency-aware multiple session interests,
but its reported benchmarks do not include OTTO and its next-item task differs from
OTTO's multi-objective targets. The proposed adaptations above are research hypotheses.
[TRON](https://arxiv.org/abs/2307.14906),
[MiaSRec](https://arxiv.org/html/2405.00986v1).

Text/image/product-semantic features and persistent-user attributes are excluded for
the currently verified inputs because the necessary data is absent. Other external
data must have a credible item join, historical availability and demonstrated relevance
before it enters an experiment. An LLM cannot recover that missing evidence from IDs.

## Leakage and comparison requirements

1. **Availability:** publish the latest permitted timestamp for each historical
   aggregate, graph, index or learned representation. Rolling updates must never
   expose future suffixes; conversion features need fully matured outcome windows.
2. **Target-derived features:** use chronological out-of-fold construction on complete
   earlier sessions, with an embargo where outcome windows overlap. Freeze smoothing,
   priors and unseen-item fallbacks on fitting data. Exclude a row's own future labels.
3. **Candidate identity:** do not inject true items. Feature-only arms preserve every
   candidate, label, baseline column and negative sample. Candidate-relative ranks
   and normalization must be computed on the complete candidate pool before sampling.
4. **Screening:** use fitting data only. A single-family arm must be screened against
   features actually eligible for that arm; a near-duplicate in another excluded
   family cannot justify silently dropping its only representation.
5. **Utility:** use the official pooled metric and task metrics, matched controls,
   candidate ceilings, paired session differences, and rare/new/repeat/short-prefix
   slices. Report latency and memory alongside marginal recall.
6. **Confirmation:** preregister new temporal confirmation before promoting a new
   feature set. Declare prior exposure; repeated development on the same selection
   cohort does not create an unbiased test or multiple-comparison-adjusted intervals.

## How the gate closes

Each high-value area requires a measured, bounded retain/reject decision or a concrete
data-based exclusion. A small negative pilot does not reject an entire representation
family. A resource-deferred experiment stays open. New features require confirmation
of incremental utility and a deployable implementation with the same availability rules.

`scripts/project_status.py` verifies the inventory's referenced file hashes and reports
the open gate after processing the submission receipt. `--require-feature-gate` exits
with status 2 until both coverage and temporal confirmation are complete. The check
verifies evidence presence and integrity; scientific interpretation still requires review.

Notebook 09 presents the original study and the subsequent development experiments.
Notebook 02 remains the historical retrieval benchmark. Neither execution success nor
the accepted baseline submission is used to imply that the broader research gate passed.

## Completed complementary-graph audit

The managed experiment completed at 03:47:56 UTC on September 10, 2026. It preserved
100,000 fitting sessions (6,083,582 sampled candidate rows) and 20,000 selection sessions
(all 8,000,000 candidate rows). Quality screening retained 120 of 144 proposed new columns.

| Arm | Features | Selection weighted Recall@20 | Change versus baseline, percentage points | Descriptive paired 95% interval, pp |
|---|---:|---:|---:|---:|
| Baseline | 102 | 0.59952343 | 0 | Reference |
| Add symmetric affinities | 162 | 0.59526937 | -0.4254 | [-0.7273, -0.1276] |
| Add forward affinities | 162 | 0.59592290 | -0.3601 | [-0.7305, +0.0016] |
| Add both | 222 | 0.59644286 | -0.3081 | [-0.6392, +0.0214] |

All three additions also have negative point differences for each individual action.
The intervals are descriptive and do not correct repeated development or multiple
comparisons. Neither an overall gain nor a task-specific gain supports promotion.
The accepted Kaggle baseline is unchanged.

The independent aggregate audit recomputed the official pooled metric and bootstrap
differences for every selection session, verified all 12 native model hashes, feature
schemas and tree counts, and recomputed gain importance from native model files.
Candidate coverage matched per session and objective. No cross-family screening
exclusion deprived a single-family arm of its only eligible representation.

This audit does not independently reconstruct every raw graph edge, candidate-feature
value or model prediction. Those limits remain explicit in the
[audit record](../reports/research/graph_feature_audit.json). The next graph question
concerns useful subsets and normalization, not another claim that more raw columns
will improve the result. Distinct sequence/funnel and learned retrieval questions
also remain open.

## Preregistered domain-feature comparison

The next bounded experiment is specified in
[domain_feature_study.json](../configs/domain_feature_study.json). It tests 172
new formulas, with the original 102 baseline columns retained in every arm. This
is a development comparison; the selection cohort was already used for baseline,
embedding and graph decisions. A positive result would require separate temporal
confirmation before changing the accepted competition system.

### Domain basis and limits of the evidence

The observed data support three different questions: what the shopper has done to
this item, whether recent activity differs from earlier activity in the session,
and whether historical associations reflect item-specific evidence or popularity.
These questions differ from simply adding more transformations of the same raw
co-visitation weights.

| Evidence | Relevant observation | Testable adaptation here | Limit |
|---|---|---|---|
| Official OTTO specification | Click, cart and order records have different targets and weights | Preserve action order within each candidate's observed subsequence; measure all three objectives | A cart is an observed action, not proof of purchase intent or abandonment |
| Third-place OTTO implementation | Item-to-session similarities are pooled with position, time and action information | Compare last item, recent distinct items and distinct cart/order items | Its much larger feature/model ensemble does not establish utility on this cohort |
| NISER | Embedding norms can introduce popularity bias in neural session recommendation | Test normalized historical affinities with the same graph and seed pools as a raw control | Normalizing a sparse co-visitation graph is an extrapolation, not the paper's method |
| MiaSRec | Frequency and multiple session interests can matter on session recommendation benchmarks | Use distinct-item pools and gap-defined recent activity before training another encoder | These fixed pools do not learn latent intents, and the reported benchmarks are not OTTO |

Sources: [official OTTO task](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md),
[third-place implementation](https://github.com/TheoViel/kaggle_otto_rs),
[NISER](https://arxiv.org/abs/1909.04276),
[MiaSRec](https://arxiv.org/abs/2405.00986).

The 30-minute and two-hour episode thresholds are prespecified sensitivity points,
not inferred session boundaries, validated dwell times or claimed industry rules.
They split only an already observed prefix. Anonymous IDs provide no evidence about
product substitutes, complements, price, stock, returns, customer identity or
demographics. No such semantics are imputed into these features.

### Exact feature definitions

All event timestamps are milliseconds. Durations are divided by 3,600,000 to obtain
hours; episode thresholds in seconds are multiplied by 1,000. Action codes are
click 0, cart 1 and order 2. Original event ordering breaks timestamp ties. The final
observed event has no observed future gap, so no future dwell duration is imputed.

For a prefix of length N and candidate a, its item subsequence retains every observed
event for a in order, even when other items occur between them. The 40 funnel
columns comprise:

- Presence, count/N, number of distinct observed actions, reciprocal rank among
  distinct items in reverse recency, log(1 + hours since last occurrence),
  log(1 + number of events since last occurrence), and the item's observed time
  span divided by max(1 millisecond, full-prefix span).
- Whether the last cart is later than the last order; whether the last click is
  later than the last cart or order; log counts of clicks after the last cart and
  carts after the last order. A missing prerequisite yields zero. These describe
  recorded action order and do not infer an unobserved purchase funnel.
- The terminal consecutive run of a divided by N; full-prefix item entropy,
  squared-share concentration, and the fraction of adjacent events switching item.
  Context columns are broadcast across candidates and can support tree interactions;
  alone they cannot change the ordering within one query.
- Three indicators for the last action and three for the penultimate action in
  the item's subsequence. The penultimate indicators are zero with one occurrence.
- For each of the nine ordered action pairs, log(1 + observed transition count)
  and log(1 + mean transition gap in hours). Empty transitions are zero. These are
  successive events for that item, not necessarily adjacent full-prefix events.

For each of the two gap thresholds, a new episode begins only when an adjacent
observed gap is strictly greater than the threshold. The 12 columns per threshold
are candidate presence, current-episode count share, distinct actions, last-cart
without-later-order flag, previous-episode presence, log count of distinct earlier
episodes containing the item, and log age within the current episode. Five query
columns encode log event count, distinct-item count, duration in hours, episode
count, and the gap opening the current episode. Every log uses log(1 + x). Candidate
statistics with no supporting occurrence are zero. The query context remains
available for unseen candidates. This contributes 24 episode columns.

For each certified graph family (symmetric and forward) and score channel (time,
cart and order), let w(s,a) be a retained edge weight, R(s) the sum of retained
outgoing weights, and C(a) the sum of retained incoming weights. We compare raw
w(s,a), row-normalized w(s,a)/(1 + R(s)), and degree-normalized
w(s,a)/sqrt((1 + R(s))(1 + C(a))). The fixed additive support is one in the graph's
score units. These are retained-graph scores; pruning means their marginals are
not full interaction counts or calibrated probabilities. Lift/PMI and learned
shrinkage remain separate, untested hypotheses.

Each score is pooled by mean and maximum over three seed sets: the last item;
the last 20 distinct items in reverse observed recency; and the last 20 distinct
items with an observed cart/order action. Distinct pools avoid weighting a repeated
item multiple times. Unknown items and empty pools contribute zero; the mean
denominator includes every selected seed, including seeds without an edge. Both
normalizations use marginals computed from the fixed historical graph, so changing
the candidate list does not change an existing candidate's score. This contributes
36 raw control columns and 72 normalized columns.

### Frozen comparisons and analysis

| Arm | Eligible additions before fitting-only screening | Purpose |
|---|---:|---|
| Baseline replay | 0 | Re-predict all selection candidates using the saved three baseline models |
| Sequence | 64 | Action progression plus gap-defined episode context |
| Raw graph control | 36 | Distinct recent and purchase seed pools, without normalization |
| Normalized graph | 72 | Row and degree normalization over those same pools |
| Combined | 136 | Sequence/episode features plus normalized graph evidence |

The normalized-versus-raw contrast controls the graphs and pools, but differs in
feature count and uses two normalization modes. It therefore compares feature
packages, not the isolated causal effect of a single formula. A favorable package
would need mode-specific and task-specific ablations. Likewise, the sequence arm
does not isolate transitions from episodes; both remain follow-up questions.

All arms use the original candidates, candidate order, targets, baseline columns
and sampled fitting negatives. The new cache verifies their equality part by part,
including on resume. No true item is injected. Baseline replay requires byte-pinned
native models and identical hits/coverage for every original selection session.
No unchanged baseline fit is repeated.

Each challenger is screened independently against only its own eligible features
on 100,000 evenly spaced fitting rows. The label-free screen removes nonfinite,
constant and almost perfectly correlated columns (absolute correlation at least
0.99999); original baseline columns remain fixed. This is quality/redundancy
screening, not per-objective utility selection. Candidate and label arrays cannot
enter the pure feature functions. Historical graph cutoffs must match the corpus
and precede every observed query.

Each challenger fits three LightGBM rankers with the frozen baseline settings:
400-round cap, 40-round patience, evaluation every five rounds, seed 20260908 and
32 threads. The checkpoint records every completed iteration and retains the best
complete-query Recall@20 iteration. Training configuration, code, graphs, reference
models and cache identities enter the immutable study contract. Checksum or contract
changes stop recovery instead of silently mixing runs.

Report pooled official Recall@20, individual objectives, full-query candidate
ceilings and paired whole-session bootstrap differences for every arm, plus the
normalized-versus-raw comparison. Percentile 95% intervals are descriptive,
unadjusted for repeated selection or multiple comparisons. A negative finding stays
in the record. Native-model gain is descriptive and cannot establish causal utility
of one column. The independent audit recomputes metrics and intervals from complete
session statistics and verifies model bytes, schemas, iterations and lineage; it
does not independently reconstruct every raw feature or replay challenger predictions.

### Execution and recovery

The managed job reuses the existing corpus, original model-input archive, both
historical graphs and baseline models. Feature parts have atomic checksum receipts;
models retain resumable iteration checkpoints. UTC progress records and 15-second
heartbeats identify the active stage. One processing instance has a 7,200-second
runtime cap. The verified processing SKU and instance-only compute cap are recorded
in [domain_feature_pricing.json](../reports/research/domain_feature_pricing.json);
storage, requests, logging and any transfer charges are additional.

The completed outcomes below retain the preregistered design and distinguish measured
results from post-hoc diagnostics. The broader feature-research gate remains open.

### Observed support for the domain hypotheses

A verified observed-prefix census, computed before inspecting the new model results,
shows that opportunity is uneven across queries.

| Observed property | Fitting sessions (100,000) | Selection sessions (20,000) |
|---|---:|---:|
| One event only | 45,482 | 9,396 |
| At least one repeated item | 31,665 | 5,934 |
| Multiple action types on the same item | 14,233 | 2,700 |
| Any cart or order | 15,514 | 2,997 |
| At least one gap over 30 minutes | 10,747 | 1,308 |
| At least one gap over two hours | 7,645 | 729 |

Thus 46.98% of selection queries have no observed action transition, and only 6.54%
have more than one 30-minute episode. These are support counts, not estimates of
feature utility. In a one-event prefix, recent-distinct and last-item graph pools
are identical, although normalization can still change cross-candidate evidence.
The smaller selection share of long-gap prefixes also makes direct extrapolation
from fitting-support counts unreliable. Exact counts, quantiles and observed-file
hashes are in `reports/research/domain_prefix_profile.json`.

Descriptive error slices use the original prefix-length boundaries (1, 2–5, 6–20,
21+), repeated versus all-distinct observed items, observed cart/order versus
click-only prefixes, and gaps above versus at most 30 minutes. These boundaries
were fixed before new model outcomes were read. Each pair or length partition
preserves complete sessions and target denominators; different slice families
overlap. They do not measure target-item repeat/new recall or item rarity.

### Completed domain comparison: September 10, 2026

The managed run completed successfully and the independent aggregate audit passed.
It fitted twelve new models and replayed three original baseline models. Every arm
uses 100,000 fitting sessions (6,083,582 sampled candidate rows) and the same 20,000
selection sessions (8,000,000 complete candidate rows). All 15 native model files,
feature schemas, stopping iterations and input lineage passed verification. Candidate
coverage is identical for every session and action; the weighted candidate ceiling
is 0.697271. The source commit used by the job is
`4336737fce1f7bc00bde939a0ec05e1dbeca1f06`.

| Arm | Retained columns | Weighted Recall@20 | Change (pp) | Descriptive paired 95% interval (pp) |
|---|---:|---:|---:|---:|
| Baseline replay | 102 | 0.599523 | +0.0000 | Reference |
| Sequence + episodes | 162 | 0.599310 | -0.0214 | -0.2572 to +0.2204 |
| Matched raw graph pools | 132 | 0.596952 | -0.2571 | -0.5810 to +0.0817 |
| Normalized graph pools | 162 | 0.599941 | +0.0417 | -0.3739 to +0.4609 |
| Sequence + normalized | 222 | 0.599598 | +0.0075 | -0.4040 to +0.4080 |

All four weighted difference intervals include zero. The normalized arm has the
highest weighted point estimate, but no supported overall improvement was established.
Its direct difference from matched raw graph pools is +0.2989 pp, with a descriptive
95% interval of −0.1098 to +0.7317 pp. Neither comparison justifies promotion.

The action-level results explain the cancellation in the weighted score. Sequence
features increase click recall by 0.0676 pp and cart recall by 0.0843 pp, while order
recall falls by 0.0890 pp; all three intervals include zero. Normalized graph features
increase order recall by 0.2671 pp but reduce click recall by 0.7805 pp. The click
interval lies below zero (−1.1450 to −0.4172 pp), while the order interval includes zero.
These intervals are unadjusted and come from an already reused selection cohort.
The combined arm does not resolve the trade-off. Cart models in the normalized and
combined arms, and the combined click model, select the first boosting iteration;
this is an observed stopping outcome, not evidence of stable feature utility.

A post-hoc diagnostic selects the best existing arm separately for each action:
sequence for clicks/carts and normalized graph for orders. It produces **0.601446**
on this same selection cohort, **+0.1923 pp** above the replayed baseline. This was
computed after viewing outcomes and is not a sixth preregistered arm, an independently
validated gain, or a new fit. Selecting on the reporting cohort creates optimism;
[Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) explain why
model-selection variance must be included in evaluation design. The mix is a follow-up
hypothesis only. Its exact selection rule and component models are recorded in the
slice report.

Prefix diagnostics also warrant restraint. The sequence arm improves the one-event
slice despite the absence of an observed transition; the model also uses static
last-state/context features and changes its learned splits. That slice cannot establish
that action transitions caused the gain. The normalized arm's positive long-prefix
point estimate is based on only 409 sessions with 21+ observed events. The cohort
partitions conserve per-action hits and full target denominators; their weighted
recalls must not be averaged by session counts to reconstruct the pooled score.

**Next controlled work:** reuse the certified caches, separate funnel from episode
features and row from degree normalization, and screen feature utility separately for
each action using fitting-only session groups. Freeze the selected schemas, models and
routing rule before a separately declared temporal confirmation. This experiment used
one seed, one development cohort and redundancy screening; it does not settle those
questions or the other open families in the coverage inventory. Feature engineering
remains open, with no final training or Kaggle promotion.

The [results](../reports/research/domain_feature_results.json),
[audit](../reports/research/domain_feature_audit.json),
[screening](../reports/research/domain_feature_screening.json),
[slices](../reports/research/domain_feature_slices.json) and
[completed run receipt](../reports/research/domain_feature_run.json) contain exact
values and hashes. The canonical [Notebook 09](../notebooks/09_controlled_feature_study.ipynb)
recomputes report consistency checks and presents the comparisons.

The job used 3,013.228 managed processing seconds (50.22 minutes), estimated at
**$2.868593056 in instance compute** at the verified $3.4272/hour rate. This is not an
invoice and excludes storage, requests, logs and transfer. No replacement compute run
was needed to recover the results. All 64 collected checkpoint files passed size and
SHA-256 checks before the independent audit. Source, launch, caches, logs and models
remain checkpointed under the receipt's owned S3 prefix.

During this run, matrix loading before each arm produced a quiet interval. The current
source extends 15-second heartbeats across that stage for future runs. The completed
job retains its original published source identity; it did not use that later logging
change.

## Bounded shortlist pilot: completed

A three-arm follow-up has now tested fitting-only shared versus per-task selection
on cached domain features. Shared selection scored **0.604069**, per-task selection
**0.599860**, and the matched small-data baseline **0.578084** on 2,048 development
sessions. The shared gain is +2.599 pp, primarily orders. Per-task selection did not
beat the shared control. These numbers are not Kaggle scores or temporal confirmation.

The completed managed job used 4.50 processing minutes, about $0.257 in instance
compute. Nine models passed independent metric/hash checks and a zero-training
restart reproduced every result. [Full pilot, uncertainty and decisions](TASK_FEATURE_PILOT.md).

The next bounded stage freezes the 32 shared additions and compares them with the
baseline using more fitting support; block ablations and temporal confirmation follow
only if warranted. The exact per-task rule will not be scaled. All 14 unresolved
families remain open. [Standing execution rules](EXECUTION_RULES.md).
