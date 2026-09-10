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
