# Round 08 — Rarity-weighted historical-session evidence

## Research question and falsifiable hypothesis

Do history-based candidate features improve when a rarer observed product contributes more to historical-session similarity than a very common product? Round07 established usable item/action support but no ranker gain. Its wider session-context and last-product proposals recovered the same seven extra target hits. That is not grounds to enlarge the candidate pool indiscriminately. This round instead holds candidates fixed and asks whether selecting and aggregating neighbors differently makes the existing candidate evidence more discriminative.

A product shared by many historical sessions may be a weak indicator of the query's particular intent. A rarer shared anchor can be more specific, although it can also be noisy. We do not assume that rarity is universally helpful: the width-matched ablation removes rarity weighting while keeping the same source pool, recency weighting, number of neighbors and aggregation formulas. This is a controlled test of a representation, not a claim that the underlying historical source is entirely new.

## Neighbor retrieval and similarity

For an observed query, choose its most recent four distinct products, or all available when fewer exist, in reverse chronological order. Call the ordered anchors a_j with j starting at one. Query recency contributes 1/j. The primary additionally divides that weight by sqrt(max(1, df(a_j))), where df counts all eligible historical sessions containing the product before the per-anchor 64-session posting cap. Thus the primary query weight is q_j = 1 / (j * sqrt(max(1,df(a_j)))). The equal-weight ablation uses q_j = 1/j. This is inverse-square-root session support, not TF-IDF, calibrated confidence or a purchase probability.

The historical vector is binary over unique retained products. Its similarity is the sum of query weights on overlapping anchors divided by the query vector norm and sqrt(number of unique historical products). Only overlapping sessions enter. Select at most 64 by descending similarity, then descending final retained timestamp, then ascending session ID. Both arms search the same union of at most 64 historical postings for each anchor; different rarity weights may legitimately change their selected neighbors. This effect is part of the declared treatment rather than an uncontrolled data change.

An unsupported anchor has no postings. Missing evidence gives zeros; an existing history overlap paired with zero declared frequency is an integrity error. Frequencies are computed without current query outcomes. Chronological exclusion and full-history cutoff are verified before the pure feature function is called.

## Exactly eight statistics for each of three actions

Let N be the selected neighbors, w_s their nonnegative similarity, r_s their one-based neighbor rank, k_s the count of query anchors present, A the number of query anchors, and I_s(c,t) indicate whether candidate c occurred with action t in retained historical session s. Let W=sum(w_s) and R=sum(1/r_s). Every item/action contributes at most once per source session even when repeated many times.

| Statistic | Definition | Interpretation |
|---|---|---|
| log_session_support | log(1 + sum I_s) | Distinct historical-session evidence |
| weighted_vote_share | sum(w_s I_s) / W | Total similarity mass supporting the item/action |
| max_similarity | max supporting w_s, otherwise 0 | Strength of the closest supporting neighbor |
| reciprocal_neighbor_rank_share | sum(I_s/r_s) / R | Evidence near the top of the neighbor list |
| multi_anchor_vote_share | sum(w_s I_s 1[k_s>=2]) / W | Support requiring at least two query anchors |
| after_anchor_vote_share | sum(w_s J_s) / W | Support after the final retained occurrence of any matched anchor |
| log_effective_support | log(1 + (sum w_s I_s)^2 / sum(w_s^2 I_s)) | Whether support is concentrated in one neighbor or distributed |
| anchor_coverage_vote_share | sum(w_s I_s k_s/A) / W | Support adjusted by breadth of query-anchor agreement |

J_s is one when the candidate/action occurs at an original event index larger than the final matched-anchor index. A candidate may also occur before the pivot; the statistic asks whether any qualifying occurrence exists, not how often it repeats. The effective-support formula is a descriptive concentration measure, not an independent effective sample size for statistical inference. Empty denominators yield zero. No additional smoothing or feature selection is tuned against results.

Eight statistics times clicks/carts/orders produce 24 primary features and the same 24 aggregation types for the ablation. Full names, formulas and zero conventions are in `feature_catalog.json`. The worked notebook provides explicit synthetic values for both arms and separate views of action support, similarity and evidence concentration.

## Interpreting a result

A win versus the original control would support further testing of this complete rarity-conditioned representation. A win only versus the equal-weight ablation would not justify retention. If both lose, inspect positive support, common-anchor scarcity, redundant columns and neighbor identity overlap before deciding whether any distinct extension is worth testing. Do not sweep rarity exponents on these same validation labels. The fixed exponent is 0.5, not a hyperparameter search. Because the original model is held fixed to isolate feature information, a negative result cannot prove all neighborhood methods ineffective or establish that model capacity is optimal.

## Frozen comparison and leakage boundaries

Both rounds use exactly the saved Round02 cohort of 4,096 fitting sessions and the original 400 candidates per query. The original 134 columns are retained in their original order; the failed timing, recent-mass, demand, typed-adjacency and repeat-conditioning additions are not reintroduced. Each round adds exactly 24 columns to the control, separately for its primary arm and its 24-column information-removal ablation. Every challenger therefore has 158 features. There is no 182-column combined representation, no ensemble, no new algorithm and no negative-sampling change.

All six saved controls must replay their per-session validation hits before the first challenger fit in each round. Each round allows twelve new native LightGBM models: two representations, two chronological folds and three objectives. The LambdaRank settings, seed, 150 boosting rounds, 60-negative budget, sampling seed and six-hour query embargo are frozen to the verified original protocol. Target censoring precedes target-dependent negative sampling. Validation uses all 400 candidates, not sampled negatives; the complete original denominators are preserved. New model reloads must reproduce predictions. Immutable model/input hashes govern reuse, and conflicting or orphan checkpoints cause a stop.

Historical rows must be strictly before 1660687200000 milliseconds (2022-08-16 22:00 UTC). All 5,120 current/prior study session IDs are excluded. Query anchors are derived only from the observed query prefix. The historical source is the certified retained Parquet tail, not complete raw historical sessions. Original event indices are retained: gaps cannot be compressed into false adjacency. No selection, evaluation-role, competition-test, or external customer data is read. The source has no verified prices, product taxonomy or persistent customer identities; these are not fabricated.

New features are computed on the entire candidate pool before labels are accessed by the supervised stage. Feature-support and redundancy diagnostics use only chronological training subsets; support targets are censored before each validation cutoff. All 24 primary features stay in the declared test regardless of the diagnostics: there is no label-driven cherry-picking or automatic feature removal.

## Shared extraction, separate research records

Round08 owns `~/otto_feature_round08/outputs/shared`. It first verifies the actual Round07 pilot, postings, history binaries and their receipts in your workspace. The current return archive omitted these large binaries, so a small receipt-only review is not treated as proof that they exist or are unchanged on disk. The 256 earlier pilot queries must embed exactly within the full-cohort anchors and candidate arrays.

All 502 earlier pilot anchors, including unsupported anchors, are reused. Previously examined items are not scanned again. At most one additional historical Parquet scan finds postings for new query anchors, and one additional scan collects missing historical sessions. The 64 most recent eligible sessions per anchor are retained; uncapped distinct-session frequencies are counted before this truncation. There is no raw JSON scan and no old co-visitation graph rebuild. Up to 400,000 distinct historical sessions and 10 million retained events are allowed; exceeding either stops instead of silently trimming. This source sampling is a cost constraint, not exhaustive nearest-neighbor retrieval.

Round09 reads that completed shared cache without changing it. Both rounds have their own 64-query feature checkpoints, feature contracts, twelve-model inventory, per-session statistics, nine-chart result report and result ZIP. Round09 does not depend on Round08 gaining score and does not inherit a selected model or retained feature set. Complete Round08 operationally before Round09's screen; a negative but valid result still permits the second predeclared test. An operational failure or timeout blocks continuation across both packages.

## Metrics, evidence, and decision

The official-style internal calculation uses weights 0.1 clicks, 0.3 carts and 0.6 orders with per-session target denominators capped at 20. Results are pooled from saved integer hits and denominators, not averaged arbitrarily across folds. Reports independently replay three pooled scores, each chronological fold, action recall, exact hit differences and three paired bootstrap contrasts. The 2,000 stratified paired resamples yield descriptive 95% intervals; they do not adjust for adaptive experiment selection, all temporal dependence, or the multiple rounds already examined.

The advancement rule is primary-minus-saved-control pooled gain at least 0.003, nonnegative gain on both folds, and no pooled order-recall decline. Primary-minus-ablation is a secondary scientific contrast, not a rescue for losing to the control. A pass earns a separate frozen temporal/cohort confirmation, not automatic promotion. The same repeatedly investigated fitting cohort is not an untouched holdout. An order improvement rests on a small number of events and must not be oversold. An increased candidate oracle is a bound, not achieved ranking performance.

Round07's training-only pilot had an oracle of 0.552793 before expansion and 0.559621 after either candidate proposal representation: six additional clicks, one cart, zero orders. It fitted no ranker. The control's earlier 0.569231 and the recorded historical private target 0.60503 are different evaluation settings from that pilot. No internal feature run alone establishes that the historical winning score has been exceeded.

## Runtime, storage and reproducibility

Useful-work budgets are 180 seconds for each shared-data stage, 240 seconds for feature generation and 240 seconds for the screen. Outer subprocess caps are 200, 260 and 260 seconds respectively. Tests and report generation have 90-second outer caps. These are stop limits, not runtime estimates or dollar guarantees. Every data-stage entry checks 10 GiB free disk; DuckDB uses 6GB memory with a 3GB spill cap; the launcher stops a worker above 26 GiB RSS. Fifteen-second UTC heartbeats and completed chunk/model counters expose progress. Instance billing continues until the application is stopped.

Data and model units have checksums and immutable receipts. At most one stage across both packages can hold the shared execution lock. No `all` command exists, so the package cannot automatically chain both rounds into a long unattended run. A pause, hard limit, previous failed run, corrupt file or changed schema means stop and bundle evidence. Do not rerun an unchanged failure, edit the limits, delete orphan evidence or overwrite a prior folder.

The package does not synchronize Git, launch cloud resources, upload S3 objects, edit IAM, install dependencies or submit to Kaggle. It reads the verified existing repository and environment. New outputs remain on the existing persistent space and are not automatically backed up. Return ZIPs include logs, notebooks, small statistics, code, manifests and index/model receipts, but omit large source, feature and model binaries. Save notebook execution outputs before the final bundle command. Stop the JupyterLab application after downloading; do not delete its persistent space.

## Test scope

Local tests exercise independent numerical fixtures, identical extraction SQL through a clearly labeled SQLite adapter, leakage constraints, source/candidate permutation invariants, storage integrity, six-control replay, native LightGBM training/reload on synthetic data, complete denominators, resumption and report generation. They are not a private-data result. The actual installed DuckDB/Parquet backend is exercised by the mandatory tiny smoke test in the AWS tests stage. Backend failure stops instead of falling back to SQLite for real data. The existing environment is preserved.

## Primary research and implementation references

- Ludewig & Jannach, *Evaluation of Session-based Recommendation Algorithms* (2018): https://arxiv.org/abs/1803.09587
- Ludewig, Mauro, Latifi & Jannach, *Empirical Analysis of Session-Based Recommendation Algorithms* (2019): https://arxiv.org/abs/1910.12781
- Garg et al., *Sequence and Time Aware Neighborhood for Session-based Recommendations*, SIGIR 2019, DOI: https://doi.org/10.1145/3331184.3331322
- Gupta et al., *NISER: Normalized Item and Session Representations to Handle Popularity Bias*: https://arxiv.org/abs/1909.04276
- AWS application shutdown: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html

Nearest-neighbor and sequence-aware recommendation research motivates these classes of information. NISER discusses popularity bias, not this experiment's exact rarity formula. None of these sources validates our feature count, cutoff, sampling cap, windows, smoothing, expected gain, or ability to beat OTTO's winning score. Our formulas are explicit hypotheses rather than reproductions of published scores.
