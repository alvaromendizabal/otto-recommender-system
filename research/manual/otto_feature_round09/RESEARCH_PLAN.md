# Round 09 — Directed historical-session continuation

## Research question and falsifiable hypothesis

Does explicitly identifying what happened just after the query's most recent product in similar historical sessions improve the existing candidate rankings? Round07 supplied support, but whole-session membership can conflate preceding context with subsequent shopping behavior. A product examined before the shared anchor may be less informative about what this query will do next than a product immediately after it. This round tests that directional distinction without requiring a Round08 winner or changing the model.

The hypothesis is deliberately not another global adjacent action-pair count. Round05 tested directed adjacency conditioned on source and destination actions; this round conditions on similar retained sessions, locates the latest query product within each session, allows original event gaps one through five, and supplies gap, elapsed-time and agreement statistics. Directional continuation might still be too sparse, particularly for orders. An unsigned-window ablation reveals whether any useful signal comes from direction or merely local co-occurrence.

## Identical neighbors in the two arms

Select the query's most recent four distinct observed products. Use reciprocal query recency weights 1/j, not Round08's rarity weights. Compute cosine similarity against the binary unique-product vector of each candidate historical session. Restrict eligibility to histories containing the query's most recent product, and keep at most 64 using descending similarity, descending final retained time and ascending session ID. Both arms use exactly the same selected histories, similarity values and normalization denominators. Neither reads Round08's gains or chooses its winning representation.

For each neighbor, locate the final retained occurrence of the most recent query product. This is the pivot. Original event indices define displacement, never retained row positions. A missing original event cannot be compressed into adjacency. The primary allows event-index displacement +1 through +5, with elapsed time no more than 1,800 seconds. The unsigned ablation allows absolute displacement 1 through 5 and absolute elapsed time no more than 1,800 seconds; it deliberately admits preceding events while changing nothing else. Tied timestamps are allowed when the original index establishes order. The historical source itself is strictly before the certified cutoff, so even its post-pivot events remain historical.

Both arms exclude candidate products equal to the most recent query anchor. The original 134-feature control still supplies revisit signals. No purchase is assumed terminal, no permanent user identity is constructed, and no event after the historical cutoff is permitted. A session containing no eligible continuation remains in the denominator and contributes no candidate evidence, so support scarcity is visible rather than silently renormalized away.

## Exactly eight statistics for each of three actions

Let N be the identical selected neighbor set, w_s its similarity, r_s its one-based rank, W=sum(w_s), and R=sum(1/r_s). I_s(c,t) indicates that at least one occurrence of the candidate/action passes the arm's window. For supporting neighbors, g_s is the minimum absolute original event gap among eligible occurrences and d_s the minimum absolute elapsed seconds among them. These two minima are computed separately for their corresponding features; the code does not claim they identify the same event. k_s counts distinct query anchors in the neighbor.

| Statistic | Definition | Interpretation |
|---|---|---|
| log_session_support | log(1 + sum I_s) | Distinct sessions supporting a local continuation |
| weighted_vote_share | sum(w_s I_s) / W | Similarity mass on eligible local evidence |
| max_similarity | max supporting w_s, otherwise 0 | Strength of the best supporting neighbor |
| inverse_event_gap_vote_share | sum(w_s I_s/g_s) / W | Prefer evidence close in original event order |
| time_decay_vote_share | sum(w_s I_s exp(-d_s/300)) / W | Prefer evidence close in elapsed time |
| immediate_event_vote_share | sum(w_s I_s 1[g_s=1]) / W | Strict original-event adjacency component |
| multi_anchor_vote_share | sum(w_s I_s 1[k_s>=2]) / W | Continuations supported by broader query context |
| reciprocal_neighbor_rank_share | sum(I_s/r_s) / R | Continuations supported by high-ranked neighbors |

Repeated events of the same item/action in one source session cannot multiply support. Closest eligible evidence determines gap and time features; other aggregations are binary per session. Empty denominators and absent candidate/action evidence produce zero. Similarity is descriptive, and the resulting shares are not calibrated probabilities. The event window, time window and decay constant are frozen before any supervised comparison.

The same eight statistics for clicks, carts and orders produce 24 primary columns and 24 ablation columns. Full formula definitions and names appear in `feature_catalog.json`. The worked notebook constructs explicit before/after examples, including a nearby preceding event excluded by the primary and an original-index gap that does not qualify as immediate adjacency. Its plots contain only synthetic illustrations, never claimed experiment scores.

## Interpreting a result

Forward-minus-control is the advancement contrast; forward-minus-unsigned tests the specific value of direction. A primary loss cannot be rescued by beating an even worse unsigned ablation. If both arms lose, examine the fraction of neighbors with qualifying continuations, cart/order exposure and correlations with existing graph evidence before proposing a genuinely different feature family. Do not repeatedly widen the gap or decay constant using these same validation labels. A positive result on one fold alone does not satisfy the stability gate.

This round is independent of the first score result, but not an independent holdout study: both use the repeatedly inspected fitting cohort. A future temporal/cohort check must freeze whichever representation is proposed and separately justify historical availability. No part of this round expands the candidate list or claims that source truncation, algorithm capacity and data scale have been ruled out as remaining limitations.

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
