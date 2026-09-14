# OTTO / Round 07 — Historical-session-neighbor evidence

## Decision from the completed experiment

Round 06: saved 134-feature control 0.5692312343124653; repeat context 0.5584510290237252; global context 0.5616706362937891. Repeat-context change −0.010780205288740019; global-context change −0.007560598018676146. Both arms lost on both chronological folds. Their descriptive paired 95% intervals include zero. Neither configuration earns retention; these results do not establish universal ineffectiveness of repeat-aware recommendation.

All 158 checksum-listed files were checked. Three pooled scores, six fold scores and all three 2,000-resample paired intervals were independently reproduced. Controls, session IDs, denominators and fold labels equal the earlier Round 05 records. Native models and large candidate/feature matrices were not included, so they were not replayed locally. The candidate-oracle arithmetic was checked, but its underlying candidate membership was not independently recalculated.

Saved candidate oracle: 0.714910231512901 versus achieved control 0.5692312343124653. The arithmetic within-pool ranking opportunity is 0.14567899720043576 and the missing-candidate gap to an ideal score of one is 0.285089768487099. This is not a causal diagnosis or a claim that either gap is attainable. Do not compare this fitting-data oracle with the historical private leaderboard target of 0.60503.

The return contains a complete generated HTML report and nine plot specifications. Its `03_saved_results.ipynb` was not executed. The supplied `00_round06_review.ipynb` is a fresh executed, self-contained eight-chart review, so you do not need to rerun the old experiment.

## Why this is not another recycled feature transformation

Earlier work tested timing, recent affinity, rolling/static demand, adjacent action-pair transitions and repeat-conditioned transformations of cached graph scores. Round 07 adds a new historical-session evidence source: **what items/actions occur in individual past sessions that resemble the current observed shopping context?** Similarity depends jointly on up to four observed products and the historical session's item set. It is not just another percentile, popularity share or adjacent-pair count.

Session-based nearest-neighbor recommendation has an established empirical research basis. Ludewig and Jannach compare session-based methods and find that nearest-neighbor approaches can be competitive with more complex methods. That motivates a candidate evidence source; it does not validate this implementation's settings or imply an OTTO gain.

Primary source: https://arxiv.org/abs/1803.09587

For comparison, the third-place OTTO team's published implementation emphasizes varied item-item signals, session-position/time/type weighting and aggregation. We should investigate complementary information rather than merely increase feature count. The present session-neighbor pilot is not claimed to reproduce their winning implementation.

Primary implementation: https://github.com/TheoViel/kaggle_otto_rs

Official dataset/evaluation repository: https://github.com/otto-de/recsys-dataset

## This milestone ends BEFORE another ranker fit

This is a **256-query training-only feature/source feasibility study**. Its deliverables are committed feature matrices, a last-anchor information-removal comparison, source proposals, support diagnostics and an eight-chart report. There is no screen/train command, no new LightGBM fit and no automatic submission.

Why this gate: multiple recent additions regressed. Before buying another twelve-model comparison, establish that a genuinely new source has enough historical and target support, whether it adds missing products, and how redundant its features are. A promising training-only audit is NOT validation and is not permission to promote features.

## Frozen pilot selection and information boundaries

Reuse the exact Round 02 fitting cohort, its verified inputs, clean Git main and original 400 candidates per query. From the first chronological fold's training IDs only, choose 256 by the fixed ID hash seed 20260912. No target, feature value, query length or measured difficulty is used for selection. The first-fold query embargo and training IDs are checked against the saved fold ledger.

Read complete observed query prefixes and validate their original event indices and timestamps against the saved query ledger. Choose the **last four distinct product IDs**, newest first. This is a fixed, explicitly limited context—not a claim to preserve every user intent in long sessions.

The historical source is the existing `retrieval/history_tail.parquet`, verified against the retained retrieval manifest and Round 05 source certificate. Its 86,735,324 certified rows end strictly before **1660687200000 (2022-08-16 22:00 UTC)**. These are retained historical tails, not necessarily complete original sessions. Do not parse the raw training JSON or rebuild the old graphs. Exclude all 5,120 recorded study sessions before source selection.

Two historical Parquet data scans are scheduled: one for anchor-to-session postings, one for selected-session events. File checksum passes, query-prefix reads and cached-feature reads are additional I/O; this is not a zero-scan lookup.

All pilot feature/proposal chunks are committed before targets are read. The final audit validates complete original target counts and then excludes targets at or after the earliest validation cutoff. Denominators are `min(number of remaining distinct targets, 20)` for each objective. Never restrict denominators to products already in the 400 candidates. Selection/evaluation roles and competition test data are not queried by this runner.

## Bounded historical candidate sessions

For each requested anchor, retrieve at most the **64 most recent distinct historical sessions containing it**. Recency is the timestamp of its latest matching anchor occurrence; ties resolve by historical session ID ascending. Count the full eligible support before the cap and report anchors whose support exceeds it. Repeated views do not create duplicate postings.

The shared source pool for a query is the union of these per-anchor postings. With four anchors, each query sees at most 256 source sessions. Across the pilot there are at most 65,536 distinct selected source sessions. Retrieve all their events that remain in the certified historical tail. Stop rather than silently truncate if this exceeds 2,000,000 retained events. Validate original event ordering, action IDs, strict cutoff and study-session exclusion.

The cap is an explicit recency-biased sampling rule. This is not exhaustive all-history nearest-neighbor search, not an IDF-corrected estimator and not an unbiased estimate of all historical session behavior. A weak result can reflect truncation or insufficient multi-product support; do not infer that the whole class is exhausted.

## Primary versus last-anchor comparison

For primary query anchors q_1,...,q_m, newest first, use weights w_i=1/i. Let H be the set of unique retained product IDs in a historical session. The similarity is:

    sim(Q,H) = sum(w_i for q_i in H) / sqrt(sum(w_i^2) * len(H))

Use the **entire retained historical item set** for `len(H)`, not only candidates or matched anchors. Historical session vectors are binary; repeated product events do not inflate similarity. Select the 64 highest-similarity source sessions per query, ties by most recent retained historical timestamp then historical session ID.

For the comparison, keep the same shared source pool but discard the older query anchors, leaving q_1 only. Recalculate similarity and choose up to 64 neighbors under that restricted context. Therefore neighbor membership and similarity change together. This is an information-removal comparison, not a claim to causally separate every component. For a one-anchor query, both arms must be identical.

## Exactly 12 features per arm

For each of clicks, carts and orders, compute four candidate-level features. Let S be the chosen neighbors; v_h their similarity; W=sum(v_h); and I_h(a,t) indicate that candidate a has action t somewhere in h's retained events.

1. **Log session support:** log(1 + sum(I_h(a,t))). Each historical session contributes at most one unit to a given item/action, regardless of repeated events.
2. **Similarity-weighted vote share:** sum(v_h * I_h(a,t)) / W. W includes all selected neighbors, including neighbors that do not contain this candidate. This is a descriptive intensity, not a calibrated purchase probability.
3. **Maximum supporting similarity:** max(v_h over supporting neighbors), or zero.
4. **After-anchor vote share:** sum(v_h * J_h(a,t)) / W, where J requires an item/action event whose original event index is greater than the last occurrence of any matched query anchor in that neighbor. This is nonadjacent follow-up evidence; gaps in original indices are allowed and never compressed into adjacency.

Missing neighbors/support yields zero. Previously observed query products are not excluded: repeat purchases and repeated views remain possible. Historical labels such as carts/orders are historical observed actions, not supervised targets from the current fitting sessions.

Both 12-column matrices are aligned to the unchanged original 400-candidate identities. The first eight queries receive a reversed-candidate and reversed-source-order exact-replay test. The 134 original control features are not rebuilt, replaced or overwritten.

## Source proposals and support diagnostics

For each objective and arm, select up to 100 distinct products from the neighbor vote map by weighted vote share, ties by product ID. This is done before looking at pilot labels. Some proposals will already be in the baseline candidate pool; do not describe them all as additional products.

For each objective separately, preserve all 400 baseline candidates and union these at-most-100 proposals: maximum 500 candidates. Report both baseline and expanded **perfect-ranking candidate Recall@20 oracles**, metric-capped extra hits, uncapped distinct newly recovered targets, and average truly novel products. A union cannot lower the oracle; that mathematical property is not evidence that a reranker will improve. No added-item baseline feature rows or expanded-pool reranker are trained in this milestone.

Also report neighbor counts, multi-anchor overlap support, feature nonzero fractions, support on censored training positives, and feature correlations against the **seven stored comparison signals only** (six graph affinities plus prefix membership). This is not an exhaustive redundancy test against every one of the 134 control columns. No feature is dropped automatically based on these diagnostics.

## Decision after the ZIP is returned

The code records a heuristic triage flag, not a statistical significance test: at least 20 supported cart/order baseline-positive items combined, plus either multi-anchor neighbor evidence in at least 26 pilot queries or at least three new cart/order target items from proposals. These thresholds only distinguish promising support from evidence requiring review. Objectives with fewer than 20 censored target denominator units are explicitly flagged as sparse. Do not tune this pilot repeatedly to meet the flag.

Next decisions, only after inspection:

- Supported and complementary features: generate the frozen family on the full existing 4,096 fitting queries, then a matched unchanged-ranker comparison with saved-control replay, temporal censoring, ablation, exact hits and descriptive uncertainty.
- Meaningful missing-product recovery: test candidate expansion separately from new feature addition; otherwise changed candidate coverage and changed ranking features would be confounded.
- Sparse/truncated evidence: inspect source-cap saturation, query lengths and multi-item overlap before considering one justified change in historical coverage. Do not keep retrying unchanged.
- Successful matched screen: freeze the configuration and confirm on a separate cohort/historical window before promotion.

Later distinct directions remain historical-session retrieval, embedding/source agreement, multi-intent session representations and broader historical windows. Feature engineering remains open. Holding the ranker fixed to isolate feature value does not prove model capacity, sampling, validation or data scale optimal. Ensembles and algorithm searches are not scheduled here.

## Runtime, durability and environment boundaries

Useful-work caps: prepare/postings/histories/features 120 seconds each; audit 90 seconds. Outer process caps: each of those first four stages 140 seconds; audit 110 seconds; tests 60 seconds. These are safety limits, not expected runtimes or future delivery promises. A worker-memory stop at 26 GiB, DuckDB memory cap 6GB, spill cap 3GB and minimum 10 GiB free disk apply. The launcher emits UTC heartbeats every 15 seconds; feature work commits 32-query chunks. First-time source scans commit at stage boundaries, not at arbitrary mid-query points.

A pause/error stops further data stages. There is no automatic retry or override flag. Return the current evidence. Committed checkpoints are preserved and have reusable contracts, but any later resumption requires review rather than unchanged repeated execution. Do not run notebook and terminal stages concurrently.

Portable local tests use the same extraction SQL in SQLite, independent feature oracles, synthetic real feature/audit stages, immutable checkpoints and an eight-chart report replay. The preparation environment lacks DuckDB/Parquet libraries; the full AWS corpus/preflight was not replayed. The mandatory `run_tests.py` additionally executes an actual installed-DuckDB/tiny-Parquet smoke in SageMaker. It is repeated before historical indexing for terminal users. No runtime package installation or fallback database is provided.

All new outputs live under `~/otto_feature_round07/outputs`. The existing repository, raw data, older round folders, environment, model binaries and old graphs remain read-only. The launcher invokes no AWS APIs, Git writes, S3 uploads, cloud launches or Kaggle submissions. This downloadable package is not a GitHub commit or cloud backup. The return ZIP excludes large historical/feature/pilot binaries; preserve the folder. Stop the SageMaker application after downloading; do not delete its persistent space.

AWS stop procedure: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html
