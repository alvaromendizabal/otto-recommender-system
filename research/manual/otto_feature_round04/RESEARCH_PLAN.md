# OTTO Round 04 — Cross-session as-of demand

## Measured reason for changing direction

Round 03 completed, but its primary 36-feature recent-affinity representation scored 0.5642245488646602 versus 0.5692312343124653 for the matched 134-column control. Its gain was -0.005006685447805026. The zero-filled ablation scored 0.5570758268709686. Missing-aware exceeded the ablation but neither exceeded the control. All 36 features had another added feature with absolute correlation at least .995 on each sampled training diagnostic (64 sessions per fold). Correlation is a redundancy warning, not a demonstrated cause of regression. Stop further ad hoc variants of these graph-timing summaries on this reused cohort.

## Domain hypothesis

Item popularity may change during the fitting period. Candidate-relative popularity, recent action intensity, and the balance of clicks/carts/orders are different information from the timing of graph-supporting observations. The first-place author's writeup reports ranked item popularity across multiple time windows and action-type ratios. That motivates this feature family, not a claim that these exact 75 formulas will help.

Primary research source: https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution

Official task, metric and permitted source distinction: https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md

Repository definitions reused: `src/otto_recsys/research/asof_demand_features.py` at `2638faa34afa427ed4ac4e92bba04deda57c688e`, blob `7008d1282d6f8f38e1bb8f1b84f21b51e28a6b3b`.

## Exactly 75 columns; no combinatorial filler

For each of clicks/carts/orders and each of 1, 6, 24, 72, 168 hours, compute log(1+count), tie-aware percentile within the full 400-candidate pool, and share of that pool's demand: 45 columns. Three short/long window pairs per action contribute log-rate differences and percentile changes: 18 columns. Action-specific historical seen flags and log ages add six columns. Support-smoothed action proportions at 24 and 168 hours add six columns.

The action proportions are descriptive event intensities, NOT conversion probabilities, purchase propensities, causal effects, or unique-user rates. Repeated interactions count as events. Global priors use all permitted catalogue items, not just the union of candidate IDs. The new adapter preserves the repository feature definitions but accepts full-catalogue totals explicitly, avoiding accidental candidate-only prior estimation.

## Availability contract and leakage controls

Use only the already verified raw TRAIN file, SHA-256 `06716132f1ab1d500f1515ba2c16187d7b0e98066ca66d33b95f022f8c3a340d`, 11,307,535,945 bytes. Neither the truncated competition test nor the contaminated complete test is opened.

Exclude ALL 4,096 current study sessions and ALL 1,024 earlier study sessions from the demand stream before counting any event. This removes their observed actions as well as their future outcomes from the aggregate source. Their prefixes still supply their frozen candidate pools and control features exactly as before. Exclusions depend on fixed session IDs, not outcomes.

For query time q, rolling cutoff C = floor(q / 3,600,000) * 3,600,000. Count only event times t in [C - window, C). A timestamp exactly at C is excluded. Counts are exact for hourly cutoffs; snapshots can be up to an hour stale. Last occurrence is the latest permitted event strictly before C, including occurrences older than 168 hours. The first collapsed hour bucket preserves older last occurrences without incorrectly counting them inside current windows.

The source assumption is a timestamped stream of other training sessions: an event from a non-study session is available to a query if it occurs before that query's cutoff. Thus a later validation-time query may legitimately use earlier non-study interactions from within that period. It cannot use an earlier or later event from ANY selected study session. No labels.parquet values enter the index. The supervised six-hour embargo and training-target censoring remain unchanged.

This is a **different availability assumption from a single frozen-history feature system**. A future submission must reproduce it using only permitted training data and original truncated test prefixes, with every input cut off correctly. This package does not build that inference path. Operational ingestion delays beyond the hourly snapshot lag are not modeled. A model trained with these signals cannot simply be plugged into old batch inference with arbitrary data. No claim of competition readiness follows from this test.

All raw training data is scanned to build the exact source aggregate, but only records before the largest allowed fitting snapshot enter it. Selection/evaluation future periods and test data are excluded. Full-catalogue totals are accumulated independently of the label-blind candidate-union storage filter.

## Matched representations and ablation

- `control_shared`: existing 134 features, six saved native controls replayed; zero refits.
- `demand_rolling` (PRIMARY): the control plus the 75 features from hourly as-of snapshots.
- `demand_static_ablation`: the control plus the SAME 75 formulas/order, with all snapshots fixed to August 16, 2022 at 22:00 UTC.

Rolling versus static tests the value of updating counts. Each addition versus control measures whether that representation helps at all. A positive secondary contrast does not rescue a failed primary contrast. The failed Round02 timing and Round03 mass columns are excluded. No model/ensemble search, candidate expansion, graph reconstruction or automatic feature selection.

Reuse the exact Round02 4,096 fitting sessions and their two chronological folds. Retain all 400 candidates for validation. Keep the existing LambdaRank capacity, seed, 150 boosting rounds, sampled training rows and label-censoring rule. Exactly 12 new experiment models at most; the test suite separately uses one tiny synthetic native-model fixture.

## Bounded execution and durable units

First build an hourly index in ~64 MiB complete-line raw chunks. Every chunk's offsets, source digest, aggregate shards and global counts get an immutable receipt. A normal pause resumes from the last full session boundary; completed chunks are checked rather than parsed again. Merged item-range shards are checkpointed independently. No cloud service writes, raw-data overwrites, graph rebuilds, installations or IAM changes.

Index work budget is 300 seconds per invocation, 320 seconds outer process limit. A 64 MiB raw chunk and a merged shard are the units; never run a new unit near the deadline. Up to two planned index invocations are the initial manual budget. After two pauses, return the receipt for inspection instead of escalating. A failure/timeout is not permission for an unchanged retry. An interrupted uncommitted unit may be replayed; committed evidence cannot be replaced with different bytes.

After the index is certified, feature budget is 300 seconds (320 outer) and screen budget 240 seconds (260 outer). New features checkpoint every 64 complete sessions; each model has its own source/input/feature-order receipt. The launcher bounds worker memory at 26 GiB, and the index checks free disk reserve. Require 10 GiB free at phase entry; index requests review below 4 GiB. These are safety caps, not runtime or cost guarantees. The application remains billable until the user stops it.

Checkpoints stay in `otto_feature_round04/outputs` on the persistent space. They are NOT automatically copied to S3 or GitHub. Do not delete the space. The return ZIP excludes raw input, index shards, feature matrices and model binaries, but includes their receipts and checksums.

## Evidence and decision

Recompute pooled hits/denominators and stratified paired bootstrap intervals from saved per-session integers. Report each time fold and action separately. Screening and redundancy diagnostics sample TRAINING sessions only. The same development cohort has now been reused, so the nominal bootstrap interval is descriptive; it does not account for all experiment selection or establish independent generalization.

Primary pooled gain >= .003, both time folds nonnegative, and no pooled order loss permit ONLY a separately preregistered different-cohort/different-history-window confirmation. No automatic feature retention or submission. If primary fails, freeze the negative result and move to a genuinely distinct family, such as action-pair retrieval/representation, rather than tune these time windows on the same validation outcomes.

## Remaining high-value investigations

Action-pair covisitation and direction/lag support; richer candidate-source/latent similarity aggregations; appropriate uses of existing neural retrieval; meaningful shopping-episode interactions; and independent chronological confirmation remain open. A raw count of features is not evidence. Algorithm and data-scale limitations have not been eliminated as possible contributors to the leaderboard gap; this matched experiment holds them fixed to measure feature value.

The 0.60503 historical private target is not comparable to one fitting fold. Neither successful code execution nor a fitting score over that number proves a winning submission.
