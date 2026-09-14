# OTTO Round 05 — Historical adjacent action-pair relationships

## Why this experiment replaces more demand/timing variations

The supplied Round04 evidence has 64 checksum-listed files, all verified. Its shared 134-feature control scored 0.5692312343124653; rolling demand scored 0.5540236571375481; fixed-snapshot demand scored 0.5528952251115945. Rolling-minus-control is -0.015207577174917164, with a descriptive stratified 95% interval [-0.028588144404762473, -0.002621046966976348]. Rolling lost in both chronological folds and had one extra click hit, eight fewer cart hits and six fewer order hits. Do not rerun or scale these two representations. These are fitting-data diagnostics, not private leaderboard scores. Reused-cohort, adaptive experiment selection limits nominal statistical inference.

## Domain hypothesis and primary sources

A viewed product followed by a cart addition is not the same behavior as two consecutive views. Pooling source and destination action types can obscure this relationship. Test independently counted historical action-pair transitions, while holding candidates, ranker settings and training data constant.

The sixth-place author's account describes behavior-conditioned i2i signals, including click2click/click2cart: https://www.kaggle.com/competitions/otto-recommender-system/writeups/thluo-6th-place-solution-single-model-lb-0-603

The seventh-place account describes action-pair and directional covisitation features: https://www.kaggle.com/competitions/otto-recommender-system/writeups/jack-toshi-k-7th-place-solution

Multi-behavior session recommendation also explicitly models global item transitions conditioned on behavior types: https://arxiv.org/abs/2109.11903

These sources motivate the class of representation. They do not validate this package's specific adjacency rule, smoothing, feature count, source truncation, or expected gain. Existing repository retrieval uses target-type weights for its time/cart/order graphs; this experiment instead preserves a 3-by-3 source/destination action table. It is not the first use of action types in the project.

## Exactly 27 primary features, plus a 9-column information ablation

For each query, select the most recently observed product for each of clicks, carts and orders. Missing action types have no anchor. For each anchor action s and candidate action t, count distinct historical sessions containing the directed adjacent product/action pair. With forward support f, correctly reversed support r (items AND action types reversed), full outgoing typed support D_s, and fixed smoothing 20, use:

1. log(1 + f).
2. f / (D_s + 20).
3. (f - r) / (f + r + 20).

Three source actions times three destination actions times three statistics give 27 columns. The complete feature catalog is feature_catalog.json. These are descriptive session-unique support intensities and direction contrasts, not causal effects or calibrated purchase probabilities. Missing anchors/no evidence yield zero. A candidate equal to its anchor is excluded; original revisit features remain in the 134-column control.

For the ablation, discard historical action labels before counting DISTINCT (session, source item, target item). Do not sum typed supports: one historical session can contribute several distinct types for the same item pair. Use the same query anchors and three statistics, yielding nine columns. This is an information-removal ablation with fewer columns, not a width-matched placebo. Compare each challenger to the control; a positive typed-minus-collapsed contrast cannot rescue a losing primary.

## Source availability and exact edge construction

Reuse the existing baseline retrieval/history_tail.parquet, verified against the prior feature contract and baseline retrieval manifest. It is a retained historical tail, NOT the entire original session or raw dataset. Both endpoints must be strictly before 1660687200000 (2022-08-16 22:00 UTC). Join by session and original event_index+1, not file row order. Use different products, nonnegative time separation <=30 minutes, and original valid action types. Tied timestamps are allowed when event indices are consecutive. Missing original indices are not compressed into adjacency.

Source/candidate filtering occurs only after the original adjacency relation is determined. All 4096 current and 1024 earlier study session IDs are excluded before counting. Requests use observed anchors and complete saved 400-candidate pools, never labels. Reverse requested pairs are included. Outgoing denominators count all historical destination products for a requested source, including destinations outside all candidate pools. Item IDs must fit the explicitly checked nonnegative 31-bit catalogue.

A failure of this small family does not exhaust action-pair relationships: longer nonadjacent paths, other time windows, source-tail choices, target-conditioned embeddings and richer pooling remain distinct hypotheses. Do not interpret the historical tail as complete-session coverage.

## Frozen supervised comparison

Re-use the exact Round02 4096 fitting sessions and 400 candidates per session; control values and models are immutable. Exclude the failed Round02 timing, Round03 recent-affinity and Round04 demand columns. New arms are control_shared (134), action_pairs (161), and collapsed_action_ablation (143).

Before fitting any challenger, replay all six saved controls with their exact feature order and confirm their per-session validation hits. Then fit at most twelve new native models: two arms x two chronological folds x three objectives. Reuse the same seed, LambdaRank settings, 150 rounds, six-hour query embargo, censored training targets BEFORE negative sampling, and complete validation denominators. Require native reload prediction parity and immutable model receipts. No algorithm or ensemble search.

This is another exploratory screen on a reused fitting cohort, not independent temporal confirmation. The separate selection/evaluation roles and competition tests are not accessed. Bootstrap intervals are descriptive and do not adjust for all prior hypothesis selection. The historic 0.60503 private target is not directly comparable to internal folds or this pooled fitting score.

## Boundaries and checkpoints

Create a NEW transition-count index in 16 source-ID partitions. Reuse original historical graphs unchanged; do not rebuild their 64-part symmetric/forward artifacts, rerun raw JSON preprocessing or parse the 11.31 GB train file. Indexing can scan the existing Parquet tail for each source partition, so this is new bounded work, not a zero-cost lookup.

Useful-work limits: index 300s, features 300s, screen 240s; outer process limits 320s, 320s, 260s. Tests/report have 60s caps. Require 10 GiB free disk at each data-stage entry, a 26 GiB worker-memory stop, DuckDB memory budget 6GB and spill cap 3GB. Bound requested directed pairs at 10 million; stop instead of silently trimming. Fifteen-second UTC heartbeats and committed partition/chunk/model counters expose progress. These are safety limits, not runtime predictions.

Each transition partition and 64-session feature chunk receives a checksum and immutable receipt. The code preserves conflicting/orphan evidence and stops. Completed native models are reused only when input and schema contracts match. A planned checkpoint pause is not automatically retried; return the ZIP before another invocation. A timeout or failure is not permission for repeated unchanged execution. Never run notebook and terminal paths concurrently.

Local tests exercised identical relational extraction SQL using SQLite and compared it with an independent Python oracle, plus partition/recovery logic via an explicitly labeled adapter. The local environment lacked DuckDB/Parquet dependencies. Therefore the mandatory backend_smoke.py uses your existing AWS DuckDB and a tiny synthetic Parquet fixture BEFORE any real source indexing. This smoke is also invoked by the index stage when terminal users skip the full test stage. No package installation is performed. A backend mismatch stops, rather than switching databases for the real experiment.

## Evidence, gate, and next decision

Report all three matched scores, each chronological fold, action-specific recall, exact hits explaining differences, descriptive stratified paired intervals, training-only feature support, and redundancy. Require primary pooled gain >=0.003, nonnegative differences on both folds and no pooled order decline to propose separate different-cohort/different-history confirmation. No automatic retention, submission, or leaderboard claim follows. The secondary typed-minus-collapsed contrast is not the advancement gate.

If the primary fails, freeze the negative result. Inspect exposure/positive-target support before spending on a larger typed relation; do not tune the 30-minute gap or smoothing repeatedly against these labels. Remaining directions include candidate availability, justified nonadjacent action pairs, calibrated uses of existing latent/neural retrieval and genuinely different historical windows. Model/data-scale limitations have not been ruled out; this comparison holds them fixed to isolate feature value.

## Storage and publication

New outputs are under ~/otto_feature_round05/outputs. Prior folders, raw source, original graph artifacts, existing environment and Git checkout are read-only. The toolkit makes no AWS API writes, cloud launches, IAM changes, Git pushes, or S3 uploads. This package is NOT a GitHub commit. The result ZIP includes small logs, statistics, notebook bytes and checksums, not full index/matrix/model binaries. Stop the SageMaker APPLICATION after download; shutting down only the notebook kernel or timing out a process does not stop instance billing. Do not delete the persistent space.
