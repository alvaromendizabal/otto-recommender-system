# 1,024-session fitting-only shared-feature scale gate

This stage is the direct successor to the passed Aug-16 256+256 reconstruction smoke. It answers a narrower engineering question before any new ranker fit: **can the exact frozen 134-column representation be reproduced over 1,024 complete fitting sessions with acceptable support, redundancy, runtime and memory?**

## Frozen scope

- Fitting role only. Selection and evaluation are forbidden.
- First 1,024 sessions under the already declared seed `20260911`; the first 256 must exactly equal the verified smoke membership.
- Exact 400-candidate policy and the certified fitting-negative cache are used only for candidate/target parity. Diagnostics use all 400 candidate rows before fitting-negative subsampling.
- Exact 102 baseline + 32 shared features. No formula edits.
- Reuse the certified Aug-16 symmetric and forward graphs. A different graph input ID, cutoff, timestamp bound, label-use flag or partition count is a hard stop.
- No model fitting, no Recall@20, no promotion and no feature deletion.

## Diagnostics

The runner accumulates exact first/second moments and cross-products session by session, avoiding retention of the full matrix in memory. It reports finite/nonconstant behavior, row and session support, exact Pearson correlation of each added feature with the frozen baseline, exact redundancy among the 32 additions, output hashes, runtime and peak RSS. Correlation flags are **diagnostic only** and cannot remove a feature.

Four complete 256-session Parquet partitions are checkpointed. A second complete pass must reproduce every per-session candidate/target/feature hash.

## Decision gate

A passing engineering gate permits only the next research step: preregister a bounded **fitting-only** supervised add/drop comparison using fixed candidates, negatives, model capacity and seed(s). Selection remains frozen until the fitting-only hypothesis is fixed. The separate 75-feature as-of demand/support family continues under its own strict historical-snapshot and temporal-stability protocol.
