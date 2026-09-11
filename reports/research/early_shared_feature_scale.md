# Aug-16 1,024-session fitting-only shared-feature scale gate

## Result

The preregistered fitting-only scale gate **passed** on merged commit `29439ec2c3e18e524762129323a1a1cdb2189fcb`. The independent post-merge CI run `34564900791` passed before AWS execution.

The run reused the already-certified Aug-16 symmetric and forward graphs, reconstructed the exact frozen **102 baseline + 32 shared = 134 features** for the first **1,024 fitting sessions** under seed `20260911`, verified the certified fitting-cache candidate/target lineage, checkpointed four complete 256-session parts, and replayed every session deterministically. **Selection/evaluation data were not accessed and zero rankers were fitted.**

| Check | Measured result |
|---|---:|
| Fitting sessions | 1,024 |
| Complete candidate rows | 409,600 |
| Feature schema | 102 + 32 = 134 |
| Pooled denominators (click/cart/order) | 987 / 340 / 168 |
| Feature construction | 29.071 s |
| Full deterministic replay | 22.044 s |
| Total scale-script time | 98.437 s |
| Peak RSS | 8,029.89 MiB |
| Added constant features | 0 / 32 |
| Added features near-duplicate with baseline at |r| >= 0.995 | 1 / 32 |
| Near-duplicate added-feature pairs at |r| >= 0.995 | 0 |
| Minimum added-feature session support | 6.055% |
| Model fits | 0 |

The one baseline-near-duplicate flag is `domain_funnel_full_count_share` with maximum absolute baseline correlation `0.9995758536`. The lowest session-support addition is `domain_funnel_transition_1_0_mean_gap_log_hours`, nonzero in `6.0546875%` of the 1,024 sessions. These are **diagnostic flags, not retention decisions**. No feature formula or shortlist was changed after seeing them.

Durable source artifacts remain under `s3://otto-recsys-560403859723-us-west-2/ranking/research/shared-feature-confirmation/early-scale-29439ec2/`. The four 102,400-row parts have SHA-256 values `b0a8c549...`, `35ce5a41...`, `b01db42d...`, and `a001ca2e...`; the 1,024-row denominator ledger is `2bfe3218...`; the full diagnostics receipt is `9712580d...`.

## Interpretation

This establishes that the exact 134-column representation scales reproducibly on a larger **fitting-only** earlier-window cohort without rebuilding or borrowing later-cutoff graphs. Memory dropped from roughly 15.7 GiB in the earlier combined smoke to about 7.84 GiB here because diagnostics stream moments rather than retaining the entire full matrix in memory.

It does **not** establish ranking lift or justify removing the high-correlation feature. A supervised fitting-only comparison is required before using these diagnostics to motivate any family change. Selection remains frozen.

## Next bounded gate

Preregister a baseline-vs-shared fitting-only supervised comparison on these exact four parts. Use deterministic whole-session folds, the existing fitting-negative sampling policy, fixed candidate rows, fixed model capacity and fixed seeds. Report objective-level and pooled out-of-fold Recall@20 plus fold stability. Do not touch selection unless the fitting-only hypothesis is frozen after this stage.

Feature engineering remains open. In parallel, the 75-feature strict as-of demand/support family still requires real-data snapshot construction, fitting-only screening, add/drop ablation and a second temporal window before any retention or promotion.
