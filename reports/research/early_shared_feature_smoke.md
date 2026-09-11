# Aug-16 frozen shared-feature reconstruction smoke

## Result

The bounded early-window reconstruction **passed** on the merged source commit `91d3dcd1324fdaf06252c5afb128795fab22203e`. The independent post-merge CI run `34561363574` also passed before AWS execution.

This stage rebuilt both wide historical graph families at the preregistered **2022-08-16 22:00 UTC** cutoff, reconstructed the frozen **102 baseline + 32 shared = 134 features** on the exact frozen 256 fitting and 256 selection sessions, checked candidate/target identity against the certified early caches, and replayed every matrix deterministically. It did **not** fit a ranker or use selection labels to alter formulas.

| Check | Measured result |
|---|---:|
| Symmetric graph | 64/64 parts, 186.804 s |
| Forward graph | 64/64 parts, 153.352 s |
| Historical max timestamp | 1660687199973 (< cutoff 1660687200000) |
| Fit cohort | 256 sessions / 102,400 complete candidate rows |
| Selection cohort | 256 sessions / 102,400 complete candidate rows |
| Feature schema | 102 baseline + 32 shared = 134 |
| Deterministic full replay | Passed |
| Peak RSS | 16,091.19 MiB |
| Smoke runtime | 396.214 s |
| Model fits | 0 |

The fit matrix SHA-256 is `957017138f27a307ce41faed47ba3a3068ceec779e95a1218f74cb5892f22ed1`; the selection matrix SHA-256 is `c708518d9325d9aab647788dc71e7e9f747f2218e7ba180195113dfe1f63c6f5`. Durable graph partitions, matrices, fitting-only diagnostics, the complete per-session replay hashes, worker log, and result receipt remain under `s3://otto-recsys-560403859723-us-west-2/ranking/research/shared-feature-confirmation/early-smoke-91d3dcd1/`.

## Interpretation

This resolves the earlier-cutoff construction blocker. It establishes that the frozen 134-column representation is reproducible on the declared Aug-16 window without borrowing the later Aug-20 graphs. It does **not** establish ranking lift, statistical significance, or promotion. No official Recall@20 was computed because fitting a model was explicitly outside this smoke gate.

The early window was previously exposed during robustness work, so this remains development temporal replication rather than a claim of an untouched holdout. Candidate identities, targets, model capacity and formulas must remain frozen as scale increases.

## Next bounded gate

Advance to **1,024 complete fitting sessions only**. Reuse the now-certified Aug-16 graphs and candidate policy; do not rebuild them unless their checksums differ. Measure construction time and peak memory, then perform fitting-only finite/support/nonconstant and redundancy diagnostics over the frozen 134 columns. Only after that passes may a bounded supervised fitting-only screen or add/drop family ablation be considered. Selection labels must not be used to redesign formulas.

Feature engineering remains open. The newly implemented 75-feature as-of demand/support family is a separate research hypothesis and must follow its own strict as-of snapshot, fitting-only screening, controlled ablation and second-window stability protocol before retention.
