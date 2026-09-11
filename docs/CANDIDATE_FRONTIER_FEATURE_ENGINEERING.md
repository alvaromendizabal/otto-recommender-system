# Winner-gap candidate-frontier feature engineering

The existing ranking path hard-caps `FeatureEngine.candidates()` at 400. Historical project evidence shows candidate coverage is a binding constraint, while the strongest OTTO solutions used much broader and more diverse retrieval frontiers. This milestone tests candidate **availability** before spending ranker compute.

## Leakage-safe hypothesis

All candidate generation is label-blind and fitted before the query cutoff. The exact baseline 400 candidates are preserved as the prefix of every expanded frontier. Additional candidates come from the already-certified symmetric and forward wide graph families plus optional two-hop propagation through their strongest historical bridges. Labels are attached only after generation to measure candidate ceiling.

The two-hop path is intentionally bounded: at most 64 first-hop bridge items per graph source, normalized by historical one-hop evidence. It never consumes query targets, selection sessions, future events or model predictions.

## Bounded arms

The first run uses the already-verified 256 fitting sessions and evaluates exact `baseline_400`, one-hop 800/1200, and two-hop 800/1200 frontiers. No ranker is fitted. A best expanded weighted candidate-ceiling gain of at least +0.005 advances the exact protocol to 1,024 fitting sessions. Otherwise this direction stops and the next retrieval work must add genuinely new sources such as action-pair covisitation or neural next-item candidates rather than tuning fusion weights on labels.

Only after a 1,024-session coverage confirmation may expanded source/provenance and candidate×session aggregation features be materialized for supervised fitting-only screening.

This work does not replace the separate strict as-of demand/support family; both remain open feature-engineering tracks.
