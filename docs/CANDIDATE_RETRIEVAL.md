# Candidate retrieval contribution analysis

This stage measures **candidate-set ceiling and marginal target recovery** before a
large ranker-training table is materialized.

The evaluator combines five retrieval sources on the frozen local validation
protocol:

- session revisit / recency;
- time-decayed co-visitation;
- objective-conditioned type co-visitation;
- buy-to-buy co-visitation for carts and orders;
- target-conditioned Item2Vec queries served by FAISS HNSW.

The analysis is intentionally **set based**, not a final ranking experiment.
Equal-weight RRF was already shown to damage shallow ranking when weak or broad
sources were added. Here, each hidden label is classified as recovered by
co-visitation, Item2Vec, both, or neither. The report also records source-exclusive
hits and average unique candidate counts.

## Why this comes before candidate materialization

The validation population has 515,702 sessions. Materializing hundreds of
candidates for all three objectives before measuring marginal value would create
a large artifact without knowing whether Item2Vec deserves a meaningful quota.
This stage computes the answer bucket by bucket with bounded DuckDB memory and
writes only a compact resumable state plus final metrics.

## Operational contract

- 32 deterministic validation buckets;
- one bucket processed at a time;
- DuckDB external-memory execution with a configured memory ceiling;
- Item2Vec ANN retrieval limited by `ann_k`;
- atomic `state.json` checkpoint after each completed bucket;
- input fingerprint over validation, co-visitation, Item2Vec, FAISS, and config;
- UTC structured logging and heartbeat telemetry;
- exact restart semantics: rerunning the same command skips completed buckets;
- no final candidate quota is selected until the marginal-recall report is read.

## Primary decision metric

The most important value is `weighted_item2vec_marginal_recall`: the OTTO-weighted
fraction of hidden labels that Item2Vec recovers **and the complete co-visitation
family misses**. This determines whether Item2Vec belongs in the downstream
candidate union and how much candidate budget it should receive.
