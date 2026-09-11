# Candidate availability: nested frontier experiment

## Research question

Do wider one-hop and bounded two-hop historical retrieval pools recover targets that the current fixed 400 candidates cannot contain? This is a representation/retrieval experiment, not a final-model transition. The 54-model temporal feature-value study is already complete; do not rerun it or reconstruct its verified graphs.

The [first-place OTTO account](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution) describes approximately 1,200 candidates, varied covisitation, repeated graph expansion and target-conditioned neural candidates. It also describes popularity ranks and candidate/session interaction signals. The [third-place implementation](https://github.com/TheoViel/kaggle_otto_rs) instead used roughly 80 candidates with 744 aggregated similarities. Therefore **candidate count alone is not evidence of quality**: this experiment tests our actual missing-target coverage and cost, not a claim that 1,200 is universally optimal. The historical private winning score is 0.60503; internal candidate ceilings and fitting scores are not comparable with that leaderboard.

## Frozen mechanism and controls

Use the same Aug-16 history, exact first 256 fitting sessions selected by seed 20260911 and exact certified 400-candidate baseline identity/order. Reuse baseline, symmetric and forward graphs, each with time/cart/order channels. Preserve the 400 candidates as a prefix, and append reciprocal-rank-fused candidates at 800 and 1,200 budgets. Two-hop expansion uses at most 64 strongest bridges per graph source, normalized by first-hop evidence. Source weights remain those proposed before the smoke; never optimize them on these target labels.

One-hop evidence is constructed once per query. Each complete candidate family is generated twice to check deterministic replay. Generation accepts observed prefixes and certified graph objects, not labels. Only afterward are full distinct target sets attached. The candidate oracle caps reachable hits at 20 while retaining the complete capped denominator. It is a bound inside a candidate pool, **not achieved top-20 ranking**. Larger nested pools cannot reduce this oracle but can hurt a learned ranker.

## Resource, availability and recovery contract

The CLI enforces a 300-second useful-work cap. An outer AWS driver enforces its own deadline, emits 15-second UTC heartbeats, persists completed session JSON checkpoints to private S3 and stops only `otto-dev` after execution. No new graphs, models, package installation, IAM change or evaluation access. Graph/history checksums, cutoff certification, frozen baseline membership and target parity must pass first. A 250,000-item per-query discovery cap stops rather than silently changing the experiment.

Each session is checkpointed with an input identity, full candidate lists, digest, denominators and reachable hits. A restart verifies candidate digests and input contracts and recomputes metrics from preserved candidates; it does not regenerate completed sessions. Wrong contracts are rejected. Do not rerun an unchanged failure.

## Decision and next research

A best expanded weighted-oracle gain of at least +0.005 permits an unchanged 1,024-fitting-session replication of all arms, not automatic source selection or promotion. Report all objectives, paired descriptive intervals, candidate counts, latency and memory. The initial cohort has just 27 order-denominator units, so it cannot establish competitive generalization.

After fitting-only replication, test source/provenance and candidate-by-session features with matched ranking comparisons and temporal stability. Separately, continue rolling as-of demand/support, action-pair graphs and target-conditioned neural retrieval. The small fixed-cutoff 12-feature relative-demand pilot is not the untested 75-feature rolling family. Failed or unstable arms remain documented. Feature engineering stays open.

The executable/replay notebook is `notebooks/research/candidate_frontier.ipynb`. Its analysis must use the saved AWS report and actual execution receipt; no synthetic score is a competition result.
