# Contextual, sequence and ensemble validation frontier · October 3, 2026

> **Post-snapshot correction — October 5, 2026:** This document preserves the October 3 research snapshot, but its Fresh Selection V2 / Final Reserve V2 promotion interpretation is no longer current. A later deployment-parity audit found that the click/cart comparator in that validation path did not exactly reproduce the deployed objective-specific routing policy. The fitting and model-family experiments remain useful evidence; the later promotion claim is withdrawn pending comparator reconciliation. The verified competition incumbent remains **0.57586 private / 0.57601 public**. See [11_validation_integrity_reconciliation.md](11_validation_integrity_reconciliation.md).

This snapshot documents the post-selection research program that followed the source-aware click/cart study. It is intentionally employer-facing and aggregate: AWS remains the canonical private execution workspace, while GitHub publishes model-family decisions, validation design, aggregate metrics, source attribution and reproducibility contracts. Raw events, row-level targets/predictions, cohort IDs, private runners, model binaries, embeddings, credentials and exact cloud orchestration remain private.

## Verified competition lineage

The strongest verified post-competition submission remains **0.57586 private / 0.57601 public**, submission **56542128**. It combines the established click/cart policy with the promoted neural-similarity order model. No newer competition submission is claimed in this snapshot.

The primary metric is pooled **Weighted Recall@20 = 0.10 clicks + 0.30 carts + 0.60 orders**, higher is better.

## Source-aware click/cart selection

A 589-feature source-aware click/cart challenger had qualified fitting with **+0.003535 combined weighted contribution**, **+186 click hits**, **+53 cart hits**, and **5/5 positive chronological folds**.

The independent 20,000-session selection result did not confirm the joint policy:

- clicks improved by **+259 hits**;
- carts declined by **61 hits**;
- combined weighted gain was **-0.001738**;
- paired 95% bootstrap interval was **[-0.003914, +0.000415]**;
- both chronological halves were negative.

Decision: **close the joint source-aware tree recipe**. The asymmetric result—stronger clicks but weaker carts—became the design signal for the next experiments.

## Contextual reliability-aware transport

The next study tested contextual feature transport and gated nonlinear rankers for cart ranking while preserving the stronger click component.

Four model families were evaluated over five OOF folds, for **20 completed GPU fits**. The strongest contextual-gating arm still regressed versus the deployment-aligned comparator:

- combined gain: **-0.002352**;
- cart hit change versus the deployment comparator: **-71**;
- cart hit change versus the source-aware fitting system: **-28**;
- combined nonnegative folds: **0/5**.

Decision: **close the contextual-transport recipe** rather than tune it on a failed validation direction.

## Candidate-conditioned sequence interaction

The next capability modeled the observed behavior sequence directly with candidate-conditioned interaction rather than only transporting aggregate source features.

The study evaluated residual-MLP, DIN-style cross-attention, behavior-aware cross-attention and sequence self-attention families over five OOF folds, for **20 completed fits**.

The strongest policy was a DIN-style long-session arm:

- combined deployment-aligned gain: **+0.001678**;
- cart hit gain versus deployment comparator: **+12**;
- cart improvement versus source-aware ranking: **+55**;
- combined nonnegative folds: **5/5**;
- cart nonnegative folds: **4/5**;
- worst combined fold: **+0.000597**.

The result was stable and directionally useful but remained below its frozen promotion threshold, so the standalone sequence recipe was not promoted. Its OOF predictions were preserved for ensemble analysis.

## Heterogeneous OOF stacking

The preserved OOF predictions created a genuinely heterogeneous ensemble set spanning:

- source-aware gradient boosting;
- residual neural ranking;
- candidate-conditioned sequence attention;
- behavior-aware sequence ranking;
- contextual feature-transport models.

Fixed reciprocal-rank and standardized-score blends were compared with leakage-safe learned meta-rankers. Learned stackers trained on four OOF folds and scored only the held-out fifth fold.

The selected fitting policy was **within-session standardized-score fusion across the complementary sequence family**.

Aggregate fitting evidence:

- combined deployment-aligned gain: **+0.004203**;
- cart hit gain: **+64**;
- combined nonnegative folds: **5/5**;
- cart nonnegative folds: **5/5**;
- worst combined fold: **+0.002071**;
- worst cart fold: **+0.001474**.

Decision: **promote to a new independent selection stage**.

## Fresh Selection V2

The prior 20,000-session selection cohort had already been consulted, so it was not reused. A new chronology-based selection cohort was frozen from the previously unopened evaluation role using observed-prefix information only.

The promoted stack passed Fresh Selection V2:

- combined weighted click/cart gain: **+0.003194**;
- click hit gain: **+136**;
- cart hit gain: **+47**;
- paired 95% bootstrap interval: **[+0.001865, +0.004554]**;
- first chronological half: **+0.004021**;
- second chronological half: **+0.002373**.

Decision: **promote to Final Reserve V2**.

## Final Reserve V2 — active

The final reserve contains **412,492 sessions**. Predictions are being generated in deterministic chronological shards while reserve labels remain sealed.

Current verified preparation state:

- **200,000 / 412,492 sessions** have frozen predictions;
- **4 / 9 shards** are complete;
- **212,492 sessions** remain;
- reserve labels remain **unopened**.

The final gate was frozen before reserve labels could be read. It requires positive overall transfer, non-regressing clicks, a material cart-hit gain, a positive paired bootstrap lower bound, stable chronological halves, and stable quarter-level behavior.

No final-reserve metric or new competition score is claimed while prediction preparation remains incomplete.

## Engineering evidence

The frontier work also hardened the research platform:

- deterministic, resumable prediction shards;
- immutable per-shard hashes;
- one-upload return bundles;
- GPU/CPU interpreter capability checks;
- process isolation around CUDA and native threaded runtimes;
- resumable feature caches and model checkpoints;
- explicit separation of execution failure from scientific rejection;
- regression tests for previously encountered serialization, subprocess, multiprocessing, inference-mode and event-indexing failures.

The final-reserve workflow reuses completed prediction shards rather than recomputing them. The first four chronological shards have completed successfully on the AWS-canonical runner.

## Publication boundary

This public snapshot includes aggregate evidence, validation roles, model-family descriptions, promotion decisions and engineering/reproducibility practices. It intentionally excludes raw competition data, row-level labels and predictions, session identifiers, private runners, full checkpoints, embeddings, optimizer state, credentials and exact private orchestration.

## Current decision

This dated snapshot is preserved for provenance. The current decision has changed: **do not use the later Fresh Selection V2 / Final Reserve V2 result as promotion evidence until the deployed comparator is reconstructed and the frozen challenger comparison is recomputed.** The externally verified similarity-order submission remains the accepted champion. See [11_validation_integrity_reconciliation.md](11_validation_integrity_reconciliation.md).
