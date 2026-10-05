# Validation integrity and comparator reconciliation · October 5, 2026

This update documents a validation-integrity correction discovered during deployment-parity work. It is intentionally employer-facing and aggregate: the public repository records the scientific contract, the correction, and the engineering controls without publishing row-level predictions, cohort/session identifiers, private runners, checkpoints, embeddings, credentials, or exact cloud orchestration.

## Verified external state

The strongest verified post-competition submission remains **0.57586 private / 0.57601 public** on submission **56542128**. That system preserves the established click/cart routing policy and uses the promoted neural-similarity order ranker.

No newer Kaggle result is claimed here.

## What the newer research added

After the externally verified similarity-order milestone, the project expanded beyond the incumbent with:

- objective-specific source-aware ranking;
- contextual reliability-aware models;
- candidate-conditioned sequence interaction;
- leakage-safe heterogeneous OOF stacking;
- deterministic, resumable inference shards;
- immutable prediction/model identities;
- deployment-parity replay and output-membership checks.

The strongest heterogeneous sequence stack produced **+0.004203 deployment-aligned fitting gain with 5/5 nonnegative combined and cart folds**. That is still valid fitting evidence.

## What the audit found

A subsequent validation path compared the frozen click/cart challenger against a baseline that did not exactly reproduce the deployed objective-specific routing policy.

That distinction matters because the deployed incumbent is not simply candidate order. It uses objective-specific routing and ranking policies that changed through the verified submission lineage.

The issue surfaced during official-inference parity checks, where the reconstructed baseline disagreed with the immutable incumbent on nearly every session in a preserved deployment batch. The project stopped the release rather than treating the mismatch as harmless.

## Scientific consequence

The later Fresh Selection V2 / Final Reserve V2 promotion interpretation is **withdrawn pending comparator reconciliation**.

This does **not** invalidate:

- the externally scored 0.57586 / 0.57601 incumbent;
- the earlier fitting evidence for the heterogeneous stack;
- the completed model-family ablations;
- the v42 similarity-order result;
- the reproducibility infrastructure and preserved artifacts.

It does mean that later click/cart gain, interval and chronology claims cannot be used as evidence of improvement over the deployed incumbent until the comparator is reconstructed and the frozen challenger is re-evaluated against it.

## Current reconciliation protocol

The current bounded audit has four goals:

1. reconstruct the deployed click/cart policy from immutable submission lineage and archived model/evaluation evidence;
2. prove parity against preserved incumbent predictions on a deployment slice;
3. recompute the frozen challenger point comparison against that true comparator without refitting or reopening raw target-item labels;
4. record an append-only release decision and registry correction.

A corrected promotion check still requires a material positive weighted gain, non-regressing clicks and a meaningful cart improvement. Passing point estimates alone is not sufficient; uncertainty and chronological stability must also be defensibly reconciled before promotion.

## Why this matters for the portfolio

The project treats validation as production infrastructure, not a presentation layer. A strong-looking result is not retained because it is convenient: comparator identity, model identity, cohort identity, and deployment parity are first-class contracts.

The correction is therefore part of the research result:

- immutable historical evidence is preserved;
- stale promotion claims are corrected rather than rewritten;
- expensive deployment is blocked until evidence is valid;
- external competition scores remain clearly separated from offline validation;
- public GitHub surfaces publish aggregate evidence and contracts, while sensitive artifacts stay private.

## Current release state

| Component | State |
| --- | --- |
| Verified competition champion | **0.57586 private / 0.57601 public** |
| Heterogeneous click/cart stack | **Research — comparator reconciliation** |
| Official competition input attestation | Verified |
| Incumbent order recommendations | Preserved and certified |
| New challenger full inference | Blocked pending corrected qualification |
| New Kaggle submission | None |

The next public update should publish the corrected comparison outcome, whether positive or negative. If the challenger fails, the project closes that promotion cleanly and moves to a materially different objective/representation rather than tuning against consumed holdouts. If it passes every corrected gate, official inference resumes from preserved checkpoints.
