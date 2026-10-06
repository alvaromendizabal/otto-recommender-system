# Validation integrity and comparator correction · October 5, 2026

This case study documents a validation-integrity correction discovered during deployment-parity work. It is intentionally aggregate and employer-facing: the public repository records the scientific contract, correction and release decision without publishing row-level predictions, cohort/session identifiers, private runners, checkpoints, embeddings, credentials, or exact cloud orchestration.

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

The heterogeneous sequence stack produced **+0.004203 fitting gain with 5/5 nonnegative combined and cart folds**. That remains valid fitting evidence.

## What the audit found

A later validation path compared the frozen click/cart challenger against a baseline that did not exactly reproduce the deployed objective-specific routing policy.

That distinction matters because the deployed incumbent is not simply candidate order. It uses objective-specific routing and ranking policies that changed through the verified submission lineage.

The issue surfaced during official-inference parity checks, where the reconstructed baseline disagreed with immutable incumbent recommendations. The project stopped the release rather than treating the mismatch as harmless.

## Corrected reconciliation result

The follow-up audit:

- reproduced incumbent click/cart top-20 membership on **4,096 / 4,096** preserved official prefixes;
- verified **423 / 423** archived statistic parts covering **432,492 sessions**;
- recomputed the frozen challenger against the true comparator without refitting.

The corrected result was:

| Quantity | Result |
| --- | ---: |
| Weighted Recall@20 gain | **+0.00395690** |
| Click-hit gain | **−3,529** |
| Cart-hit gain | **+1,922** |

The frozen qualification contract required non-regressing clicks. The challenger therefore **failed qualification and was rejected**.

Passing the aggregate weighted point estimate did not override a failed objective-level gate.

## Scientific consequence

The earlier Fresh Selection V2 / Final Reserve V2 promotion interpretation is superseded by the corrected comparison.

This does **not** invalidate:

- the externally scored 0.57586 / 0.57601 incumbent;
- the earlier fitting evidence for the heterogeneous stack;
- the completed model-family ablations;
- the v42 similarity-order result;
- the reproducibility infrastructure and preserved artifacts.

It does mean that the newer click/cart stack is **not promoted**, its full inference remains stopped, and later research must begin from a scientifically valid baseline.

## Why this matters for the portfolio

The project treats validation as production infrastructure, not a presentation layer. A strong-looking result is not retained because it is convenient: comparator identity, model identity, cohort identity, and deployment parity are first-class contracts.

The correction demonstrates:

- immutable historical evidence is preserved;
- stale promotion claims are corrected rather than rewritten;
- expensive deployment is blocked when a correctness gate fails;
- external competition scores remain clearly separated from offline validation;
- negative scientific results become inputs to the next architecture decision;
- public GitHub exposes aggregate evidence and tests while sensitive artifacts stay private.

## Current release state

| Component | State |
| --- | --- |
| Verified competition release | **0.57586 private / 0.57601 public** |
| Heterogeneous click/cart challenger | **Rejected after corrected comparator** |
| Official competition input attestation | Verified |
| Incumbent order recommendations | Preserved and certified |
| New challenger full inference | Stopped |
| New Kaggle submission | None |
| Current research focus | Candidate availability under point-in-time controls |

The follow-on research is summarized in [12 · Corrected comparator and candidate-coverage frontier](12_corrected_comparator_candidate_coverage.md).
