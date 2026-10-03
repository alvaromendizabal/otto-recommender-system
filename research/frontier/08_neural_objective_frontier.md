# Neural and objective frontier · October 2, 2026

This employer-facing snapshot records the completed work after the October 1 candidate/model frontier. AWS remains the canonical private execution workspace. The public repository contains aggregate evidence, source attribution, validation contracts and decisions; it intentionally excludes raw competition data, row-level labels/predictions, cohort IDs, fitted checkpoints, full embeddings, optimizer state, private runners, credentials and exact cloud orchestration.

## Verified competition state

The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public**, submission **56542128**. No experiment in this snapshot has produced a newer competition submission.

The primary metric is pooled **Weighted Recall@20 = 0.10 clicks + 0.30 carts + 0.60 orders**, higher is better. Candidate coverage is reported only as an oracle ceiling and is never substituted for achieved ranking quality.

## First-place neural family: materially covered

The first-place public source remains `mrkmakr/OTTO-Multi-Objective-Recommender-System`, pinned at commit `befc46bdfb747a1bcd840bd04af7ef2e84052b9c`.

The project has now independently adapted and evaluated the principal v15/v18/v21/v23/v27/v29/v31/v42 mechanism family under bounded, point-in-time experiments:

- **v42** remains the productive transfer: its learned sequence representation became candidate-to-session similarity evidence and contributed to the promoted 0.57586 / 0.57601 submission.
- **v31** attention/candidate-union and later dense-interaction variants did not qualify under fitting-only gates.
- **v27 + v29** added **+0.005065 candidate-ceiling headroom** beyond MF+v42, recovering **+217 / +56 / +7 click/cart/order targets**, but missed the frozen +10-order retrieval gate. Their stable-pool similarity stack later added only **+4 order hits / +0.000700 weighted contribution**, with 3/5 nonnegative folds.
- **v21 + v23** added **+0.001580** equal-budget candidate ceiling with **+52 / +9 / +5** click/cart/order targets; below its +0.0025/+10-order gate.
- **v15 + v18** regressed in the equal-budget candidate pool by **-0.001138**, with **-19 / -7 / -4** click/cart/order targets.

These results do not establish checkpoint parity or reproduction of the complete winning ensemble. They establish mechanism coverage and controlled transfer evidence.

## Order-side heterogeneous integration: closed under current recipes

The 1,200-candidate heterogeneous pool exposes substantial order oracle headroom: **2,628 candidate-ceiling order hits versus 2,238 achieved baseline hits** on the fitting cohort. Two materially different attempts failed to convert that headroom into stable top-20 ranking gains.

### Flat source-aware fusion

A 199-feature representation added source presence, rank, reciprocal rank, normalized score and agreement features across base, MF-general, MF-intent and the neural retrieval family.

- stable 400-candidate arm: **2,238 order hits**, aggregate tie to baseline, 3/5 nonnegative folds;
- 1,200-candidate arm: **2,198 hits**, **-40 orders / -0.006997 weighted contribution**, 0/5 nonnegative folds.

Decision: **close flat boosted-tree heterogeneous source fusion for orders**.

### QueryFormer-style field-sequence interaction

A clean-room task adaptation tested a nonlinear field control and explicit candidate/source-field ↔ observed-sequence cross-attention.

- field control on 400 candidates: **+8 order hits / +0.001399**, 4/5 nonnegative folds but worst fold below the frozen stability bound;
- field control on 1,200 candidates: **-14 order hits**;
- QueryFormer-style 400: **-10 order hits**;
- QueryFormer-style 1,200: **-44 order hits / -0.007697**.

All ten model fits and five folds completed; a final reporting assertion later failed, so the owner execution is tracked as an engineering failure while the complete fold evidence is still preserved. The **current QueryFormer-order recipe is scientifically closed** rather than retuned on head count, width or learning rate.

## Objective shift: click/cart multisource ranking

The strongest remaining internal gap was click/cart ranking rather than order retrieval. On the 20,000-session fitting cohort, the sealed candidate-aware base rankers recovered **10,151 click hits and 2,616 cart hits**, while the strongest equal-budget heterogeneous candidate pool contained **14,672 click targets and 3,858 cart targets**.

A 589-feature challenger retained the established candidate/session representation and added 62 heterogeneous source-evidence fields. The candidate pool stayed fixed at the stable 400 items.

The winning frozen combination was **clicks_source400 + carts_source400**:

| Metric | Clicks | Carts |
| --- | ---: | ---: |
| Baseline hits | 10,151 | 2,616 |
| Challenger hits | **10,337** | **2,669** |
| Net hits | **+186** | **+53** |
| Official weighted contribution | **+0.000961** | **+0.002574** |

Combined official-metric contribution improved by **+0.003535**. All **5/5** chronological folds were positive; the worst fold remained **+0.001635**. This clears the frozen fitting gate of +0.003 combined gain, nonnegative click hits, +10 cart hits, at least 4/5 nonnegative folds and worst fold >= -0.001.

The 1,200-candidate variants regressed, so the promotion is specifically about **better source-aware ranking on the stable candidate pool**, not broader candidate union.

Decision: **PROMOTE_TO_SELECTION**. Selection labels and reserved-evaluation labels remain unopened in this snapshot.

## Current decision

The next milestone is the independent **20,000-session selection cohort**, using the actual incumbent click/cart policy as comparator and leaving orders unchanged. The fitting winner is frozen before selection.

Selection advances only if the combined click/cart weighted gain is at least **+0.0025**, click hits do not regress, cart hits improve by at least **+10**, the paired session-bootstrap 95% interval has a positive lower bound, and both chronological halves are nonnegative.

If selection passes, reserved temporal evaluation opens next. No new Kaggle inference or submission is justified before those gates pass.

## Engineering lessons

The frontier work also hardened long-run execution:

- CPU pipelines derive process/thread counts from the detected SageMaker instance instead of fixed concurrency.
- Large transient caches use local NVMe while durable results remain on persistent storage.
- ZIP-backed NumPy containers are fully materialized and closed before multiprocessing.
- Validation/scoring starts in a fresh interpreter after threaded native model fitting, preventing fork-after-runtime deadlocks.
- GPU-dependent work uses the actual sealed representation geometry instead of assumed embedding dimensions.
- Every owner run is classified separately as a scientific result, an execution failure, or a nonblocking incident; valid negative experiments remain successful executions.

## Publication boundary

The public review intentionally excludes private PYZ runners, exact cloud paths, raw inputs, row-level labels/predictions, session IDs, full embeddings/checkpoints, optimizer state and credentials. Source commits, aggregate metrics, frozen gates, decisions, and reproducibility contracts are published so the research story is inspectable without exposing private competition artifacts.
