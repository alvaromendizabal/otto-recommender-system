# Model card · retained recommendation system

## Task and intended use

Given an anonymous observed shopping-session prefix, return separate top-20 lists for clicks, carts and orders. The project is an offline research and batch-inference portfolio, not an online retail service.

## Retained approach

Historical co-visitation and observed-prefix candidates feed candidate/session features and task-specific LightGBM ranking. The controlled reference selected 102 features from 1,482 formulas using fitting-only screening. The later delivered release preserves established click/cart routing and uses a promoted 137-feature similarity orders ranker. Private similarities, routing and final model artifacts are outside the public reproduction scope.

## Measurements

| Record | Value | Scope |
|---|---:|---|
| Compact control | 0.564904 weighted Recall@20 | 432,492 temporal evaluation sessions |
| Selected representation | 0.584392 weighted Recall@20 | The same matched population |
| Paired improvement interval | +1.840 to +2.065 percentage points | Recorded session-bootstrap 95% interval |
| Retained scored release | 0.57586 private / 0.57601 public | Post-competition evaluation, not an official rank |
| Batch coverage | 1,671,803 sessions / 5,015,409 rows | Three objective-specific rows per session |

[Controlled-study manifest](../reports/research/manifest.json) · [Evaluation](../reports/research/evaluation.json) · [Release receipt](../reports/submissions/similarity_stack_20260925.json) · [Scope record](../reports/release_scope.json).

These populations are different. The earlier incorrect full-test input was excluded from accepted performance; [the source-provenance audit](../reports/submissions/data_provenance_audit.json) retains that limitation. This publication does not claim a new model score, online revenue lift or a complete independent authentication of historical receipts.

## Validation and failure boundaries

History, fitting, selection and evaluation have explicit temporal roles. Feature screening uses fitting labels only; reserved evaluation follows a model seal. Retrieval coverage is an upper bound, not achieved ranking. Missing candidates, cold-start products and changes in session behavior remain failure modes. Metrics weight clicks 10%, carts 30% and orders 60%, using the declared capped pooled denominator for each objective.

Source identities, input hashes and complete output coverage protect reuse. These controls catch artifact and population inconsistencies; they do not establish future business performance.

## Reproduction

The [public review](../review/README.md) is dependency-free. The native training profile uses actual project components with invented records. Full reference research needs independently obtained official training data, the locked environment and declared resources. The complete private release is not recoverable from the fixture, and former AWS object locations are no longer a supported download path.
