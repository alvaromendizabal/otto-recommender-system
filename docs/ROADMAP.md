# Project scope and completion

**All nine frozen temporal validation cells are complete and independently audited.** Full inference on the official competition prefixes is also complete, and the downloaded gzip passed the exact session/action ledger, every top-20 list and full-file checksum validation.

| Milestone | Current verified result |
|---|---|
| Event processing and temporal protocol | 216.7M training events; 217 partitions; chronological role contracts |
| Retrieval and neural/ANN benchmarks | Executed notebooks 02–06, with their original scope preserved |
| Feature research | 1,482 explicit formulas; fitting-only screening; eight ablations per cell |
| Robustness | **9/9 audited** across three windows and three model seeds |
| Matched feature gain | **+1.791 to +2.180 percentage points** over compact controls |
| Original reference model | **0.584392** weighted Recall@20; remains frozen for submission |
| Highest observed offline score | **0.590759**, middle seed 20260908; a different evaluation cohort |
| Official-prefix full prediction | **1 verified file**, 1,671,803 sessions and 5,015,409 rows |
| Kaggle delivery | File uploaded; final Submit action blocked by automatic approval review |
| Corrected Kaggle score | **Pending**; no value inferred from the offline study |
| Employer-facing publication | Canonical notebooks, model card, complete comparison and Plotly report |
| Earlier 50-file collection | **Not executed**; one full file is verified, not 50 |

## Remaining closeout

The remaining immediate action is to submit the prepared file to Kaggle, verify its accepted status and score, and record that outcome in the canonical receipt and narrative. The first 0.93583 private score is invalidated because its input contained future events. It is retained as incident evidence, not as a performance claim.

The broader previously requested 50-file collection remains unfulfilled and must not be described as complete. It would require distinct model or ensemble recipes chosen on fitting/selection evidence, cache reuse, and content deduplication. The present delivery prioritizes the single corrected submission requested for closeout; it does not start additional training or manufacture renamed copies.

## Runtime and monitoring

The managed remaining queue ran from **20:36 to 22:26 UTC on September 9**. Its final two training runs executed concurrently, and both audits succeeded. Corrected full inference completed at **21:43 UTC**. No validation or inference jobs remain in progress for this batch.

```bash
cd "$HOME/otto-recommender-system" &&
uv run --frozen --extra cloud python scripts/robustness_status.py
```

This read-only command reports actual SageMaker job names. An offline saved view is available with `--snapshot reports/robustness/batch/execution.json`. The queue used up to three simultaneous steps under the authorized scheduling amendment while preserving the frozen data, feature and evaluation protocol.

[Complete results](ROBUSTNESS.md) · [Submission and source audit](INFERENCE.md) · [Model card](MODEL_CARD.md)

## Further research

Certified neural retrieval within the same cutoffs, new sequence models, additional rankers, blending studies and online experiments are extensions. Current evidence supports repeatable offline feature gains on this historical dataset; it does not claim state-of-the-art performance or online lift.
