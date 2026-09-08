# Competition inference and model replay

The canonical entry point is `notebooks/10_competition_inference.ipynb`.
Its default mode applies the actual frozen native models to a compact,
checksum-verified set of real competition candidate features and checks exact
agreement with the corresponding full-run output. It does not require AWS.

The full mode runs the production inference command from the notebook. A managed
delivery job installs a separate, fully hashed Python 3.12 analysis environment,
binds the notebook kernel to that interpreter, and invokes the locked Python
3.13 project runtime for feature generation and model prediction. UTC notebook
heartbeats expose elapsed time and the number of prediction-part receipts.
The notebook execution and final prediction manifest have separate receipts.

## Recorded full execution

The managed run completed on September 8, 2026. Notebook 10 executed all four code
cells in full mode in **2,439.492 seconds** and generated **5,015,409 rows** for
**1,671,803 sessions**. All 1,633 prediction parts and receipts are durable in S3.
The full gzip output is 296,087,864 bytes; its digest matches the notebook validator
and S3 metadata. The default replay reproduces 24 rows from eight actual test sessions.

[Full notebook receipt](../reports/research/competition_notebook_execution.json) ·
[Prediction manifest](../reports/research/competition_prediction.json) ·
[Cloud verification and object locations](../reports/research/competition_cloud_verification.json)

The receipt identifies the notebook bytes at source commit
`565ed1f15c9786a905a0cff70ac0a58ed8b1261c`; later canonical output publication preserves
those source cells. The complete executed full-mode notebook remains at the recorded
S3 object, separately from the convenient default-mode render committed in Git.

## Availability boundaries

The chronological research study fixes historical retrieval before its fitting
queries, selects task-specific model weights on the selection interval, and
evaluates the sealed models on the reserved interval. Competition inference
reuses those model weights. Its retrieval graph and historical item statistics
are refreshed from **all official training events**, which must precede the
observed competition sessions and have disjoint session IDs. No hidden test
targets are accepted by the inference API.

This refresh is a deployment operation with its own identity and directory.
It does not change the research evaluation or justify transferring an offline
score to the competition test. Updated feature distributions and time drift
remain limitations of the deployment result.

## Full workflow

With the completed research artifacts in `artifacts/research`, verified training
partitions in `artifacts/train`, and verified test partitions in `artifacts/test`:

```bash
.venv/bin/python scripts/run_inference.py --stage prepare \
  --train artifacts/train --test artifacts/test --threads 16 --memory-gib 64
.venv/bin/python scripts/run_inference.py --stage retrieval --threads 16 --memory-gib 64
/tmp/otto-analysis/bin/python scripts/execute_inference_notebook.py
```

Create the isolated analysis environment using [Reproducibility](REPRODUCIBILITY.md)
before the last command. That command executes full mode of the actual notebook,
including inference, full validation, preview and execution receipt. The managed
delivery launcher stages these verified inputs and invokes this sequence automatically.

The inference CLI uses canonical `part-*.parquet` event inputs. The recorded managed
run uses a 128 GiB host, 16 history threads and a 64 GiB aggregation memory limit.
A constrained local graph build also completed with one thread and a 16 GiB limit.
Changing resource limits preserves compatible history and graph receipts. For direct
CLI operation with a different verified test directory, pass `--test` explicitly to
`--stage predict`; the notebook's managed layout expects `artifacts/test`.

In the managed delivery environment, `OTTO_FULL_INFERENCE=1` enables the full
notebook mode. The launcher supplies the model thread count, feature worker
count, expected AWS account, region and checkpoint prefix. Optional publication
uses the standard execution-role credential chain; credentials and signed URLs
are never embedded in notebook sources.

## Completion checks

Each 1,024-session part has an atomic content digest and input contract. An
incompatible model seal, test dataset, retrieval identity or implementation
cannot silently reuse old predictions. Completed parts survive interruption.
The final gzip stream has a deterministic header and a complete validation pass:

- Exactly the official `session_type` and `labels` columns.
- Every observed session appears once for each of the three objectives.
- Every list contains 20 unique nonnegative integer item IDs.
- No duplicate, missing or unexpected session/objective rows.
- Complete output SHA-256 and traceable model and input identities.

Format and coverage validation are separate from Kaggle acceptance. No hidden
test score or leaderboard position is inferred from a valid file.

## Interpretation after selection

The delivery job also measures two whole-query block permutations per feature
family, native LightGBM TreeSHAP with an additivity check, and matched warm
feature-generation costs for the broad catalog, screened schema and selected
models. These diagnostics use selection queries and leave the evaluation seal
unchanged. Correlated-feature permutation effects are diagnostic rather than
causal, and SHAP explains ranking scores rather than calibrated probabilities.
