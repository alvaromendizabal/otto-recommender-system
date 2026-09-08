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

With verified official event partitions and the completed research artifacts:

```bash
.venv/bin/python scripts/run_inference.py --stage prepare
.venv/bin/python scripts/run_inference.py --stage retrieval --threads 1 --memory-gib 16
.venv/bin/python scripts/run_inference.py --stage predict
```

The inference CLI uses canonical `part-*.parquet` event inputs. Resource settings
are explicit; changing thread or memory limits preserves compatible history
and graph receipts. The full-data historical aggregation was run with one
thread to avoid excessive spill storage in a constrained local workspace.

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
