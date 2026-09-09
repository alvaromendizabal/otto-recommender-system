# Competition inference and model replay

**Accepted Kaggle result: 0.56842 private / 0.56862 public.** The frozen reference model used the verified official competition prefixes. Kaggle reports **Complete (after deadline)**.

**Source correction:** the initial Kaggle submission is invalidated because its
input contained full post-competition test sessions, including future events.
The 0.93554 / 0.93583 scores are retained as incident evidence, not valid model
performance. Replacement inference uses the attested official truncated test.
See [the source audit](../reports/submissions/data_provenance_audit.json).

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

## Download and submit the completed full run

**The corrected full run is complete and verified.** Its gzip contains 289,541,110 bytes and SHA-256 `49e332948b47396fb609eaaac09a4cf3089468c38b58a15203075d6c107260f2`. Kaggle accepted this exact file and scored it at **0.56842 private / 0.56862 public**. No further upload is needed for this release.

To download the exact verified artifact in the SageMaker project terminal:

```bash
cd "$HOME/otto-recommender-system" &&
mkdir -p artifacts/competition &&
aws s3api get-object --region us-west-2 \
  --bucket otto-recsys-560403859723-us-west-2 \
  --key ranking/research/55ad451e895863af311e4a917a6fe0d4ab9165ad6b406fffb066d25bf0af4754/delivery/90bb97d39ec49999261d73b166581258961362a6ac4ee237c7c341c7c957dfc8/inference/prediction/submission.csv.gz \
  --version-id i0gMnNzo7V1V.SpQoFqGW5XberD.8QcG \
  artifacts/competition/submission.csv.gz &&
sha256sum artifacts/competition/submission.csv.gz
```

The download is available for reproduction and inspection. Its digest must match the value above. [Managed run](../reports/submissions/competition_inference.json) · [Current submission receipt](../reports/submissions/kaggle_submission.json).

A valid submission contains the columns `session_type` and `labels`, with one
row per session and action and 20 unique product IDs per list: **5,015,409 rows
for 1,671,803 sessions**. The compact `inference_replay.csv` has only 24 rows and
is a software review example, never a competition submission.

The official truncated input has **6,928,123 events** and a raw-file size of
**402,090,304 bytes**. The original full release had **13,851,293 events** and
**750,426,722 bytes**. Both contain the same session IDs, so coverage checks
alone cannot distinguish them. The new inference guard verifies the raw-source
attestation, conversion manifest, and every Parquet partition hash before work
begins. See [input attestation](../reports/submissions/competition_input.json).

To prepare these exact inputs after downloading `test.jsonl.zip` from the
[competition Data page](https://www.kaggle.com/competitions/otto-recommender-system/data):

```bash
python -m zipfile -e test.jsonl.zip data/competition/raw
uv run --frozen --extra ml python scripts/prepare_competition_input.py \
  --raw data/competition/raw/test.jsonl --output artifacts/test
```

The existing frozen models are reused. A new inference namespace prevents any
old prediction part from being treated as compatible. The replacement file passed source, coverage, content, and execution checks.
The accepted Kaggle score is recorded below; it is distinct from the offline temporal study.

The [50-file collection](ROBUSTNESS.md#path-to-50-submission-files) remains a
separate milestone. Each file needs distinct predictions and a recorded model
or ensemble recipe. Repeating validation seeds does not automatically produce
additional competition submissions.

## Recorded Kaggle result

The submission list and **Submission Details** panel agree on the corrected result:

| Evidence | Observed value |
|---|---|
| File | `submission.csv.gz` · 289,541,110 bytes |
| Status | **Complete (after deadline)** · Success |
| Public score | **0.56862** |
| Private score | **0.56842** |
| Observation time | 2026-09-09T23:49:34.868965+00:00 |
| Coverage | 1,671,803 sessions · 5,015,409 rows |
| Frozen model | Reference seed 20260908 · 102 selected features |

[Open the account's submissions](https://www.kaggle.com/competitions/otto-recommender-system/submissions)
· [Accepted submission receipt](../reports/submissions/kaggle_submission.json)

The description includes the complete SHA-256 above. The receipt binds the browser-observed
scores to the independently verified file, exact S3 version and official input attestation.
The interface did not expose a numeric submission ID, so that field remains null.

This is the valid competition result for the frozen reference model. Its earlier reserved
**0.584392** score uses different sessions and must not be substituted for the Kaggle score.
The highest observed temporal score, **0.590759** for middle seed 20260908, also uses a
different cohort. No new model is selected using these evaluation outcomes. A late
submission does not establish an official leaderboard rank or state-of-the-art performance.

## Historical invalidated submission

**Historical incident: the scores in this section are invalidated.**

The signed-in submission page and its **Submission Details** panel both reported:

| Evidence | Observed value |
|---|---|
| File | `submission.csv.gz` · 296,087,864 bytes |
| Status | **Complete (after deadline)** · Success |
| Public score | **0.93554** |
| Private score | **0.93583** |
| Observation time | September 9, 2026, 20:18:19 UTC |
| Coverage | 1,671,803 sessions · 5,015,409 rows |

[Open the account's submissions](https://www.kaggle.com/competitions/otto-recommender-system/submissions)
· [Historical receipt](../reports/submissions/history/adc1c7d496249b8a37550c4c077dba8d12bc413d0fe89a80221472cd813a7ef3/kaggle_submission.json)

The receipt links the browser-observed status and displayed score precision to the
full file's SHA-256, exact S3 version, prediction input identity and inference evidence.
The inspected interface did not expose a numeric submission ID; the receipt records
that field as null rather than inventing one. The description includes the complete
file digest so this submission can be identified in the account.

**Interpretation: invalid evaluation.** The source audit confirmed that the
first file used the full test sessions released after the competition. These
include events that the official task withholds as targets. The organizer
[documents the full release](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md).
For example, official session 12899779 has one observed click; the original input
also supplied its next click. Thus **0.93583 cannot be compared with the historical
winning 0.60503** or cited as model performance. Kaggle acceptance verifies neither
input provenance nor freedom from leakage.

The training-only **0.584392** reference evaluation and the frozen temporal study
are unaffected. No feature, seed, or model choice is being changed using the
invalidated Kaggle score.

## Verified full execution

The corrected managed run completed on **September 9, 2026 at 21:43 UTC**. Full Notebook 10 executed four code cells in **2,413.288 seconds**, generating all 5,015,409 task rows. The downloaded gzip was independently validated against the official session ledger in a further 23.118 seconds. The refreshed compact replay reproduces 24 rows from eight official-prefix sessions with the frozen native models.

[Full notebook receipt](../reports/research/competition_notebook_execution.json) · [Prediction manifest](../reports/research/competition_prediction.json) · [Cloud verification](../reports/research/competition_cloud_verification.json)

The full-mode notebook and execution receipt remain tied to source commit `1fb654831a47cfddf1c47d5b51030d04f2746841`. The later GitHub notebook is the convenient default-mode replay with the current delivery status. Historical inference receipts are retained beside the invalidated submission record; current receipts refer only to the official-prefix run.

## Availability boundaries

The chronological research study fixes historical retrieval before its fitting
queries, selects task-specific model weights on the selection interval, and
evaluates the sealed models on the reserved interval. Competition inference
reuses those model weights. Its retrieval graph and historical item statistics
are refreshed from **all official training events**, which must precede the
observed competition sessions and have disjoint session IDs. The inference API now rejects any test input whose bytes do not match the
attested official prefixes. Training/test chronology alone did not detect the
original within-session target contamination.

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
.venv/bin/python scripts/run_inference.py --stage retrieval --test artifacts/test --threads 16 --memory-gib 64
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
