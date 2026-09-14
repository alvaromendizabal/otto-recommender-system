# OTTO · Round 02 — frozen timing-feature replication

## One question, one bounded experiment

Do the **same 24 timing features** that improved Round 01 survive a larger sample of different fitting-session IDs? This is not an environment repair, algorithm search, or final-model transition.

The supplied `otto_notebook_check.zip` passes its three file hashes, two execution-count/Idle protocol checks and all seven saved-chart emission markers. No kernel registration or package installation is needed again. The named kernel is **OTTO - Notebook**. It uses `/opt/conda/bin/python` for the interface and launches the unchanged project's `.venv/bin/python` for ML.

The new cohort has 4096 fitting sessions, excluding all 1024 Round 01 sessions. It is new only relative to that pilot; the wider training pool has been explored previously. It is **not an untouched holdout** or a new historical window.

## Exact browser-to-notebook steps

1. Download `otto_feature_round02.zip` to your computer. Leave it zipped.
2. Open `https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2`. Select Studio → domain `QuickSetupDomain-20260902T115323` → profile `default-20260902T115323` → JupyterLab → existing `otto-dev`. Open it; use Run space only if it is stopped. Keep existing CPU/storage. Do not create or delete a space.
3. In the JupyterLab file browser navigate to your home/top-level directory. Upload the ZIP there, not inside `otto-recommender-system` or `otto_feature_round01`.
4. Open a JupyterLab Terminal and paste:

```bash
cd "$HOME"
/opt/conda/bin/python -m zipfile -e "$HOME/otto_feature_round02.zip" "$HOME"
```

5. Open `otto_feature_round02/01_run_timing_replication.ipynb`. Select **OTTO - Notebook**. Run the first cell with Shift+Enter. It must print `KERNEL_READY` and receive a number. Then choose Run → Run All Cells. This notebook intentionally runs **this new** experiment, not the completed Round 01 experiment.
6. Continue only if the four phases complete: tests, features, screen and report. Expected messages are `ROUND02_FEATURES_READY`, `ROUND02_SCREEN_COMPLETED`, and `ROUND02_REPORT_READY`.
7. Open `02_saved_results.ipynb`, select the same kernel, and choose Run → Run All Cells. It sends seven Plotly JSON payloads explicitly, with a completion marker after each. It does **not** fit models or regenerate features. The final marker is `ROUND02_NOTEBOOK_REVIEW_COMPLETE`.
8. Save both notebooks. Download `otto_round02_return.zip` from the `otto_feature_round02` folder and attach it to ChatGPT. To include the most recently saved notebook bytes, the report-only command below may be used after saving:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round02/launch.py" bundle
```

9. Stop unused compute in Studio → Running instances → `otto-dev` / JupyterLab → Stop. Do not delete the space. A notebook kernel shutdown alone does not stop the billable application. The scripts do not stop the application for you.

## Stop/continue handling

**STOPPED_REVIEW_REQUIRED**, a red error cell, a resource limit or failed assertion: stop that phase. Do not reinstall, reset Git, delete files, change thresholds or repeat it blindly. Return the available ZIP. Completed feature chunks and native models remain in `outputs`.

**PAUSED_CHECKPOINTED**: the phase stopped at a complete-unit boundary near its work budget. The notebook intentionally halts instead of launching subsequent work. Return the ZIP so measured progress can be inspected before authorizing a resume. No unattended retry loop is present.

If the first simple kernel cell does not finish within 15 seconds, do not start computation. Return the error/screenshot. The existing kernel registration need not be recreated.

## Terminal alternative — choose this OR notebook execution

Run commands one at a time and stop on a failure:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round02/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round02/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round02/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round02/launch.py" report
```

Then open the saved-results notebook. Do not launch notebook and terminal experiments concurrently. A file lock prevents concurrent experiment writes.

## Resource limits and preservation

- Tests: 60-second process cap; isolated native-model fixture only, not production model fitting.
- Features: 300-second useful-work / 320-second outer process cap; 64 complete sessions per immutable chunk; 15-second heartbeats; worker RSS cap 26 GiB.
- Screening: 240-second useful-work / 260-second outer cap; 12 native models maximum, one immutable checkpoint per fold/arm/objective. No score-driven early stopping among the prespecified arms.
- Reporting: 60-second outer cap; reads saved evidence only.
- Require 6 GiB free before work. Reuse existing raw/corpus/retrieval/graph files. No downloads, graph rebuilding, installations, AWS resource changes, Git writes or submissions.
- These are process limits, not an AWS billing cap. Startup/shutdown time and retained storage are separate.

Every phase produces a small return ZIP on success or failure. It excludes feature matrices and native weights; those stay on the persistent space volume. The ZIP is a diagnostic return, **not a full disaster-recovery backup**. No new S3 upload occurs automatically.

## What is measured

A new 134-feature control versus that same representation plus the exact 24 timing features. The 24 support breadth/entropy fields are **not** added. There are 12 new models because the controls need training on the new cohort. The old pilot's score is not a valid matched control for different sessions.

The first 8 prior sessions must reproduce their stored control and timing columns exactly before new work. The first 16 new sessions undergo a second complete numerical reconstruction. Existing historical graphs and their exact Aug-16 cutoff are checked and reused.

Two forward validation folds each have 1024 sessions. Training queries precede the validation start by six hours; training positives are censored at the exclusive validation-start cutoff **before** negative sampling. Candidate pools, model settings, 150 rounds and seeds are fixed. All full distinct validation target denominators remain, including targets absent from retrieval.

A gain of at least 0.003 with nonnegative changes in both folds and no pooled order decline permits later temporal confirmation. An interval including zero remains uncertain. No branch of this decision promotes a production feature recipe or claims to beat the 0.60503 historical private target.

## Files and status

`00_experiment_design.ipynb` explains the prior measured finding and the new question. Its executed charts describe Round 01, not a fabricated Round 02 result. `01_run_timing_replication.ipynb` starts the new computation. `02_saved_results.ipynb` is review-only. Use the latter for subsequent chart viewing.

GitHub was read at preparation time and main was `2638faa34afa427ed4ac4e92bba04deda57c688e`. This manual package remains outside the checkout; it has **not** been committed, pushed, or merged. Your checkout/data are not wiped or modified. Experimental results will require a separate reviewed publication milestone.

## Local testing boundary

See `evidence/PACKAGE_VALIDATION.json`. Tests exercise pure equations, disjoint selection, temporal censoring, ranking denominators, native model serialization, checkpoints, report recomputation, and saved-result notebook completion with a synthetic fixture. The private AWS graphs, actual Parquet adapters and full 4096-session run have **not** been executed here. The eight-session real-data compatibility check is an explicit run-time gate, not claimed local validation.
