# OTTO — Feature Round 01

**One experiment: 48 new observed-intent support/timing features. Your workspace setup already passed. Do not wipe it or repeat the previous preparation commands.**

This package is separate from the unexecuted candidate-frontier experiment. It leaves the existing 400 candidates unchanged and tests richer representations of them. It is not a new submission or an assertion of improvement. The real-data experiment has not been run by the assistant.

## Before you start

Use the existing `otto-dev` JupyterLab space, Oregon (`us-west-2`), domain `QuickSetupDomain-20260902T115323`, profile `default-20260902T115323`. Keep its existing instance/storage settings. Do not launch another project, CloudShell workflow, GPU job or new environment.

Your returned report says clean GitHub main `2638faa34afa427ed4ac4e92bba04deda57c688e`, correct training and truncated test data, all five research input groups verified, package versions matching, and 43.7 GiB free. This is the report's state, not a live filesystem observation. The experiment rechecks its actual source/input requirements before computation.

## 1. Upload and extract

Download `otto_feature_round01.zip` to Windows Downloads; leave it zipped. Open the existing SageMaker space. In the JupyterLab file browser, navigate to the home/top-level folder and upload the ZIP there, NOT inside `otto-recommender-system`.

Open a JupyterLab Terminal and run:

```bash
cd "$HOME"
/opt/conda/bin/python -m zipfile -e "$HOME/otto_feature_round01.zip" "$HOME"
```

The new folder is `/home/sagemaker-user/otto_feature_round01`. Do not upload into or modify the existing Git checkout.

## 2. Open the execution notebook

Open `otto_feature_round01/01_run_and_analyze.ipynb`. Choose the normal **Python 3** kernel. Select **Run → Run All Cells**. The notebook delegates computation to the EXISTING `/home/sagemaker-user/otto-recommender-system/.venv/bin/python`; it does not install packages or require a new kernel.

The first optional notebook, `00_feature_design.ipynb`, contains an executed mathematical illustration. Its toy charts are explicitly NOT real OTTO experiment scores. It explains exactly what the new features mean. It is not necessary to rerun it before the actual experiment.

## 3. Watch the bounded stages

The execution notebook runs the following stages in order. Do not start another copy of the notebook or a parallel terminal experiment.

| Stage | What actually happens | Limit |
|---|---|---|
| Tests | Independent formula/checkpoint/decision tests with warnings as errors | 45-second process cap |
| Features | Verify existing data and graphs; repeat a 16-session numerical smoke; construct 48 features for 1,024 existing fitting sessions | 300-second work cap; outer process cap 315 seconds |
| Screen | Read six tiny version-pinned S3 receipts; compare 3 new feature arms against saved 134-feature control predictions | 18 new models maximum; 240-second work cap; outer cap 255 seconds |
| Report | Recompute pooled metrics from saved hit counts and generate 7 Plotly charts | 60-second work cap; outer cap 75 seconds |

After the outer cap, a terminated process gets up to 8 seconds to finish reporting before forced termination. These are process-work bounds, NOT automatic shutdown of the billable SageMaker app. Heartbeats appear every 15 seconds; progress counts identify completed feature chunks and models. A memory watchdog stops work if the monitored Python process exceeds 26 GiB RSS; this is not a reservation or guarantee against an operating-system out-of-memory kill.

Expected success markers:

```text
RESULT: ROUND01_FEATURES_READY
RESULT: ROUND01_SCREEN_COMPLETED
RESULT: ROUND01_REPORT_READY
```

The expected small reference transfer is 57,969 bytes, not a raw-data download. References are private S3 objects from the already completed feature-value pilot. An authorization or version failure is a stop for review. The code never needs Kaggle credentials.

## 4. Read the result, not just the score

The primary comparison is **all 48 additions versus the same saved 134-feature control**. The other two arms add only 24 support features or only 24 timing features. Their matched comparisons also measure each group's contribution when the other group is present.

The control's pooled forward-fold score should reproduce approximately **0.484737**. It is not the 0.56842 Kaggle score: the data/protocol are different. No new score is supplied in advance.

The notebook displays seven interactive charts: pooled comparison, descriptive uncertainty, chronological-fold consistency, objective-specific recall, group ablations, training-only support, and overlap with existing features. It also prints the primary stop/replicate decision. A point gain of at least 0.003 with no negative fold or order change permits only a proposed larger fitting-only replication. An interval including zero remains uncertain; it does not establish feature retention or a leaderboard improvement.

## 5. Download the return ZIP and stop the app

Download `otto_feature_round01/otto_round01_return.zip` through the JupyterLab file browser (right-click → Download). Attach that file in ChatGPT. It includes small reports, per-session validation hit statistics, run receipts and logs, plus the interactive HTML report after success. It does not include bulky feature matrices or native models.

Then return to Studio → **Running instances** → find `otto-dev` / JupyterLab → **Stop**. Stop the application; **do not delete the space**. Files persist when the app stops. Retained storage may remain billable.

Source: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html

## On an error or timeout

Stop immediately if the notebook shows an exception, `STOPPED_REVIEW_REQUIRED`, an environment mismatch, or an input/checksum problem. Do not install packages, change the budget, reset Git or repeat an unchanged failure. Download the available return ZIP. If no ZIP exists, download `outputs/features.log`, `outputs/screen.log`, or the indicated stage log and return it.

Completed feature chunks and model receipts remain in `outputs/`; a later approved resume verifies their hashes and contracts and reuses them. Model files without valid receipts are preserved as orphans for inspection rather than silently overwritten. Do not delete them.

## Terminal alternative (choose this OR the notebook execution)

Use this only instead of the notebook's compute cells. Run one line, verify success, then the next:

```bash
/opt/conda/bin/python "$HOME/otto_feature_round01/launch.py" tests
/opt/conda/bin/python "$HOME/otto_feature_round01/launch.py" features
/opt/conda/bin/python "$HOME/otto_feature_round01/launch.py" screen
/opt/conda/bin/python "$HOME/otto_feature_round01/launch.py" report
```

The notebook plotting cells can then read the existing result without refitting. The self-contained `outputs/round01_report.html` can also be downloaded and opened locally. The ZIP is refreshed after each compute/report stage.

## Preservation and GitHub status

No package installation, raw-data modification, graph rebuild, candidate change, automatic feature removal, Git commit/push, IAM/cloud-resource change, competition-test read, or submission occurs. All results are saved on this space's persistent volume. **This package does not automatically upload them to S3.** Download the return ZIP before stopping the app; native models and feature chunks stay in the space.

Your existing checkout remains pinned to `2638faa`. The new manual package is an attachment outside Git and has NOT been committed/merged into the repository. The report will let us publish verified results and code in a later, explicit source-publication milestone without pretending a local experiment is already on GitHub.

## Testing boundary

Local tests verify feature definitions, leakage guards, mathematical edge cases, checkpoint behavior and reporting. The assistant's local runtime did not have Polars/PyArrow, so it could not execute the full real-Parquet/AWS integration here. Those packages are present in your returned environment inventory. Input parsing, source hashes, target parity and saved-control compatibility are checked as part of your bounded run; failure stops rather than fabricating a result.
