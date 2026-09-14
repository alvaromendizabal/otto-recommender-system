# OTTO — Round08 and Round09 manual execution

## Download / upload

Two complete rounds, each with 24 primary features, a 24-feature ablation, the same saved 134-feature control, two chronological folds, up to twelve new challenger models, five notebooks and nine saved result charts. These are two different scientific comparisons, not halves of one reduced experiment.

Upload **otto_feature_rounds08_09.zip** and **install_rounds08_09.py** to the top-level home folder of the existing JupyterLab space. Leave the ZIP zipped. Do not upload inside the repository or a prior round directory.

## Open the existing workspace

Use AWS console, US West (Oregon): SageMaker AI → Studio → QuickSetupDomain-20260902T115323 → default-20260902T115323 → Open Studio → JupyterLab → otto-dev. Start the existing space only when ready. Preserve its instance configuration, environment, storage, raw data, previous round folders and Git checkout. No replacement space or package installation is needed.

In File → New → Terminal:

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_rounds08_09.py"
```

Success prints **ROUNDS08_09_INSTALLED**; identical existing folders print **EXISTING_ROUNDS08_09_PRESERVED**. A changed or incomplete prior folder produces a stop, not an overwrite. Follow the printed NEXT instruction. This installs only the files; it starts no experiment.

## Round08 — rarity-weighted historical-session evidence

Open `otto_feature_round08/01_build_features.ipynb` using the existing OTTO - Notebook kernel. Press Shift+Enter for one code cell at a time:

| Code cell | Stage | Required success |
|---|---|---|
| 1 | Initialize launcher | KERNEL_READY |
| 2 | Contract tests and installed backend smoke | ROUND08_TESTS_PASSED |
| 3 | Prepare full 4,096-query cohort | SHARED_COHORT_READY |
| 4 | Extend postings for only new anchors | SHARED_POSTINGS_READY |
| 5 | Extend history for only missing sessions | SHARED_HISTORY_READY |
| 6 | Generate 24 + 24 feature matrices | ROUND08_FEATURES_READY |
| 7 | Check completed feature manifest | ROUND08_FEATURES_READY |

Save. Open `otto_feature_round08/02_run_comparison.ipynb`, again cell-by-cell:

| Code cell | Stage | Required success |
|---|---|---|
| 1 | Initialize | KERNEL_READY |
| 2 | Feature gate | ROUND08_FEATURES_READY |
| 3 | Six control replays, at most twelve challenger fits | ROUND08_SCREEN_COMPLETED |
| 4 | Recalculate metrics and nine charts | ROUND08_REPORT_READY |
| 5 | Inspect recorded decision and gate | ROUND08_REPORT_READY |

Save. Open `03_saved_results.ipynb`, select Run → Run All Cells, and save again. This notebook reads saved results only. The executed `00_round07_review.ipynb` and `04_feature_examples.ipynb` are optional explanatory notebooks, not prerequisites.

## Round09 — directed historical-session continuation

Proceed only after Round08 completes without an operational error. A completed negative Round08 score is acceptable: Round09 is a separately predeclared hypothesis, not a retest chosen from its score. Do not combine the rounds' features or choose new settings.

Open `otto_feature_round09/01_build_features.ipynb`, run one code cell at a time:

| Code cell | Stage | Required success |
|---|---|---|
| 1 | Initialize | KERNEL_READY |
| 2 | Contract tests and backend smoke | ROUND09_TESTS_PASSED |
| 3 | Read-only shared-history gate | SHARED_HISTORY_READY |
| 4 | Generate separate 24 + 24 feature matrices | ROUND09_FEATURES_READY |
| 5 | Feature completion check | ROUND09_FEATURES_READY |

Save. Open `otto_feature_round09/02_run_comparison.ipynb`, run cell-by-cell:

| Code cell | Stage | Required success |
|---|---|---|
| 1 | Initialize | KERNEL_READY |
| 2 | Feature gate | ROUND09_FEATURES_READY |
| 3 | Six control replays, at most twelve challenger fits | ROUND09_SCREEN_COMPLETED |
| 4 | Recalculate metrics and nine charts | ROUND09_REPORT_READY |
| 5 | Inspect recorded decision and gate | ROUND09_REPORT_READY |

Save. Open its `03_saved_results.ipynb`, Run All, and save. Neither result notebook initiates training.

## Resource and failure limits

The two rounds share historical extraction, but have separate source-aware feature checkpoints, models, statistics and reports. Round08 reuses all verified Round07 evidence; only new anchors and missing histories require scans. At most two additional full source Parquet scans serve both rounds. No raw JSON scan or old graph rebuild. Rarity weighting counts uncapped historical frequencies before 64-posting truncation; continuation uses original event indices and a frozen five-event/30-minute window.

Tests: 90 seconds outer cap per round. Shared prepare/postings/history: 200 seconds each. Features: 260 seconds per round. Screen: 260 seconds per round. Report: 90 seconds per round. These are safety limits, not predicted runtime or billing amounts. Keep 10 GiB disk free; worker cap is 26 GiB, DuckDB memory 6GB and spill 3GB. The application still bills until stopped; source storage can continue to bill afterward.

On any red error, PAUSED_CHECKPOINTED, PRIOR_STOP_DETECTED, or STOPPED_REVIEW_REQUIRED, **stop both rounds**. Do not continue into another notebook, change limits, install packages, delete/check out files, overwrite a folder or retry the unchanged stage. Use only the bundle command below and return evidence. If Round08 fails, bundle it and do not run Round09. If Round09 fails, bundle both. Never run notebook and terminal experiments concurrently.

## Save, bundle and return

After saving all notebooks, run these report-only commands:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round08/launch.py" bundle
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" bundle
```

Download both files from the JupyterLab file browser:

- `otto_feature_round08/otto_round08_return.zip`
- `otto_feature_round09/otto_round09_return.zip`

Do not upload the old Round07 ZIP as the next report. Return bundles exclude large source/feature/model binaries while retaining their integrity receipts. Save notebooks before bundling so execution outputs are included. After download, choose Studio → Running instances → otto-dev / JupyterLab → Stop. Do not delete the persistent space. Closing a browser or notebook kernel is insufficient to stop instance billing.

## Terminal-only execution

Each round's START_HERE.md has the exact one-stage-at-a-time terminal commands and success markers. Use that path instead of notebook execution, not alongside it. Do not paste every training command at once or wrap them in a shell loop. The launcher deliberately has no unattended `all` stage.

## What to evaluate

Both rounds: primary gain ≥0.003 over the exact original control, both chronological folds nonnegative, no pooled order-recall decline. A positive primary-minus-ablation alone does not pass. A pass earns a separate frozen temporal/cohort confirmation, not automatic retention. Small order samples, source truncation and repeatedly inspected fitting data remain limitations. Model settings are held fixed to isolate features; their optimality is not asserted. No achieved new score or leaderboard improvement exists until your real run completes, and a local fold does not establish that the recorded 0.60503 private target has been exceeded.

## Delivery boundary

Code and synthetic native-model tests were run locally. The real private Parquet data, full native controls and existing AWS environment were not replayed here; mandatory installed-backend smoke and exact control replay run in your workspace. There were no cloud launches, API mutations, Git pushes, S3 uploads or Kaggle submissions. GitHub main was read at 2638faa34afa427ed4ac4e92bba04deda57c688e; this delivery is not a synchronization or Git commit.
