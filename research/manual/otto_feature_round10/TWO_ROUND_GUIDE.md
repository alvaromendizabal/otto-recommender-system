# OTTO — finish Round 09, then independently run Round 10

## Start here: what your two uploaded archives establish

Round 08 completed. The 134-feature control scored 0.5692312343124653; rarity-weighted neighbors scored 0.5625975289453764 (−0.006633705367088849); equal-weight neighbors scored 0.5658773777030842 (−0.003353856609381034). The primary lost on the earlier time fold and gained on the later fold. Its net differences were −12 click hits, −8 cart hits and −1 order hit. Neither arm passed the original advancement gate. The descriptive uncertainty intervals include zero; do not call the effect statistically decisive.

Round 09's submitted archive has no `outputs/` files. Its build, comparison and result notebooks have zero executed code cells. Its already-executed historical review and synthetic example are not experiment results. This is a missing run record, not a negative Round 09 result. The installer checks the actual workspace before directing you; a completed workspace run must not be retrained just because an older ZIP was uploaded.

The next two investigations are therefore pending Round 09 and new Round 10. Do not rerun Round 08's extraction, features, models or reports to clear an old display. Preserve all earlier round directories, private artifacts, raw data and the Git checkout.

## What is delivered

| Property | Round 09 — pending, unchanged | Round 10 — new |
|---|---|---|
| Hypothesis | Different products appearing after the latest query product in similar historical sessions | A candidate product progressing from click to cart/order within similar historical sessions |
| Primary | 24 directed product-continuation features | 24 ordered same-product action-progression features |
| Ablation | 24 otherwise-matched unsigned-window features | 24 otherwise-matched unordered co-action features |
| Control and each challenger | 134; 158; 158 columns | 134; 158; 158 columns |
| Data | Same 4,096 fitting sessions and 400 candidates | Same 4,096 fitting sessions and 400 candidates |
| Models | Six original-control replays, at most twelve new challenger fits | Six original-control replays, at most twelve new challenger fits |
| Presentation | Five notebooks, nine saved-result Plotly charts | Five notebooks, nine saved-result Plotly charts |
| New history work | Read completed Round 08 cache | Read completed Round 08 cache |

Both protocols are specified before seeing Round 09's result. A completed negative result does not block Round 10. An operational error, pause or incompatible checkpoint does block it. There is no combined 48-feature arm, candidate expansion, algorithm search, ensemble search, new data download, new source scan, graph rebuild, S3 upload, Git write or submission.

The Round 09 directory is included byte-for-byte from its previously delivered package so the installer can preserve your existing code, notebooks and outputs. Its older installation prose references the earlier 08/09 delivery; **use this new guide for installation and sequencing**, not the older installer. Do not overwrite its files manually.

## 1. Download and open the existing space

Download `otto_feature_rounds09_10.zip` and `install_rounds09_10.py`. Leave the ZIP zipped. No local Python installation is needed on Windows.

Open AWS in your browser. Choose **US West (Oregon)**, then **SageMaker AI → Studio → QuickSetupDomain-20260902T115323 → default-20260902T115323 → Open Studio → JupyterLab → otto-dev**. This is the recorded existing project setup; no new live AWS inventory was performed for this delivery. Start the existing stopped application only when ready. Keep its instance configuration, storage, kernel and environment. Do not create another space.

In JupyterLab's file browser, navigate to the home/top-level folder, outside the repository and all round folders. Use Upload Files to upload both downloaded files. Open **File → New → Terminal**.

## 2. Install without overwriting existing work

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_rounds09_10.py"
```

For a new Round 10 installation, expect `ROUNDS09_10_INSTALLED`. If both exact packages already exist, expect `EXISTING_ROUNDS09_10_PRESERVED`. Follow the printed `NEXT:` instruction. The usual case is existing Round 09 preserved and Round 10 installed.

The installer validates the archive, notebook sources and both destinations before installing either. Notebook execution outputs and extra private files are preserved. Changed/missing package source, an unsafe path or a mismatched ZIP causes a stop. Do not delete, rename, unzip over, or force-replace existing round folders to bypass it. Installation runs no experiment and installs no Python packages.

If the installer finds an existing Round 09 report, skip its training and continue at Round 10. If both reports exist, save the notebooks and run only the final bundle commands. If it reports `PRIOR_STOP_DETECTED` or `STOPPED_REVIEW_REQUIRED`, stop and return the exact message with the available bundles.

## 3. Run Round 09 — feature notebook

Open `otto_feature_round09/01_build_features.ipynb`. Keep the working **OTTO - Notebook** kernel. The launcher uses the original project interpreter at `~/otto-recommender-system/.venv/bin/python`, independently of the interactive kernel. Do not reinstall packages or select a newly created environment.

Run one numbered code cell at a time with **Shift+Enter**. Read the result before moving on. Do not use Run All here.

| Code cell | Purpose | Required output |
|---:|---|---|
| 1 | Initialize | `KERNEL_READY` |
| 2 | Tests, including mandatory installed-backend smoke | `ROUND09_TESTS_PASSED` |
| 3 | Require completed shared history | `SHARED_HISTORY_READY` |
| 4 | Generate both 24-feature representations | `ROUND09_FEATURES_READY` |
| 5 | Confirm completed feature manifest | `ROUND09_FEATURES_READY` |

The shared-history gate is a check, not permission to rerun Round 08. Save the notebook with **Ctrl+S**.

## 4. Run Round 09 — comparison notebook

Open `otto_feature_round09/02_run_comparison.ipynb`. Run code cells individually.

| Code cell | Purpose | Required output |
|---:|---|---|
| 1 | Initialize | `KERNEL_READY` |
| 2 | Require completed features | `ROUND09_FEATURES_READY` |
| 3 | Replay six controls, then fit challengers | `ROUND09_SCREEN_COMPLETED` |
| 4 | Recalculate statistics and create charts | `ROUND09_REPORT_READY` |
| 5 | Print the recorded decision and completion gate | `ROUND09_REPORT_READY` |

Every saved control must reproduce its original per-session validation hits before the first challenger is fitted. Native model reload must reproduce predictions. The screen adds no rejected earlier-round columns and uses all 400 validation candidates with complete denominators.

Save. Open `otto_feature_round09/03_saved_results.ipynb`, choose **Run → Run All Cells**, and save. This notebook displays saved results only. A negative feature gain is an experimental outcome, not an execution error. `ROUND09_REPORT_READY` permits the independent Round 10 investigation even when Round 09 loses.

## 5. Run Round 10 — feature notebook

Open `otto_feature_round10/01_build_features.ipynb`. Run code cells one at a time.

| Code cell | Purpose | Required output |
|---:|---|---|
| 1 | Initialize | `KERNEL_READY` |
| 2 | Tests and installed-backend smoke | `ROUND10_TESTS_PASSED` |
| 3 | Check completed Round 08 history and Round 09 report | `SHARED_HISTORY_READY` and `ROUND09_REPORT_READY` |
| 4 | Build both same-product action representations | `ROUND10_FEATURES_READY` |
| 5 | Confirm the feature manifest | `ROUND10_FEATURES_READY` |

The data adapter verifies the actual prior source code, completed receipts, cached arrays and source identity. It invokes only the earlier reader's context/require/evidence operations. It exposes no prepare/postings/history stage. Missing data stops rather than silently downloading or rebuilding it.

Save with **Ctrl+S**. The features are committed before training labels are read. The bounded in-process historical-session summary cache is discarded when the feature stage exits; no persistent index is built.

## 6. Run Round 10 — comparison notebook

Open `otto_feature_round10/02_run_comparison.ipynb`. Its five code cells initialize, require features, screen, report and display the decision. Required markers, in order, are `KERNEL_READY`, `ROUND10_FEATURES_READY`, `ROUND10_SCREEN_COMPLETED`, `ROUND10_REPORT_READY`, `ROUND10_REPORT_READY`.

Save. Open `otto_feature_round10/03_saved_results.ipynb`, **Run All**, and save. It displays nine saved Plotly charts without training. No full-cohort score is embedded in the newly supplied notebook; charts appear after your report stage completes.

The optional `00_round08_review.ipynb` is already executed from your uploaded statistics (eight charts). `04_feature_examples.ipynb` is already executed from labeled synthetic events (five charts); it illustrates the actual formula implementation and is not a competition result.

## 7. Stop rules and budgets

A red error, `PAUSED_CHECKPOINTED`, `PRIOR_STOP_DETECTED`, or `STOPPED_REVIEW_REQUIRED` means **stop both rounds**. Do not run later stages, increase the budget, modify the protocol, reinstall anything, delete artifacts, clear failures by retraining, or repeat the same failed command. Run only the bundle commands and return the evidence. A planned pause is a checkpoint boundary, not permission for an unchanged retry.

| Stage, per round | Outer process cap |
|---|---:|
| Tests (synthetic fixtures plus actual-backend smoke) | 90 seconds |
| Features | 260 seconds |
| Matched screening | 260 seconds |
| Report | 90 seconds |

These are safety limits, not expected runtimes, future-delivery promises, or cost guarantees. Useful-work deadlines are 240 seconds for feature and screen stages. The launcher stops a worker above 26 GiB RSS; each data stage requires 10 GiB free disk. Fifteen-second UTC heartbeats, 64-query feature checkpoints, immutable model receipts and a shared execution lock expose progress and preserve completed work.

Tests train tiny synthetic fixture models; those are not project fits or scores. Each real screen fits at most twelve challenger models. The two experiment limits together permit at most twenty-four new project models, with zero refits of the six saved project controls. Do not run notebook and terminal stages simultaneously. Stopping a process does not stop the SageMaker application.

## 8. Save and return results

After saving every notebook, run the following **bundle-only** commands. They do not retry stages, fit models, rebuild features, or include large raw/history/feature/model binaries.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" bundle
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" bundle
```

Download, by right-clicking each file in JupyterLab and choosing Download:

```text
otto_feature_round09/otto_round09_return.zip
otto_feature_round10/otto_round10_return.zip
```

Attach these two ZIPs in your next message. When Round 09 fails, bundle it and stop before Round 10; an unrun Round 10 ZIP is optional failure context, not a result. When Round 10 fails, return both available bundles. Do not send the old Round 08 result as the new report. The last bundle command after Ctrl+S matters because earlier automatic bundles may contain notebook bytes saved before execution.

When finished, go to **Studio → Running instances**, find **otto-dev / JupyterLab**, and choose **Stop**. Do not delete the persistent space. Stopping the application preserves files; deleting the space deletes its storage/data. A stopped kernel or closed browser is insufficient to stop the application. Storage charges can remain after compute stops. AWS reference: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html

## 9. The evidence gate and next scientific decision

Each primary arm must beat the original control by at least **+0.003 pooled weighted Recall@20**, show nonnegative gain on both chronological folds, and avoid pooled order-recall loss before separate confirmation is proposed. Beating only the ablation is insufficient. This gate proposes confirmation; it never automatically promotes features, changes the control or submits to Kaggle.

Compare the primary and ablation independently to the same control. The two rounds are not sequential additive stacking. The report includes all target-specific recalls, exact hit contributions, descriptive paired intervals, within-pool candidate ceilings, training-only nonzero support, and redundancy against the 134 controls. No feature filtering, parameter tuning or hypothesis selection is based on validation-label diagnostics in this package.

Repeated inspection of the same fitting cohort is adaptive exploration. The descriptive intervals do not account for all earlier hypotheses or all temporal dependence. Any apparent winner needs a frozen, different-session/different-time confirmation. An observed feature loss does not prove its entire feature class useless or the model capacity optimal. A coverage oracle is not achieved Recall@20. The recorded historical private target 0.60503 is not directly comparable to these internal folds.

After these two rounds, inspect support, action-specific errors and redundancy before choosing another family. A promising result earns independent confirmation. Sparse same-product funnels call for diagnosing retained-history coverage before scaling. Repeated negative results with saturated graph/history evidence favor a genuinely distinct embedding/retrieval-source investigation rather than retuning these formulas repeatedly. Candidate expansion must remain a separate controlled comparison.

## 10. Terminal alternative (only instead of notebook execution)

Run one command, inspect its success marker, and only then copy the next. No shell loop or chained all-stages command is provided. Do not bypass `launch.py` with `experiment.py`, because the launcher supplies the outer limits, locks, logs and prior-failure checks.

### Round 09

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" tests
```
Continue only after `ROUND09_TESTS_PASSED`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" features
```
Continue only after `ROUND09_FEATURES_READY`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" screen
```
Continue only after `ROUND09_SCREEN_COMPLETED`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" report
```
Continue only after `ROUND09_REPORT_READY`. Display and save the results notebook.

### Round 10

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" tests
```
Continue only after `ROUND10_TESTS_PASSED`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" features
```
Continue only after `ROUND10_FEATURES_READY`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" screen
```
Continue only after `ROUND10_SCREEN_COMPLETED`.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" report
```
Continue only after `ROUND10_REPORT_READY`. Display/save the results notebook; run the final bundles.

Terminal execution does not populate execution counts in the build/comparison notebooks. Logs and receipts remain authoritative in that path; the saved-results notebook displays identical reported metrics.

## Publication and verification limits

This delivery is not a Git commit, merged PR, AWS synchronization, backup or Kaggle submission. GitHub main was read at 2638faa34afa427ed4ac4e92bba04deda57c688e. No cloud account changes or large private-data experiments were performed in preparation. Preparation used uploaded statistics and clearly labeled synthetic tests. The actual installed DuckDB/Parquet backend must pass its smoke test in your workspace, then real-source and control-replay checks must pass before fitting. The companion validation record reports the tests actually executed. Do not publish private returns, session-level results, raw data, or native artifacts merely to make the repository current.
