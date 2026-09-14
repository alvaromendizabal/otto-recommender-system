# OTTO — Next manual step: Round 06

**Round05 completed. Keep the 134-feature control. Do not rerun Round05 indexing or training.**

This package tests 12 repeat-conditioned graph-affinity features versus a 12-column global-context ablation, with fixed candidates and models. No new index, raw-data scan or installation is required. This is a new exploratory feature screen, not a new Kaggle score.

## 1. Download two files

Download `otto_feature_round06.zip` and `install_round06.py`. Leave the ZIP zipped. Upload BOTH to the existing JupyterLab home/top-level folder, not the repository or any earlier round folder.

From only ChatGPT open: open the AWS console → choose US West (Oregon) → SageMaker AI → Studio → domain `QuickSetupDomain-20260902T115323` → profile `default-20260902T115323` → Open Studio → JupyterLab → existing **otto-dev**. Use Run space only if it is stopped and you are ready to work. Keep its existing CPU, storage and environment. Do not create/delete a space.

## 2. Safe install

Open File → New → Terminal. Paste:

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_round06.py"
```

Expect `ROUND06_INSTALLED`. If the exact folder already exists, expect `EXISTING_ROUND06_PRESERVED`; follow the printed NEXT instruction. Nothing is overwritten. Changed source or a previous stopped stage means preserve everything and return evidence, not reset/reinstall.

## 3. Review and create the features

Open `otto_feature_round06/00_review_and_design.ipynb`. It contains an executed Round05 review with six Plotly plots. Reading it runs no training. Its cells may be rerun to redisplay saved data.

Open `01_build_context_features.ipynb`. Keep the existing **OTTO - Notebook** kernel that worked for Round05. No new kernel is needed. Run one code cell at a time with Shift+Enter:

| Cell | Action | Continue only after |
|---|---|---|
| 1 | Initialize the explicit Round06 launcher | `KERNEL_READY` |
| 2 | Tests, including tiny synthetic end-to-end fixtures | `ROUND06_TESTS_PASSED` |
| 3 | Verify inputs and build 12+12 features once | `ROUND06_FEATURES_READY` |
| 4 | Check the completed feature manifest | `ROUND06_FEATURE_GATE_PASSED` |

Save the notebook. Feature cap is 180 seconds useful work, 200 seconds outer process. This is a safety cap, not a runtime estimate.

**Stop on any red error, `PAUSED_CHECKPOINTED`, `PRIOR_STOP_DETECTED`, or `STOPPED_REVIEW_REQUIRED`. Do not run later cells, raise limits or retry unchanged. Go to Step 5's bundle command and return the ZIP.**

## 4. Matched screen and Plotly report

Only after `ROUND06_FEATURE_GATE_PASSED`, open `02_run_context_comparison.ipynb`. Same kernel; Shift+Enter one cell at a time:

| Cell | Action | Continue only after |
|---|---|---|
| 1 | Initialize | `KERNEL_READY` |
| 2 | Require completed feature manifest | `ROUND06_FEATURE_GATE_PASSED` |
| 3 | Replay six controls, then at most 12 challengers | `ROUND06_SCREEN_COMPLETED` |
| 4 | Independently recalculate saved metrics; make report | `ROUND06_REPORT_READY` |
| 5 | Summarize recorded result | `ROUND06_EXECUTION_COMPLETE` |

Screen cap: 240 seconds useful work, 260 seconds outer. Report cap: 60 seconds. Stop on any failure; do not move to later cells. No algorithm changes, hyperparameter search or ensembling are run.

After success, open `03_saved_results.ipynb` and choose Run → Run All Cells. This notebook only displays nine saved Plotly charts and bundles evidence; it does not retrain. Expect `ROUND06_NOTEBOOK_REVIEW_COMPLETE`. The self-contained HTML is `outputs/round06_report.html`.

## 5. Save and return the correct ZIP

Save all notebooks with Ctrl+S. In a terminal, run:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round06/launch.py" bundle
```

This is also the command to use after a failed/paused stage. It does not require the ML environment or a completed experiment and does not retry anything.

In the file browser, open `otto_feature_round06`, right-click `otto_round06_return.zip`, choose Download, and attach that ZIP to ChatGPT. Do not upload the old Round05 ZIP again. The final bundle command is needed after saving notebooks so it includes their current outputs.

Then open Studio → Running instances → find **otto-dev / JupyterLab** → **Stop**. Stop the application, not just its kernel. **Do not delete the persistent space.** Files remain; storage charges can remain. The scripts do not stop the app for you.

## Terminal alternative — use instead of, not alongside, the execution notebooks

Run each line separately and inspect success before the next. Do NOT paste the whole block as an unattended sequence. Do not use this alternative after the corresponding notebook stages completed.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round06/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round06/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round06/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round06/launch.py" report
```

Then use the saved-results notebook and Step 5. There is deliberately no `all` command that launches everything without inspection.

## What constitutes progress

Primary pooled gain ≥0.003, nonnegative on both folds, and no pooled order decline proposes **separate frozen confirmation**, not retention or promotion. A loss freezes another negative result; diagnostic plots guide the next genuinely different hypothesis. This cohort has been repeatedly inspected, so no internal score proves the historical 0.60503 private-leaderboard target has been beaten.

Preparation did not launch AWS jobs, modify cloud resources, push GitHub, or submit predictions. The full AWS Polars/private-data integration was not replayed locally. Tiny synthetic native LightGBM tests exercise the new runner, but do not establish actual feature value. The real SageMaker preflight verifies existing inputs and controls before fitting.

See `RESEARCH_PLAN.md`, `feature_catalog.json`, `protocol.json`, `LOCAL_VALIDATION.json`, and `evidence/ROUND05_AUDIT.json` for formulas, provenance, limitations and tests.

AWS stop guidance: https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-running-stop.html
