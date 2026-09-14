# Round 09 — run this milestone manually

## Directed historical-session continuation

Both round folders are installed by the single `install_rounds08_09.py` supplied alongside `otto_feature_rounds08_09.zip`. Upload both files to SageMaker home, not inside the Git repository. Leave the ZIP zipped. In a JupyterLab terminal:

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_rounds08_09.py"
```

Expect `ROUNDS08_09_INSTALLED` or `EXISTING_ROUNDS08_09_PRESERVED`. The installer validates both destinations before installing either; it preserves outputs and allows notebooks with identical source but saved execution outputs. It never overwrites changed code. A prior failure means collect evidence, not retry. No Python packages are installed by this command.

## One round at a time

Start this round only after Round08 completes operationally. A valid negative Round08 score is not a block. Any failure or pause in either round is a block: bundle evidence rather than continue.

Keep the existing OTTO - Notebook kernel. The launch module explicitly uses `~/otto-recommender-system/.venv/bin/python`. Do not install another kernel or alter the environment. Keep the original Round02, Round05, Round06 and Round07 folders, raw data, indexes and model checkpoints. Their contracts are verified read-only.

Open `00_round07_review.ipynb` for the executed pilot review. `04_feature_examples.ipynb` contains executed synthetic feature examples, not model performance. Neither is required to run the experiment.

## Notebook execution — recommended path

Open `01_build_features.ipynb`. Run each numbered code cell with **Shift+Enter**, inspect its output, then proceed. Do not use Run All in this notebook. The first cell prints `KERNEL_READY`; the next runs the tests. The subsequent cells require the completed Round08 history, then generate features. The final cell requires `ROUND09_FEATURES_READY`. Save with Ctrl+S.

Open `02_run_comparison.ipynb`. Run one code cell at a time. The gate requires the completed features; the screen replays six original controls before fitting up to twelve challengers. Report generation recalculates statistics and writes nine Plotly payloads. The last cell prints the exact recorded comparison and requires `ROUND09_REPORT_READY`. Save with Ctrl+S.

Open `03_saved_results.ipynb` only after the report succeeds. Run All is safe here: it displays saved charts without training or extraction. Save all notebooks and execute the bundle command below in a terminal. This last command includes the saved execution outputs rather than the earlier autosaved notebook bytes.

| Stage | Required success message | Outer process cap |
|---|---|---:|
| `tests` | `ROUND09_TESTS_PASSED` | 90 seconds |
| `features` | `ROUND09_FEATURES_READY` | 260 seconds |
| `screen` | `ROUND09_SCREEN_COMPLETED` | 260 seconds |
| `report` | `ROUND09_REPORT_READY` | 90 seconds |

The cap is a safety boundary, not an expected runtime. Read the high-level UTC heartbeat and committed-query/model counters in the cell output. A screen can legitimately produce a negative score; `SCREEN_COMPLETED` records operational success, not promotion.

## Stop rule

Any red error, `PAUSED_CHECKPOINTED`, `PRIOR_STOP_DETECTED`, or `STOPPED_REVIEW_REQUIRED` means stop both rounds. Do not execute later stages, change source/limits, install anything, clear errors by rebuilding, or repeat the same failed command. Preserve existing files and run only bundle/report-reading commands. Never run a notebook stage and a terminal stage concurrently. The shared execution lock guards against this.

## Save and return

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" bundle
```

Download `~/otto_feature_round09/otto_round09_return.zip`. After both complete, return both ZIPs. On a failure in Round08, return its ZIP immediately rather than running Round09. On a failure in Round09, return both rounds' ZIPs. Bundling does not retry an experiment, fit a model or include the large private binaries.

In Studio, choose **Running instances**, find **otto-dev / JupyterLab**, and choose **Stop**. Do not delete the space. Stopping the application preserves its files; closing the browser or notebook kernel is insufficient to stop instance billing. Persistent storage charges can remain even with the application stopped.

## Terminal alternative — use instead of notebook execution, not alongside it

Run one command below, check the printed success marker, and only then copy the next command. No shell loop, `all` command or chained training script is provided.

### 1. tests

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" tests
```

Continue only after `ROUND09_TESTS_PASSED`.

### 2. features

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" features
```

Continue only after `ROUND09_FEATURES_READY`.

### 3. screen

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" screen
```

Continue only after `ROUND09_SCREEN_COMPLETED`.

### 4. report

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round09/launch.py" report
```

Continue only after `ROUND09_REPORT_READY`.

After the final command, open the saved-results notebook, save it, and run `bundle`. Terminal execution does not populate the execution counters of the training notebooks; its logs and receipts remain the authoritative run record.

## What success can and cannot establish

Each primary must beat the original 134-feature control by at least 0.003 pooled weighted Recall@20, be nonnegative on both folds and avoid a pooled order-recall loss to earn separate confirmation. No automatic promotion, combined feature set, GitHub write, AWS API mutation or Kaggle submission occurs. These are exploratory comparisons on a reused fitting cohort, not independent proof of beating the historical private leaderboard.
