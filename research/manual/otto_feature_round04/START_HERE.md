# OTTO Round 04 — manual, bounded, no reinstall

Keep the `otto-dev` space, repository, raw data and all prior experiment folders. Round03's new configurations lost to the matched control. Do not retrain or submit them.

## One upload, two execution notebooks

1. Download `otto_feature_round04.zip` to Windows Downloads; leave it zipped.
2. Open Oregon SageMaker AI -> Studio -> `QuickSetupDomain-20260902T115323` -> profile `default-20260902T115323` -> JupyterLab -> existing `otto-dev`. Open it; Run space only if it is stopped. Keep existing CPU/storage settings.
3. Upload the ZIP into the JupyterLab home folder, outside the repository and previous experiment folders.
4. Open a SageMaker JupyterLab Terminal and run:

```bash
cd "$HOME"
/opt/conda/bin/python -m zipfile -e "$HOME/otto_feature_round04.zip" "$HOME"
```

5. Open `otto_feature_round04/01_build_demand_index.ipynb`, select **OTTO - Notebook**, run the first cell with Shift+Enter. Expect `KERNEL_READY` and a number. If it hangs for 15 seconds, stop. Otherwise Run -> Run All Cells. It tests the code and builds one reusable index from the existing 11.31 GB TRAIN file. No download, graph building or experiment model fitting occurs in this notebook.
6. Continue only after `RESULT: ROUND04_INDEX_READY`. This new stage must scan the complete raw training source once; it is not a rerun of previous graph work. Expect UTC progress with raw bytes and committed chunks.

### Planned pause, failure, and spending boundary

A normal `PAUSED_CHECKPOINTED` saves completed chunks near the five-minute work budget. It is NOT a code failure. For the **index phase only**, one further manual invocation is included in this initial budget:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" index
```

This resumes, not restarts, the committed raw chunks. If that invocation also pauses, stop and return the ZIP. Do not increase limits or continue looping. If either invocation fails, times out, prints `STOPPED_REVIEW_REQUIRED`, or shows a different error, stop immediately; do not use the resume instruction for an error. Completed files remain intact.

7. Once `ROUND04_INDEX_READY` is confirmed, open `02_run_demand_comparison.ipynb`, select **OTTO - Notebook**, run the first cell, then Run All Cells. It checks index completion, builds only the two 75-column demand representations, replays the six existing controls, and fits up to twelve new challenger models. Expected final markers:

```text
RESULT: ROUND04_FEATURES_READY
RESULT: ROUND04_SCREEN_COMPLETED
RESULT: ROUND04_REPORT_READY
```

These stages have their own work budgets. Stop and return the ZIP for a pause or failure; no automatic retries.

8. Open `03_saved_results.ipynb`, select **OTTO - Notebook**, and Run All Cells. This reads saved results only and emits seven Plotly charts, then `ROUND04_NOTEBOOK_REVIEW_COMPLETE`. Use it for later viewing, not the execution notebooks.
9. Save notebooks. In the terminal, run:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" bundle
```

Right-click `otto_feature_round04/otto_round04_return.zip` -> Download. Attach that ZIP to ChatGPT. After a pause or failure the same bundle command collects evidence without resuming work. If it cannot create a ZIP, send the error text.
10. Studio -> Running instances -> `otto-dev` / JupyterLab -> **Stop**. Do not delete the space. Kernel shutdown and computation timeout do not stop the SageMaker application or all billing. Retained storage remains billable.

## Terminal alternative, one stage at a time

Do not run both terminal and execution notebook paths simultaneously.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" index
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" report
/opt/conda/bin/python -u "$HOME/otto_feature_round04/launch.py" bundle
```

Copy only one command at a time and inspect its result. The displayed block is not a shell chain which keeps going after an error.

## Limits and exact paths

Outer process limits: tests 60s; index 320s per invocation; features 320s; screen 260s; report 60s. Worker-memory stop at 26 GiB. Require 10 GiB available at phase entry. There are no package installs and no changes to the existing locked project environment.

Notebook interpreter: the existing **OTTO - Notebook** registration. ML interpreter invoked by launch.py: `$HOME/otto-recommender-system/.venv/bin/python`. Code requires the reviewed clean `main` commit `2638faa34afa427ed4ac4e92bba04deda57c688e`; a mismatch stops rather than resetting your work.

Raw input: `$HOME/otto-data/raw/train/otto-recsys-train.jsonl`. Prior controls/candidate chunks: `$HOME/otto_feature_round02/outputs/`. Source reference/helpers: `$HOME/otto_feature_round02/`. No test source is opened. New index/chunks/models: `$HOME/otto_feature_round04/outputs/`.

Source is a new manual package, not a pushed or merged GitHub change. It writes no AWS resources, credentials, policies, S3 objects or Git references. The raw source, existing models, graph files and repository are read-only. Back up persistent evidence before ever deleting the space.

## Reading the result

The PRIMARY is `demand_rolling` versus `control_shared`. `demand_static_ablation` uses the same formulas but freezes the snapshot at Aug16 22:00 UTC. A positive rolling-minus-static result does not rescue a failed rolling-minus-control result. No score is a Kaggle result. This screen does not create a submission and does not promote features.

The full definitions, source availability assumptions, exclusion rules, research citations and decision gate are in RESEARCH_PLAN.md. Stage-specific failure logs and immutable receipts remain available in outputs.
