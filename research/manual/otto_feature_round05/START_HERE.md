# OTTO Round 05 — one bounded feature investigation

Do not rerun Round04: both demand arms lost to its saved control. Keep the 169-chunk demand index, original graphs, raw data, all earlier experiment folders and the existing ML environment intact.

## 1. Open the existing workspace

Open Oregon SageMaker AI at https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2 . Choose Studio -> QuickSetupDomain-20260902T115323 -> profile default-20260902T115323 -> JupyterLab -> otto-dev. Open JupyterLab; Run space only if its app is stopped. Keep current CPU/storage. No new space, CloudShell, install, Git reset, Git pull or kernel setup is needed.

## 2. Upload one ZIP

Download otto_feature_round05.zip to Windows Downloads and leave it zipped. Upload to the JupyterLab home/top-level folder, outside the repository and prior round folders. In a SageMaker JupyterLab terminal:

```bash
cd "$HOME"
/opt/conda/bin/python -m zipfile -e "$HOME/otto_feature_round05.zip" "$HOME"
```

Expect a new otto_feature_round05 directory. Leave otto_feature_round02/outputs intact: it supplies verified control features, candidates and six models.

## 3. Build a NEW historical transition index, one stage at a time

Open otto_feature_round05/01_build_transition_index.ipynb. Select **OTTO - Notebook**. Run the first cell with Shift+Enter; expect KERNEL_READY and an execution number. Stop if that simple cell remains busy for 15 seconds.

Run the next code cell, stage('tests'), and wait for ROUND05_TESTS_PASSED. The tests include a mandatory synthetic integration check using the ALREADY installed DuckDB and Parquet backend. This real-backend smoke was not executable in the preparation environment; it must pass in your AWS environment before real-data indexing. No installation is performed.

Then run stage('index'). It hashes the existing historical Parquet tail and saves up to 16 source-partition count tables. It does not scan raw JSON or rebuild your existing graphs. Expect UTC heartbeats and transition_partition_committed counters. Continue only after RESULT: ROUND05_INDEX_READY. Run the final summary cell and save the notebook.

One index invocation is the initial work allowance: 300s useful work / 320s outer process cap. If it pauses, times out or fails, stop and return evidence before any additional invocation. A planned pause preserves committed partitions; do not automatically retry or raise limits.

## 4. Run the matched feature comparison

Only after the index is ready, open 02_run_action_pair_comparison.ipynb with OTTO - Notebook. Run cells one at a time:

- First cell: KERNEL_READY.
- Gate cell: INDEX_GATE_PASSED.
- stage('features'): RESULT: ROUND05_FEATURES_READY (320s outer cap).
- stage('screen'): RESULT: ROUND05_SCREEN_COMPLETED (260s outer cap).
- stage('report'): RESULT: ROUND05_REPORT_READY (60s cap).
- Final cell: ROUND05_EXECUTION_COMPLETE and RETURN_FILE.

The primary adds 27 action-specific features; the collapsed ablation adds nine. All six old controls must replay before the twelve maximum new challenger fits. Feature chunks contain 64 sessions each. The first 16 receive deterministic replay. No previous model is refitted, and failed earlier additions are excluded.

## 5. Read seven Plotly charts without restarting training

Open 03_saved_results.ipynb, select OTTO - Notebook, and Run -> Run All Cells. It reads saved results only. Expect CHART_1_PAYLOAD_SENT through CHART_7_PAYLOAD_SENT and ROUND05_NOTEBOOK_REVIEW_COMPLETE. Subsequent chart viewing uses this notebook, not the execution notebooks. The optional 00_experiment_design.ipynb already contains three executed plots and the research rationale. Its structural illustration is not an experiment score.

## 6. Save and return the small result ZIP

Save all notebooks, then run in the SageMaker terminal:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" bundle
```

Right-click otto_feature_round05/otto_round05_return.zip -> Download. Attach it to ChatGPT. Bundle is report-only: it does not start computation. It is also the correct command after a failure or pause. If no ZIP is created, return the error text.

Then Studio -> Running instances -> otto-dev / JupyterLab -> **Stop**. Stop the application, not just the kernel. DO NOT delete the space. Source and model files persist when the app stops; retained storage can still incur charges.

## Error handling

Any red cell, STOPPED_REVIEW_REQUIRED, PAUSED_CHECKPOINTED, memory stop, backend-smoke error, or timeout means stop before the next stage. Collect the ZIP. Do not increase memory/time settings, reinstall packages, delete checkpoints or keep rerunning. Earlier completed data remains preserved. Never run terminal and notebook execution simultaneously.

A planned pause currently appears through a RuntimeError so that queued notebook cells cannot continue. Its explicit PAUSED_CHECKPOINTED label identifies the planned boundary, not a model-quality result. We still require inspection before further compute.

## Terminal alternative — one command at a time

Use these instead of the execution notebooks, continuing only after success. Keep the saved-results notebook for plots.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" index
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round05/launch.py" report
```

Do not paste all five as an unattended batch. No cloud launches, automatic S3 upload or Git changes occur. New source is a manual attachment, NOT pushed/merged. The script requires existing reviewed main 2638faa34afa427ed4ac4e92bba04deda57c688e and stops rather than discarding a changed checkout.

## Limits and interpretation

Tests 60s; index/features 320s outer each; screen 260s; report 60s. Worker memory 26 GiB; 10 GiB free disk at entry. DuckDB 6GB memory and 3GB spill maximum. These caps do not stop the SageMaker app or guarantee that the full phase completes in one invocation.

Reused 4096 fitting sessions, 400 candidates, unchanged two chronological folds, six-hour embargo and censored training targets. Scores are internal fitting diagnostics, not Kaggle results. A primary gain >=.003, both folds nonnegative and no pooled order loss warrants separate confirmation only. Beating the ablation while losing to control is still a failed primary. No submission is created. The 0.60503 private target remains unmet.
