# OTTO feature Round 03 — run manually in the existing workspace

Round 02 completed but the timing additions did not replicate. **Do not rerun Round 02, replace the kernel, wipe the space or download the data again.** This is a new 36-feature experiment with a saved matched control and one encoding ablation. The failed 24 timing features are excluded.

## 1. Download and open your existing workspace

Save `otto_feature_round03.zip` in Downloads; leave it zipped.

Open the Oregon console: https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2

Studio → domain `QuickSetupDomain-20260902T115323` → profile `default-20260902T115323` → JupyterLab → `otto-dev` → Open JupyterLab. Run that existing space first only if it is stopped. Keep its existing CPU and storage settings. No CloudShell or new space is required.

## 2. Upload to the home folder and extract

In the JupyterLab file browser, navigate to your home/top-level folder (not the repository, Round 01 or Round 02). Upload `otto_feature_round03.zip`. Open Terminal and run:

```bash
cd "$HOME"
/opt/conda/bin/python -m zipfile -e "$HOME/otto_feature_round03.zip" "$HOME"
```

Leave `otto_feature_round02`, its `outputs/features` and `outputs/models` folders intact. They provide the reused candidate arrays, controls and prior evidence. Round 03 does not overwrite them.

## 3. Run the new experiment notebook

Open `otto_feature_round03/01_run_recent_mass.ipynb` and select **OTTO - Notebook**. Run the first cell with Shift+Enter. It should print `KERNEL_READY` and receive an execution number promptly. If it remains `[*]` for more than 15 seconds, stop before any experiment launch.

Then Run → Run All Cells. Cells run tests → features → screen → report; if any stage fails, later queued stages are guarded against running. Computation uses the existing project `.venv`; the named notebook kernel only delegates and shows saved results. No installation is performed.

Expected successful markers:

```
RESULT: ROUND03_FEATURES_READY
RESULT: ROUND03_SCREEN_COMPLETED
RESULT: ROUND03_REPORT_READY
```

The exact 36 new features are built on saved candidates. The first 16 sessions are numerically replayed. Screening reloads six existing control models and proves exact per-session hits before fitting the two new challengers (12 models maximum). All stage work is bounded and progress messages appear every 15 seconds. A test stage may fit a tiny synthetic model for reload testing; that is separate from the 12 experimental models.

## 4. Errors and pauses

On a red cell, `STOPPED_REVIEW_REQUIRED`, or `PAUSED_CHECKPOINTED`, stop and return the available bundle. No repeated retries, increased limits, dependency changes or Git resets. Completed chunks/models are preserved. To collect without running training:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round03/launch.py" bundle
```

If no ZIP is created, return the terminal/cell error.

## 5. Review seven saved Plotly charts, with no training

After success open `otto_feature_round03/02_saved_results.ipynb`, select **OTTO - Notebook**, and Run All Cells. Nine code cells display the audited result and seven explicit Plotly MIME payloads. It must finish with `ROUND03_NOTEBOOK_REVIEW_COMPLETE`.

Use this notebook for repeated viewing, not the execution notebook. An offline HTML report is also produced at `outputs/round03_report.html`.

## 6. Save, bundle, download, stop compute

Save both notebooks. Run the report-only bundle command above once to capture their saved bytes. Right-click `otto_feature_round03/otto_round03_return.zip` → Download. Attach that ZIP to ChatGPT.

Studio → Running instances → `otto-dev` / JupyterLab → Stop. **Do not delete the space.** Notebook kernel shutdown and computation timeouts do not stop the chargeable app. Files remain on the space; retained storage can remain billable.

## Terminal alternative

Use either the notebook or these commands, not both simultaneously. Run one at a time and continue only after success:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round03/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round03/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round03/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round03/launch.py" report
```

## Bounds, source and publication status

Tests 60 seconds; features 320 seconds outer cap (300 internal); screen 260 seconds outer cap (240 internal); reporting 60 seconds. Max worker RSS 26 GiB, 4 GiB free-disk gate. No paid service is created, stopped or changed by the toolkit. Stop the existing app yourself afterward.

The toolkit remains outside Git and does not commit, push, modify the project environment, upload to S3, or submit to Kaggle. The reviewed repository remains `2638faa`. The return bundle is small and excludes the native models and full feature chunks; those remain on your persistent workspace. Do not mistake the bundle for a complete off-site backup of all new experiments.

No primary feature, model or source is promoted from this screen. `RESEARCH_PLAN.md` contains exact formulas, leakage assumptions, negative findings, decision gates and next information families.
