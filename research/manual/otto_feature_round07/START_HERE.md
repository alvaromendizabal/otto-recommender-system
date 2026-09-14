# OTTO Round 07 — manual execution

**Do not rerun Round 06. Keep its saved 134-feature control.** This round creates genuinely new historical-session features and measures support/candidate coverage before buying another ranker comparison. **Zero new model fits.**

## 1. Open the existing space

Open AWS, select US West (Oregon), then SageMaker AI → Studio → `QuickSetupDomain-20260902T115323` → profile `default-20260902T115323` → Open Studio → JupyterLab → `otto-dev`. This is the previously recorded workspace location, not a new cloud inventory claim. Keep the working instance configuration and existing environment. Start the existing space only when ready to use it. Do not create/delete a space or reinstall packages.

## 2. Upload two files into the home/top-level JupyterLab folder

Upload `otto_feature_round07.zip` and `install_round07.py`. Leave the ZIP zipped. Upload outside the repository and outside previous round folders.

Open File → New → Terminal:

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_round07.py"
```

Expected: `ROUND07_INSTALLED`. An exact existing installation yields `EXISTING_ROUND07_PRESERVED`; follow its `NEXT:` instruction. Changed source or a previous failed stage means stop and return evidence. Do not overwrite/delete a prior result. This installer does not start research, change the environment or modify cloud/Git resources.

## 3. Review the completed result

Open `otto_feature_round07/00_round06_review.ipynb`. It is already executed, self-contained, and contains eight charts based on the returned Round 06 evidence. No AWS experiment is needed to inspect it.

## 4. Build historical evidence, one code cell at a time

Open `01_build_neighbor_evidence.ipynb`. Keep the existing `OTTO - Notebook` kernel. If a kernel selector appears, choose that existing kernel. The notebook interface calls the already-installed project `.venv`; do not make a new environment.

Run code cells individually with Shift+Enter. **Do not Run All on execution notebooks.**

| Code cell | Action | Required success message |
|---|---|---|
| 1 | Initialize launcher | `KERNEL_READY` |
| 2 | Tests and tiny actual DuckDB/Parquet smoke | `ROUND07_TESTS_PASSED` |
| 3 | Select 256 early-training queries and verify inputs | `ROUND07_PILOT_READY` |
| 4 | Scan historical anchor-to-session postings | `ROUND07_POSTINGS_READY` |
| 5 | Retrieve events for the selected historical sessions | `ROUND07_HISTORY_READY` |
| 6 | Check completion | `ROUND07_HISTORY_READY` |

Save with Ctrl+S. The postings and history stages read the existing Parquet source, not raw JSON; they do not rebuild the old graphs.

## 5. Build features and audit them

Only after the history gate passes, open `02_features_and_audit.ipynb`.

| Code cell | Action | Required success message |
|---|---|---|
| 1 | Initialize | `KERNEL_READY` |
| 2 | Require historical evidence | `ROUND07_HISTORY_READY` |
| 3 | Build 12 primary features and 12 comparison features | `ROUND07_FEATURES_READY` |
| 4 | Audit censored training-target support, new candidates and redundancy; generate report | `ROUND07_AUDIT_READY` |
| 5 | Print the recorded decision | `ROUND07_AUDIT_READY` |

Save with Ctrl+S. There is deliberately no ranker-training cell in this milestone.

## 6. Inspect eight charts and collect the return file

After the audit succeeds, open `03_saved_results.ipynb`. **Run → Run All Cells is appropriate for this saved-results notebook only.** It reads saved figures and bundles evidence; it does not build features or train models. The HTML report is `outputs/round07_report.html`.

Save all notebooks with Ctrl+S, then use the terminal for one final bundle so their saved execution outputs are included:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" bundle
```

Download `otto_feature_round07/otto_round07_return.zip` from the JupyterLab file browser (right-click → Download). Attach that Round 07 ZIP to this chat. Do not attach Round 05 or Round 06 again.

## Stop rule — applies at every stage

After any red error, `PAUSED_CHECKPOINTED`, `PRIOR_STOP_DETECTED` or `STOPPED_REVIEW_REQUIRED`, stop. Do not run later stages, increase limits, reinstall anything or retry the same command. Run only:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" bundle
```

Return the ZIP, even when incomplete. Existing evidence remains saved. A bundle command never retries the experiment.

## Resource limits

| Stage | Outer process cap |
|---|---:|
| Tests | 60 seconds |
| Prepare pilot | 140 seconds |
| Historical postings | 140 seconds |
| Selected historical events | 140 seconds |
| Feature generation | 140 seconds |
| Audit + report | 110 seconds |

These are safety caps, not predicted runtimes. Useful-work caps are shorter. There are 15-second UTC heartbeats, 32-query feature checkpoints, a 26 GiB worker stop, DuckDB memory/spill caps of 6GB/3GB and a 10 GiB free-disk requirement. Historical sampling has explicit limits of 64 postings per anchor, 65,536 selected source sessions and 2,000,000 retained events. Exceeding a hard limit stops rather than silently trimming data.

## Terminal alternative — not an additional execution path

The notebook path above is recommended. For terminal-only execution, run each line individually, check its success message, and stop on the first failure. Never run these concurrently with notebook stages. Do not paste the entire block as an unattended chain.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" prepare
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" postings
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" histories
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" audit
/opt/conda/bin/python -u "$HOME/otto_feature_round07/launch.py" bundle
```

## Stop billing without deleting files

After the download, go to Studio → Running instances → `otto-dev` / JupyterLab → Stop (or the space's Stop action). Stop the running application, not merely the kernel or browser tab. **Do not delete the persistent space.**

## What the result can and cannot establish

The feature and candidate support measurements are on a fixed training-only pilot, with censored targets. A larger candidate oracle is not a trained model improvement. The original control's validation score remains 0.569231 until a separately specified matched model test proves otherwise. The recorded historical private-score target is 0.60503, but these internal diagnostics are not directly comparable to it. Feature engineering remains open; nothing is automatically retained or submitted.

If the family has useful support, the next bounded round will test frozen features on the full fitting cohort with unchanged ranker settings and saved-control replay. Candidate expansion must be evaluated separately, followed by distinct-cohort/time-window confirmation for promising results.
