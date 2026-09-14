# Round 10 — manual execution

Start with `TWO_ROUND_GUIDE.md`, which covers installation and both rounds. Pending Round 09 comes first. Round 08 completed and must not be rebuilt. The Round 09 files in this delivery are unchanged from the earlier package; use the new guide instead of the older installation prose.

Download/upload `otto_feature_rounds09_10.zip` and `install_rounds09_10.py` to SageMaker home. Run:

```bash
cd "$HOME"
/opt/conda/bin/python "$HOME/install_rounds09_10.py"
```

After Round 09 has `ROUND09_REPORT_READY`, open `01_build_features.ipynb` here. Run each numbered code cell with Shift+Enter and require the success marker before proceeding. The five cells initialize, test, check prerequisites, generate features, and verify completion. Save with Ctrl+S.

Next open `02_run_comparison.ipynb`; initialize, require features, screen, report and inspect the recorded decision, one cell at a time. Require `ROUND10_SCREEN_COMPLETED` then `ROUND10_REPORT_READY`. Save. Open `03_saved_results.ipynb`; Run All is safe here because it only reads the saved nine charts. Save all notebooks.

On success or failure, collect evidence with:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round10/launch.py" bundle
```

Download `otto_feature_round10/otto_round10_return.zip`. Also return the pending Round 09 result once completed. A red error or pause means stop both rounds and bundle; never raise limits, reinstall, delete evidence, or repeat an unchanged failure. Tests/report have 90-second outer caps; feature/screen stages have 260-second outer caps. These are safety limits, not runtime estimates.

`00_round08_review.ipynb` is an executed uploaded-data review, not another run. `04_feature_examples.ipynb` is an executed synthetic illustration of all 24 primary and 24 ablation features, not a model score. `RESEARCH_PLAN.md` documents formulas, sources, limits and the advancement gate. `feature_catalog.json` gives each feature definition and source constraints.

Stop **the JupyterLab application** from Studio → Running instances when done. Do not delete the space, raw data, old models or prior rounds. No new history index, graph, candidate pool, source download, Git write or submission is needed.
