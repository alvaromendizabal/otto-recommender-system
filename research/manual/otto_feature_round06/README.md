# OTTO · Repeat-versus-discovery feature study

**One hypothesis. Two matched representations. No historical rebuild.**

Start with **START_HERE.md**. The executed `00_review_and_design.ipynb` reviews the actual Round05 negative result. Run `01_build_context_features.ipynb`, then `02_run_context_comparison.ipynb` one cell at a time. After success, `03_saved_results.ipynb` displays nine result charts without fitting.

All new work is in this folder. Existing data, model checkpoints, environment and Git checkout are read-only. No Kaggle score is produced. Unchanged failures are not retried.

`relative_features.py` defines the twelve repeat-conditioned and twelve global-context features. `experiment.py` verifies/replays existing controls and runs the matched screen. `report.py` recomputes metric evidence and writes Plotly HTML/JSON. `launch.py` enforces process limits and creates the return ZIP. `tests/` contains metric, feature, storage, native-model and fixture integration tests. `RESEARCH_PLAN.md` and `feature_catalog.json` contain the frozen scientific specification.

Manual package; not a GitHub commit or a claim of AWS synchronization.
