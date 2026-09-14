# OTTO / Historical-session evidence

**Round 07 • Feature engineering • Training-only feasibility • Zero model fits**

The 134-feature control remains the incumbent. Instead of another reparameterization of existing affinities, this milestone constructs action-specific evidence from individually similar historical sessions. A fixed 256-query pilot tests whether the source has support and can recover missing candidates before another ranker experiment.

Start with [START_HERE.md](START_HERE.md). Scientific definitions, information boundaries, source caps and next-decision criteria are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md).

| Notebook | Purpose |
|---|---|
| `00_round06_review.ipynb` | Executed eight-chart review of the actual completed result |
| `01_build_neighbor_evidence.ipynb` | Preflight, tests, frozen pilot and two bounded Parquet scans |
| `02_features_and_audit.ipynb` | Feature matrices, censored training-support audit, candidate oracles and report |
| `03_saved_results.ipynb` | Eight saved-result charts; no training |

`neighbors.py` defines the 12 primary and 12 last-anchor comparison features. `history_io.py` defines the bounded relational source extraction and actual-backend smoke. `metrics.py` keeps censored target support and candidate oracles separate from achieved ranking accuracy. `research.py` orchestrates checksummed stages; `launch.py` bounds time/memory and collects small return evidence. `core.py` and `intent_features.py` retain compatibility with the certified prior project helpers.

Local tests include independent formula oracles, exact extraction SQL in SQLite, source/candidate-order invariance, strict cutoff/exclusion checks, immutable artifact behavior and synthetic real feature/audit/report stages. Installed DuckDB/Parquet and private-data preflight remain mandatory in SageMaker. See `LOCAL_VALIDATION.json`; a synthetic test is never a new project score.

Existing repository/data/models/environment are read-only. No AWS API writes, Git commits/pushes, S3 uploads or Kaggle submissions are made. This package is not a cloud backup. Preserve the working folder and return `otto_round07_return.zip` after one bounded pass or the first stop.
