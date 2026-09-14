# Existing environment only

Use `~/otto-recommender-system/.venv/bin/python`, invoked by launch.py through the existing notebook kernel. NumPy, SciPy, scikit-learn, LightGBM, Polars, Plotly and DuckDB must already be installed. `run_tests.py` prints versions and stops on a missing package. The historical source adapter also checks the original environment/data contracts. No package versions are silently changed and no installations are performed. Missing dependencies are a gate to report, not an instruction to pip-install latest.

Four numerical threads; 26 GiB worker RSS ceiling; 10 GiB free disk. Existing instance size is not verified by the assistant. Keep the current instance; do not upgrade blindly. No cost estimate is implied by timeout caps; use the actual instance rate shown in your AWS console. Stop the SageMaker application after evidence is downloaded; closing a tab or notebook kernel is insufficient.
