# Round 11 — start here

Read RESEARCH_PLAN.md and DEPENDENCIES.md. All project execution belongs to you. This package is prepared and statically reviewed, not runtime-tested by the assistant.

1. Open existing AWS Oregon Studio / otto-dev; retain the OTTO - Notebook kernel and existing environment.
2. Open `00_review_and_design.ipynb` for the prior result. No new project score is shown.
3. Open `01_build_representations.ipynb`. Run its six code cells ONE AT A TIME: initialize; tests; prepare; inspect prepared inputs; learn representations; inspect spectra. Stop after any failure. Save.
4. Open `02_features_and_comparison.ipynb`. Run one cell at a time: initialize; representation gate; features; feature gate+screen; report; decision. Save.
5. Open `03_saved_results.ipynb`; Run All (read-only charts); save.
6. Optional `04_feature_examples.ipynb` uses tiny explicit synthetic vectors only and is not an experiment.
7. Save ALL notebooks. In a terminal run:

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" bundle
```

Return `otto_feature_round11/otto_round11_return.zip`. Large source matrices, factors and models remain private in AWS; the return contains receipts, logs, metrics and notebooks. Do not upload it publicly.

## Terminal alternative — instead of notebook execution, not concurrently

Run one command, inspect completion, then the next. Never paste this block as an unattended batch. The notebook path is preferred for inline plots.

```bash
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" tests
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" prepare
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" representations
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" features
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" screen
/opt/conda/bin/python -u "$HOME/otto_feature_round11/launch.py" report
```

Success markers: ROUND11_TESTS_PASSED, ROUND11_REPRESENTATION_INPUTS_READY, ROUND11_REPRESENTATIONS_READY, ROUND11_FEATURES_READY, ROUND11_SCREEN_COMPLETED, ROUND11_REPORT_READY.

PAUSED_CHECKPOINTED / PRIOR_STOP_DETECTED / any error: stop both rounds, save notebooks and run ONLY bundle. No retries or increased limits. A completed negative R11 is allowed before R12; an operational failure is not. Stop the SageMaker APPLICATION after download, not just the kernel. Preserve the space.
