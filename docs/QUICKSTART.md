# Reproduce the retained public reference

## Public review · Python 3.11+

```bash
python3 -S review/run.py --output ../otto-review
python3 -S review/run.py --output ../otto-review --check
python3 -S scripts/review_portfolio.py --json
```

No cloud account, API key or private data is needed. Open `../otto-review/index.html` for the authored session walkthrough and independent recommendation diagnostics. Add `--with-node` to exercise the browser handlers with Node 22 or 24. These are synthetic component checks, not competition scores.

## Native training · locked Python 3.13 environment

```bash
uv sync --frozen --extra dev --extra ml
uv run --frozen --extra dev --extra ml python reproduce.py train --output artifacts/reference
uv run --frozen --extra dev --extra ml python reproduce.py train --output artifacts/reference
uv run --frozen --extra dev --extra ml python reproduce.py verify --output artifacts/reference
```

The pipeline validates JSONL, converts events, creates four chronological roles, builds historical retrieval, generates features, fits objective-specific LightGBM models, seals model choices and validates predictions. A second identical run verifies and reuses its native artifacts. Changing data, configuration, code or runtime requires a new output directory.

## Full reference research

Obtain the OTTO training data through the original provider with the necessary rights. The first command only prints a resource plan:

```bash
uv run --frozen --extra ml python reproduce.py research --train-jsonl /path/to/train.jsonl --output artifacts/full-reference
```

After reviewing the plan, add `--execute` and appropriate `--threads`, `--workers`, `--memory-gib` and `--max-seconds`. Defaults are intended for the small profile; full data requires substantially more resources. Execution makes no submissions or data downloads. Termination stops the managed child process tree, but **does not stop a cloud instance**.

## Tests and controlled notebook

```bash
uv run --frozen --extra dev --extra ml python scripts/run_quality_gate.py
python -S -m unittest discover -s review -p test_run.py
node --test tests/public_demo_ui.test.mjs
```

The retained notebook is `notebooks/09_controlled_feature_study.ipynb`. Its saved output is historical evidence; the notebook executor re-runs the aggregate analysis in the separately hashed analysis environment. CI checks both native reference execution and public review.

The public reference is **not** exact reproduction of the private neural-similarity release. Model artifacts, raw source populations and former AWS backups are not bundled. [Publication boundary](PUBLICATION_SCOPE.md).
