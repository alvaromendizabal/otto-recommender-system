# Reproducibility

Source, compact evidence and executed notebooks are versioned in Git. Large event
partitions, feature caches, native checkpoints and prediction parts live in S3 under
content identities. A report records the exact model/data contract used for its result.

## Review and replay without AWS or the dataset

The Linux CI environments use Python **3.13.15** for the project and **3.12.13** for
analysis. `uv.lock` pins the project stack. `notebooks/requirements.in` records the
analysis dependencies; `notebooks/requirements.txt` locks every transitive dependency
and its allowed hashes.

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
.venv/bin/python scripts/project_status.py

uv venv /tmp/otto-analysis --python 3.12.13 --no-project
uv pip install --python /tmp/otto-analysis/bin/python --require-hashes -r notebooks/requirements.txt
# Needed for Plotly's static PNG fallback if Chrome is not already installed:
/tmp/otto-analysis/bin/plotly_get_chrome -y
/tmp/otto-analysis/bin/python scripts/execute_notebooks.py
/tmp/otto-analysis/bin/python scripts/execute_notebooks.py
```

The second execution verifies and reuses the completed notebooks. Each receipt records
source cells, evidence hashes, interpreter/dependency versions, execution time and output
SHA-256. Kernels run in isolated workspaces with local IPC sockets. If the host blocks
kernel/browser processes, use the GitHub workflow or the documented managed environment.
The notebook runner never substitutes fabricated outputs for a failed kernel.

Notebook 09 rebuilds its tables and interactive Plotly figures from committed research
reports, with static PNG representations for GitHub. Notebook 10's default mode loads
three actual native LightGBM models, scores real example candidate features, writes its
own CSV and checks exact agreement with the full prediction run. This small replay does
not rerun historical graph construction or prove full-dataset runtime by itself.

## Rebuild the README and case-study figures

The seven public figures use Plotly and the same locked analysis environment as the
notebooks. They read the verified research reports; they do not fit models or rerun
full inference. With the analysis environment above installed:

```bash
/tmp/otto-analysis/bin/python scripts/build_portfolio_figures.py
/tmp/otto-analysis/bin/python scripts/build_portfolio_figures.py --check
```

The builder writes SVG previews, Plotly JSON definitions, an interactive HTML report,
a ZIP download containing that report, and a checksum receipt under `reports/portfolio/`. The HTML report includes its Plotly library and
works offline; it does not need Python or cloud access.
`--check` verifies the evidence and generator identities, every output checksum,
exact chart values, the HTML report, and the ZIP contents. It fails on stale or modified artifacts.

The CI `portfolio` job uses the locked environment, verifies the committed report,
and generates fresh SVG and PNG previews as a downloadable `portfolio-figures` artifact.
The PNGs are review outputs; SVG and HTML are the published presentation formats.
Generated reports are excluded from GitHub's source-language statistics through
`.gitattributes`. Notebook and Python source remain ordinary source files.

## Rebuild the controlled study

Obtain the official OTTO data and run the repository's streaming conversion workflow.
The research entry point expects `part-*.parquet` files with `session`, `aid`, `ts`,
`event_type` and `event_index`, plus the original conversion manifest. It hashes all
source partitions before building chronological queries.

A full run needs substantial memory and disk for history, feature caches and temporary
aggregation. The managed profile used one **ml.c7i.16xlarge**, 100 GiB disk, 16 feature
workers and 32 model threads. The commands below specify a 64 GiB aggregation limit;
use an appropriately sized machine. Smaller resource limits can spill heavily to disk.
No fixed wall-clock duration is promised across hardware.

```bash
.venv/bin/python scripts/run_research.py --stage prepare --source data/processed/train --threads 16 --memory-gib 64
.venv/bin/python scripts/run_research.py --stage retrieval --threads 16 --memory-gib 64
.venv/bin/python scripts/run_research.py --stage features --threads 16
.venv/bin/python scripts/run_research.py --stage screen --threads 16
.venv/bin/python scripts/run_research.py --stage model_features --role fit --workers 16
.venv/bin/python scripts/run_research.py --stage model_features --role selection --workers 16
.venv/bin/python scripts/run_research.py --stage ablate --threads 32
.venv/bin/python scripts/run_research.py --stage evaluate --threads 32 --workers 16
.venv/bin/python scripts/audit_research.py --source data/processed/train --threads 16 --memory-gib 64
```

The protocol is in [configs/research.toml](../configs/research.toml). The declared feature
lengths, weights, historical windows and candidate budgets describe a fixed schema;
unsupported changes fail explicitly. A new schema requires corresponding feature code
and fresh artifacts. Changing a protocol, learned history, selected features, model
weights or fingerprinted implementation cannot silently reuse an incompatible cache.

Fitting uses sampled negatives; selection and evaluation retain complete candidate
pools and target denominators. The model seal is written before evaluation access.
Changing a selected model after evaluation requires a new experiment, not editing the
seal. Paired intervals describe one frozen model and cohort, not repeated training runs.

## Reconstruct published evidence

`scripts/audit_research.py` independently rebuilds all observed prefixes and future
targets from the original Parquet event partitions, checks both directions of the
comparison, pools the official metric counts, verifies all 24 native models and replays
a deterministic sample with separate ranking/set arithmetic. Its source scope explicitly
excludes independently replaying the original raw JSONL conversion.

After the managed interpretation job has produced `artifacts/research/interpretation/`:

```bash
.venv/bin/python scripts/publish_research_report.py
```

This command requires a matching successful audit and interpretation, verifies their
model and report identities, and produces the compact `reports/research/` evidence.
The feature catalog includes every formula and rejection reason. The publication
manifest records all evidence-file hashes; it does not contain credentials or signed URLs.

## Full competition prediction through the notebook

Follow [INFERENCE.md](INFERENCE.md). Full mode requires the verified test partitions,
selected model artifacts and a refreshed historical graph. It runs `run_inference.py`
from Notebook 10 and writes a deterministic gzip CSV, validates all rows, and records a
separate full-notebook execution receipt. The managed delivery job stages existing
training/test data directly from the owned S3 bucket; only public code and small input
manifests need to be uploaded for a new launch.

The batch run and compact replay have distinct identities. Research evaluation uses its
original historical cutoff. Full prediction uses refreshed training history with frozen
ranking weights. Format validation does not imply Kaggle upload, acceptance or a score.

## Recovery and publication

Processing jobs continue without an attached terminal. Native models and partition
receipts are uploaded during computation. Recovery checks expected bucket ownership,
path containment, input identities and file SHA-256 before reuse. Atomic data writes
precede receipts; workspace locks reject duplicate writers. Tests exercise corrupt
checkpoints, missing parts, interruption and exact reuse on real small fitted models.

On `results/` branches, CI first passes the project, neural and notebook jobs. A restricted
publisher builds and verifies the portfolio figures, then executes all notebooks against
that exact published evidence. It stages only the canonical notebook outputs,
`notebooks/execution.json`, and the explicitly allowed files under `reports/portfolio/`. It refuses to overwrite a concurrently advanced branch.
The final results PR must pass checks at its published head before merging to `main`.
