# Clone → execute → verify

Three reproduction paths serve different questions. Start with the public review; use the native profile to inspect real training code; use locally obtained data for a full reference rebuild. None requires the author's cloud account, credentials, bucket or workspace.

| Path | Inputs | What runs | What it proves |
|---|---|---|---|
| **Review** | Repository; Python 3.11+ | Recorded-evidence audit and transparent interactive demo | Arithmetic, artifact consistency and a working synthetic recommendation example |
| **Native training** | Repository; locked Python 3.13 ML environment | Raw conversion → temporal corpus → historical retrieval → features → native rankers → sealed evaluation → CSV checks | The public research components connect end to end on invented records |
| **Full reference research** | Same environment; official training JSONL obtained independently | Original published protocol, feature screening, ablations and evaluation | A new execution of the public reference; agreement with a historical score must be measured, not assumed |

The native fixture is intentionally small. It has 512 invented sessions spanning four time-separated roles and includes unseen evaluation targets. It uses a predeclared feature subset and short fits, not the historical 102-feature study configuration. The reference engine is real; the data are synthetic.

## 1. Review without installation

```bash
git clone https://github.com/alvaromendizabal/otto-recommender-system.git
cd otto-recommender-system
python3 -S reproduce.py review --output artifacts/review --max-seconds 120
```

Open **artifacts/review/demo/report.html**. It runs offline and shows observed events, recommendations and separate evaluation targets. The console ends with **OTTO_REPRODUCTION_PASSED**. The existing [hosted walkthrough](https://alvaro-otto-session-lab.tartmacaw2.chatgpt.site) is an alternative, not a prerequisite.

## 2. Train the actual public reference components

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git. On Linux, macOS or Windows with WSL, use the checked-in project lock instead of mixing global dependencies. Linux is the CI-tested platform.

```bash
uv sync --frozen --extra dev --extra ml
uv run --frozen --extra dev --extra ml python reproduce.py train --output artifacts/reference --threads 2 --workers 2 --memory-gib 2 --max-seconds 600
uv run --frozen --extra dev --extra ml python reproduce.py train --output artifacts/reference --threads 2 --workers 2 --memory-gib 2 --max-seconds 600
uv run --frozen --extra dev --extra ml python reproduce.py verify --output artifacts/reference
```

The first run builds the full chain from raw JSONL. The second requires the same source, data, versions and configuration, then verifies and reuses completed artifacts. Verification fails on modified or missing sealed outputs. A source/configuration change requires a new output directory; existing evidence is not overwritten to disguise incompatibility.

### Outputs worth opening

| Output | Review question |
|---|---|
| **summary.json** | Which profile, versions, feature set, models and population produced this result? |
| **corpus/manifest.json** | What are the chronological boundaries and source identities? |
| **models/** | Which actual native LightGBM models and matched variants were fitted? |
| **evaluation/report.json** | What did the sealed models achieve on the complete evaluation cohort? |
| **recommendations.csv** | Are there three task rows per evaluation prefix with 20 distinct items each? |
| **reproduction.json** | Do all published run artifacts still match their SHA-256 identities? |

The CSV is for **evaluation prefixes**, not a competition upload. The runner never contacts Kaggle or submits a file. Positive, negative and unseen targets remain in the original metric denominators. Model selection occurs before evaluation targets are opened.

## Full reference research

Obtain the official **OTTO training JSONL** from the dataset publisher or competition under its access terms. See the [publisher repository](https://github.com/otto-de/recsys-dataset) and [competition input contract](../src/otto_recsys/research/competition_input.py). Do not use a post-competition full-test file as an observed test prefix. Data are not redistributed here.

Keep raw data immutable outside the output directory. The schema is one session per JSON line: session ID and events containing aid, millisecond ts, and clicks/carts/orders type. The built-in validator checks the actual schema before conversion.

Start with the plan:

```bash
uv run --frozen --extra dev --extra ml python reproduce.py research --train-jsonl /path/to/train.jsonl --output artifacts/full-reference --threads 16 --workers 16 --memory-gib 64 --max-seconds 3600
```

This does **not** start training. Review the printed resources. Add **--execute** to run that same command. The original chronological, feature, retrieval and training definitions come from [configs/research.toml](../configs/research.toml). A full-data run is substantially larger than the fixture; plan adequate memory and disk. The memory option constrains DuckDB aggregation, **not the total resident memory of all workers**. The supervisor's wall-time bound terminates the process group and leaves committed stage artifacts available for a compatible restart. It does not stop a cloud instance or cap a cloud bill.

Full-data historical-score agreement is not asserted until a new complete run is compared under the same protocol. Exact byte replay of the later neural-enhanced release is outside this entry point. Follow [the detailed research guide](REPRODUCIBILITY.md) for stage-level controls, native historical replay, notebook execution and official-prefix inference prerequisites.

## Troubleshooting without changing the experiment

**Dependency import failure:** use the frozen ML environment; the dependency-free interpreter is only for review and verification. Do not install arbitrary versions into an existing research environment.

**Different contract:** choose a new output directory. Do not delete a receipt or replace expected hashes to force reuse.

**Deadline stop:** inspect the last completed stage, retain the output directory, and rerun the identical contract with an appropriate wall-time allowance. Raw-input hashing and validation may repeat; completed native chunks and models are reused under their existing contracts.

**Insufficient resources:** reduce concurrency before starting a new contract or use a suitable machine. This project does not create resources automatically.

**No historical score match:** check the data version, chronological protocol, selected features and runtime. Fixture metrics are not historical performance evidence.

## Safety and provenance

The entry point performs no cloud API calls, package installation, dataset downloads or publication. Lockfile, runtime, source and raw-data identities are recorded. Only voluntarily supplied input paths are read. Keep private datasets and generated full-data outputs untracked. Current public cloud metadata uses placeholders; see [the redaction record](../reports/privacy/redactions.json). Git history has not been rewritten.
