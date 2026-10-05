# OTTO · Session-based recommendation

**End-to-end recommender-system research at production scale:** multi-source retrieval, learning to rank, neural representations, temporal validation, resumable AWS inference, and reproducible experiment evidence.

[![CI](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml)

**Verified release:** **0.57586 private / 0.57601 public** on post-competition submission **56542128**. The released system combines the established click/cart routing policy with a promoted neural-similarity order ranker. No official medal, rank, online-service deployment, or business-lift claim is made.

## 60-second overview

| Area | Evidence |
| --- | --- |
| **Problem** | Rank the next products a shopper may click, add to cart, or order from an anonymous observed session prefix |
| **Scale** | **216.7M** training events · **1.67M** official inference sessions · **5.02M** validated output rows |
| **Retrieval** | Co-visitation channels, session revisits, popularity, neural and latent retrieval studies; up to **400 candidates/session** |
| **Ranking** | Task-specific LambdaRank plus neural-similarity, source-aware, sequence and heterogeneous-ensemble research |
| **Feature research** | **1,482** engineered formulas screened into a **102-feature** controlled reference schema; later specialized rankers extend that representation |
| **Validation** | Chronological roles, point-in-time feature contracts, OOF stacking, paired uncertainty, frozen promotion gates, deployment-parity checks |
| **Engineering** | AWS/SageMaker + S3, resumable batch jobs, immutable manifests, checkpoint reuse, native-model replay, resource/cost telemetry |
| **Quality** | Locked environments, pytest/ruff/mypy, GitHub Actions, executed notebooks, reproducible portfolio figures |
| **Verified result** | **0.57586 private / 0.57601 public** after the competition deadline |

## What I owned

I designed and implemented the project as an end-to-end ML system rather than a standalone model notebook:

- **recommender architecture:** candidate generation, feature computation, objective-specific ranking and routing;
- **validation design:** temporal fit/selection/evaluation roles, point-in-time correctness, leakage controls and promotion gates;
- **feature research:** large candidate/session feature catalog, screening, ablations, interpretation and error analysis;
- **model research:** LightGBM/XGBoost ranking, neural retrieval, learned similarity, sequence interaction and leakage-safe OOF ensembling;
- **cloud execution:** AWS-canonical training/inference, checkpointing, resume logic, deterministic artifact identities and bounded resource use;
- **delivery:** full-population inference, output validation, submission provenance and reproducible replay;
- **engineering quality:** CI, tests, executed notebooks, immutable evidence, failure regression tests and public/private publication boundaries.

## Choose your review depth

| Time | Start here | Best for |
| --- | --- | --- |
| **60 seconds** | [Employer overview](docs/EMPLOYER_OVERVIEW.md) | Recruiters, hiring managers, general ML leaders |
| **5 minutes** | [System architecture](docs/ARCHITECTURE.md) · [Research case study](docs/PORTFOLIO.md) | Senior ML engineers, data scientists, recommendation/search teams |
| **Deep dive** | [Controlled study](notebooks/09_controlled_feature_study.ipynb) · [Frontier research](research/frontier/README.md) · [Reproducibility](docs/REPRODUCIBILITY.md) | Applied scientists and technical interviewers |

## System architecture

~~~mermaid
flowchart LR
    A[216.7M historical events] --> H[Point-in-time history]
    P[Observed session prefix] --> C[Multi-source candidate retrieval]
    H --> C
    C --> K[Up to 400 candidates/session]
    P --> F[Candidate × session features]
    H --> F
    K --> F
    F --> R1[Click ranker]
    F --> R2[Cart ranker]
    F --> R3[Order ranker]
    P --> N[Neural / sequence representations]
    N --> R1
    N --> R2
    N --> R3
    R1 --> O[3 × top-20 recommendation lists]
    R2 --> O
    R3 --> O
    O --> B[Resumable batch inference]
    B --> V[5.02M validated rows + immutable provenance]
~~~

The metric is pooled **Weighted Recall@20 = 10% clicks + 30% carts + 60% orders**. Candidate coverage is treated as an oracle ceiling, never as achieved recommendation quality.

[Architecture and data contracts →](docs/ARCHITECTURE.md)

## Measured outcomes

### Controlled temporal study

On a **432,492-session reserved temporal cohort**, the selected controlled reference reaches:

| System | Weighted Recall@20 |
| --- | ---: |
| Candidate fusion | 0.535244 |
| Compact ranker | 0.564904 |
| **Selected ranker** | **0.584392** |

The selected representation improves on the compact ranker by **1.949 percentage points**, with a paired 95% session-bootstrap interval of **1.840 to 2.065 points**. These are offline results on matched temporal data, not competition scores.

### Verified full-population lineage

| Release | Private | Public |
| --- | ---: | ---: |
| Frozen 102-feature reference | 0.56842 | 0.56862 |
| Objective router | 0.57100 | 0.57121 |
| Long-session cart router | 0.57140 | 0.57166 |
| **Neural-similarity order release** | **0.57586** | **0.57601** |

Full inference covers **1,671,803 official sessions** and **5,015,409 recommendation rows**. The current verified release preserves the established click/cart policy and replaces orders with a task-specific ranker that combines the controlled representation with learned candidate-to-session similarity and neural query diagnostics.

[Inference and provenance →](docs/INFERENCE.md)

## Research depth

The repository preserves successful and unsuccessful hypotheses so model decisions remain inspectable. Major studies include:

- temporal and candidate-budget experiments;
- graph/co-visitation path features;
- matrix-factorization and Word2Vec retrieval;
- first-place-inspired neural sequence encoders;
- learned candidate-to-session similarity;
- source-aware ranking and XGBoost variants;
- candidate-conditioned sequence interaction;
- contextual reliability-aware models;
- heterogeneous OOF stacking;
- candidate-ceiling versus achieved-ranking diagnosis.

A recent deployment-parity audit found that one newer click/cart validation path had not reconstructed the exact deployed comparator. That challenger was returned to **research / comparator reconciliation** status rather than being presented as a release. The fitting evidence is preserved; the externally verified release remains unchanged.

[Validation-integrity case study →](research/frontier/11_validation_integrity_reconciliation.md)

## Engineering and reproducibility

The repository is intentionally more than a modeling report:

- **content-addressed inputs and artifacts** prevent silent cache reuse;
- **atomic receipts and immutable manifests** bind models, data, code and metrics;
- **resumable batch inference** reuses completed partitions after interruption;
- **native-model replay** verifies selected model behavior;
- **resource and runtime bounds** make expensive jobs inspectable;
- **GitHub Actions** runs quality, neural-contract, notebook and portfolio jobs;
- **executed notebooks** persist Plotly outputs and replay from checked evidence;
- **regression tests** encode previously encountered failure modes.

Public GitHub contains selected implementations, aggregate evidence, tests, notebooks and reproducibility contracts. Raw competition data, row-level labels/predictions, private runners, full checkpoints, embeddings, credentials and exact private orchestration remain outside the public repository.

## Technology

**Python · Polars · PyArrow · LightGBM · XGBoost · PyTorch · FAISS · scikit-learn · AWS SageMaker · S3 · Plotly · pytest · ruff · mypy · GitHub Actions**

## Reproduce the public review

~~~bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
.venv/bin/python scripts/project_status.py
~~~

The full research and inference workflows require the official OTTO data and larger private artifacts. The public review path is intentionally semi-reproducible: enough code, contracts, tests and executed evidence to inspect the engineering and scientific decisions without publishing restricted data or private competitive artifacts.

[Reproducibility guide →](docs/REPRODUCIBILITY.md)

## Deep research archive

For the full experiment lineage—including rejected hypotheses, source attribution, neural reproduction status and validation decisions—use the [frontier research index](research/frontier/README.md).

**Current release state:** the verified competition release remains **0.57586 private / 0.57601 public**. The newest click/cart research stack is **not currently promoted or deployed** while comparator reconciliation is in progress.
