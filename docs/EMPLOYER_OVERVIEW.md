# Employer overview · OTTO session recommender

## Executive summary

I built this repository as an end-to-end recommendation-system research and delivery project, not as a single competition notebook.

The system converts **216.7 million anonymous shopping events** into objective-specific product recommendations for clicks, carts and orders. It combines multi-source candidate retrieval, task-specific learning to rank, neural representations, controlled temporal validation and resumable AWS batch inference.

The strongest verified post-competition release scored **0.57586 private / 0.57601 public** and generated **5,015,409 validated recommendation rows for 1,671,803 sessions**.

## What I built

| Layer | Implementation |
| --- | --- |
| Data | Typed, hashed event partitions with source provenance and point-in-time contracts |
| Retrieval | Co-visitation, repeat/session signals, popularity, neural and latent retrieval research |
| Candidate budget | Up to 400 products per session |
| Feature research | 1,482 candidate/session formulas; controlled reference narrowed to 102 selected features |
| Ranking | Task-specific LambdaRank, plus XGBoost, neural-similarity and sequence-model research |
| Validation | Chronological fit/selection/evaluation roles, OOF stacking, paired uncertainty and frozen gates |
| Inference | Resumable partitioned batch scoring with deterministic receipts and full-output validation |
| Cloud | AWS SageMaker + S3 as the canonical private execution environment |
| Software quality | pytest, ruff, mypy, GitHub Actions, executed notebooks and publication regression tests |

## What I owned

I was responsible for the full ML lifecycle represented here:

1. **problem formulation and metric implementation**;
2. **candidate-retrieval architecture**;
3. **feature engineering and selection**;
4. **model training and controlled ablations**;
5. **validation and leakage prevention**;
6. **neural/sequence research and ensembling**;
7. **AWS execution, recovery and artifact lineage**;
8. **full-population inference and output validation**;
9. **CI, notebooks, documentation and publication boundaries**;
10. **scientific incident correction when deployment parity exposed invalid comparison assumptions**.

## Why the project is technically difficult

### 1. Retrieval and ranking are coupled

A ranker cannot recover targets that never enter the candidate set. The project therefore measures both candidate ceiling and achieved Recall@20, and treats them as different quantities.

### 2. Three objectives behave differently

Clicks, carts and orders require different ranking behavior and carry different metric weights. The system uses objective-specific models and routing rather than assuming one universal ranking policy.

### 3. Temporal leakage is easy to introduce

Sessions are split chronologically and feature history is bounded by the prediction point. Learned preprocessing and model-selection roles are separated from reserved evaluation.

### 4. Large jobs need recovery semantics

Full inference spans 1.67M sessions. Intermediate partitions are sealed with identities and receipts so interrupted runs reuse completed work rather than restarting.

### 5. Validation must match deployment

The repository includes deployment-parity checks because a good offline score is only meaningful if the comparator, feature contract and release policy are the same ones being evaluated. A deployment-parity audit caught a comparator reconstruction mismatch in a newer research path; the corrected replay reproduced the true incumbent and **rejected the challenger because click hits regressed**, despite positive cart and aggregate point estimates.

## Measured evidence

The controlled reference study evaluates **432,492 reserved temporal sessions**:

- candidate fusion: **0.535244**
- compact ranker: **0.564904**
- selected ranker: **0.584392**
- selected-vs-compact improvement: **+1.949 percentage points**
- paired 95% session-bootstrap interval: **+1.840 to +2.065 points**

The full official-prefix release lineage progressed from **0.56842 / 0.56862** to **0.57586 / 0.57601** private/public.

These competition measurements were recorded after the deadline and are presented as engineering/research evidence, not as an official medal or rank.

## Research breadth

The project evaluates mechanisms from multiple recommender families:

- weighted co-visitation and transition graphs;
- candidate/session interaction features;
- matrix-factorization retrieval;
- Word2Vec retrieval;
- task-conditioned neural sequence encoders;
- candidate-to-session similarity;
- sequence cross-attention;
- source-aware boosting;
- XGBoost and LightGBM ranking variants;
- OOF blending and stacking;
- hard-negative and candidate-budget studies.

Negative experiments remain documented so the repository demonstrates decision quality, not just successful endpoints.

The current time-controlled diagnosis also separates ranking and retrieval error: on a 4,778-target cart denominator, the strongest ranked system recovers **2,181** targets while the existing candidate pool contains **2,824**. This makes candidate availability the active research bottleneck and gives the next experiment a concrete reason to exist.

## Production-style engineering signals

Employers reviewing the codebase should notice:

- explicit data/model/metric contracts;
- deterministic artifact identities;
- checkpoint/restart behavior;
- resource and cost awareness;
- immutable evidence and append-only corrections;
- tests for corrupt/missing checkpoints and stale caches;
- isolated neural and notebook environments;
- CI jobs for quality, neural contracts, notebooks and portfolio artifacts;
- clear separation between public reproducibility evidence and private large artifacts.

## Review next

- [Architecture](ARCHITECTURE.md)
- [Research case study](PORTFOLIO.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Inference and provenance](INFERENCE.md)
- [Controlled feature study](../notebooks/09_controlled_feature_study.ipynb)
- [Research frontier](../research/frontier/README.md)
