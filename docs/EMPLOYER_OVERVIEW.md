# Employer overview · OTTO session recommender

## Executive summary

I built this repository as an end-to-end recommendation-system research and delivery project, not as a single competition notebook.

The system converts **216.7 million anonymous shopping events** into objective-specific product recommendations for clicks, carts and orders. It combines multi-source candidate retrieval, task-specific learning to rank, neural representations, controlled temporal validation and resumable AWS batch inference.

The strongest verified post-competition release scored **0.57586 private / 0.57601 public** and generated **5,015,409 validated recommendation rows for 1,671,803 sessions**.

## Try the engineering in one command

```bash
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo
```

Open `/tmp/otto-public-demo/report.html`. The interactive report traces synthetic sessions through retrieval, task-specific ranking and held-out evaluation, with item score explanations and verified artifact identities. Python 3.11+ is the only requirement. The example runs locally without accounts or cloud resources; its metrics are illustrative and separate from the measured research below. [Demo walkthrough](PUBLIC_DEMO.md).

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

An archived time-controlled diagnosis separates ranking and retrieval error: on a 4,778-target cart denominator, the reference system recovers **2,181** targets while its candidate pool contains **2,824**. That leaves **643 available targets missed by ranking** and **1,954 outside the pool**. This evidence supports investigating both mechanisms, without treating a candidate ceiling as achieved recommendation quality.

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

## What a reviewer can reproduce

| Path | What it verifies | Boundary |
| --- | --- | --- |
| Synthetic demo | Complete local retrieval, ranking, evaluation, explanations and deterministic replay | Conventional teaching rules; no research-performance claim |
| Evidence review | Published aggregate arithmetic, release identities and error decomposition | No retraining or new external evaluation |
| Historical native-model replay | Three public reference models on eight example sessions | Bounded historical sample; not the private release |
| Full research workflows | Code, protocols, tests and executed reports are inspectable | Original data and larger private artifacts required |

The [publication scope](PUBLICATION_SCOPE.md) identifies the existing public reference artifacts and what remains private. Online serving and business impact were not evaluated.

## Review next

- [Architecture](ARCHITECTURE.md)
- [Research case study](PORTFOLIO.md)
- [Reproducibility](REPRODUCIBILITY.md)
- [Public demo](PUBLIC_DEMO.md)
- [Inference and provenance](INFERENCE.md)
- [Controlled feature study](../notebooks/09_controlled_feature_study.ipynb)
- [Research frontier](../research/frontier/README.md)
