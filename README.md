# OTTO · Session-based recommendation

**I built a large-scale offline recommender, from anonymous event data to verified batch delivery.**

[![CI](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml)
[![Public demo](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/public-demo.yml/badge.svg?branch=main)](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/public-demo.yml)

**Verified release: 0.57586 private / 0.57601 public**, post-competition submission **56542128**. The system combines the established click/cart routing policy with a promoted neural-similarity order ranker. This is an offline research and delivery result, not an official rank, online deployment or measured business lift.

**[Explore the synthetic demo](https://alvaro-otto-session-lab.tartmacaw2.chatgpt.site)** · [Five-minute review](docs/REVIEWER_GUIDE.md) · [Case study](docs/PORTFOLIO.md) · [Reproduce the evidence](docs/REPRODUCIBILITY.md)

The public demo runs a small Python recommendation pipeline and produces an interactive offline report. Inspect observed sessions, ranked items, score components and separately evaluated future targets. Its invented data and transparent rules are independent of the private research system.

## 60-second overview

| Area | Measured or implemented scope |
|---|---|
| Data and delivery | **216.7M** training events · **1.67M** official inference sessions · **5.02M** validated output rows |
| Retrieval and ranking | Multiple retrieval sources, up to **400 candidates/session**, objective-specific ranking and routing |
| Feature research | **1,482** formulas screened to a **102-feature** controlled reference; later specialized rankers extend it |
| Controlled result | **0.584392 weighted Recall@20** versus **0.564904** for the compact control on **432,492 temporal sessions** |
| Recovery and verification | Immutable manifests, resumable AWS partitions, native-model replay, executed notebooks and CI |

## What I owned

- **The recommendation pipeline:** candidate generation, feature computation, click/cart/order ranking, neural similarity and full-population output validation.
- **The experimental decisions:** temporal roles, point-in-time controls, fitting-only screening, matched ablations, uncertainty and frozen promotion gates.
- **The execution and evidence:** AWS/S3 recovery, deterministic artifact identities, replay checks, failure regressions, executed reviews and public/private boundaries.

[Employer overview](docs/EMPLOYER_OVERVIEW.md) · [Implementation case study](docs/PORTFOLIO.md)

## System architecture

~~~mermaid
flowchart TD
    H["Historical events with a time cutoff"] --> C["Multi-source candidate retrieval"]
    P["Observed session prefix"] --> C
    C --> F["Candidate and session features"]
    H --> F
    P --> F
    F --> R["Task-specific ranking and routing"]
    R --> B["Resumable batch inference"]
    B --> V["Validated top-20 lists and artifact provenance"]
~~~

The pooled metric is **Weighted Recall@20 = 10% clicks + 30% carts + 60% orders**. I measure **candidate availability** as an oracle ceiling, separately from achieved ranking. [Architecture and contracts](docs/ARCHITECTURE.md).

## Two results worth examining

**The controlled representation improved recall.** On the matched 432,492-session temporal cohort, candidate fusion scores **0.535244**, the compact ranker **0.564904**, and the selected ranker **0.584392**. The selected-versus-compact gain is **1.949 percentage points**, with a paired 95% session-bootstrap interval of **1.840 to 2.065 points**. This is a local study, not a competition score. [Executed study](notebooks/09_controlled_feature_study.ipynb).

**The delivered release was independently measured by Kaggle.** Full inference produced **5,015,409 validated rows for 1,671,803 official sessions**. The verified post-competition lineage progressed from **0.56842 private / 0.56862 public** to **0.57586 private / 0.57601 public**. [Recorded release](reports/submissions/similarity_stack_20260925.json) · [Inference provenance](docs/INFERENCE.md).

A meaningful negative result remains visible: the corrected click/cart challenger is **rejected** because replay against the actual deployed comparator exposed a **3,529-click regression**, despite a positive weighted point estimate. The correction preserved the existing release. [Validation-integrity case](research/frontier/11_validation_integrity_reconciliation.md).

On a separate archived cart diagnostic, **2,181 of 4,778 capped targets** were ranked successfully and **2,824** were available in the candidate pool. That leaves **643 within-pool ranking misses** and **1,954 outside-pool misses**. These counts identify different failure mechanisms; an oracle ceiling is not an attainable forecast. [Coverage case study](research/frontier/12_corrected_comparator_candidate_coverage.md).

## Run the public example and audit

From a checkout with **Python 3.11+**, without package installation, accounts or cloud resources:

```bash
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo --check
python3 -S scripts/review_portfolio.py
```

Open `/tmp/otto-public-demo/report.html`. Repeating generation verifies and reuses the completed artifacts. The synthetic pipeline produces predictions, metrics and a hash manifest; the separate evidence audit recomputes published aggregate arithmetic and checks recorded identities. Neither path reproduces the private release or obtains a new competition score.

[Demo guide](docs/PUBLIC_DEMO.md) · [Locked quality and notebook commands](docs/REPRODUCIBILITY.md) · [Bounded historical model replay](notebooks/10_competition_inference.ipynb)

## Release and research status

The delivered release remains **0.57586 private / 0.57601 public**. The inspected **v27 owner return** completed its scientific comparison using two reused ranker checkpoints and **zero new fits**. The candidate failed the matched-control gate; later independent confirmation and submission were skipped. No new score or pending submission was created. [Current evidence and remaining work](docs/RESEARCH_STATUS.md).

Feature and candidate research remains open under the [standing rules](docs/EXECUTION_RULES.md). Completed delivery does not close every research hypothesis or turn previously inspected cohorts into fresh tests.

Published materials include public research code, tests, aggregate reports, executed notebooks, this synthetic demo and a deliberately bounded historical replay with three native reference models. Full event/prediction populations, current private checkpoints, embeddings and private orchestration remain outside the release. [Publication scope](docs/PUBLICATION_SCOPE.md).

**Stack:** Python · Polars · LightGBM · XGBoost · PyTorch · FAISS · AWS SageMaker/S3 · Plotly · pytest · Ruff · mypy · GitHub Actions.
