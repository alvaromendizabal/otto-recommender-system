# OTTO · Session-based recommendation

**From anonymous shopping events to three ranked product lists.** An end-to-end recommendation project combining co-visitation retrieval, task-specific learning to rank, temporal validation, controlled feature research, and cloud batch inference.

[![CI](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/alvaromendizabal/otto-recommender-system/actions/workflows/ci.yml)

**Current verified result:** **0.57586 private / 0.57601 public** on submission **56542128**, scored after the competition deadline. The promoted similarity-stack order model improves the prior 0.57140 / 0.57166 incumbent by **+0.00446 private / +0.00435 public** while preserving its click and cart lists. The recorded historical private winning benchmark is **0.60503**, leaving a **0.02917** private-score gap. **Current frontier:** the major first-place neural variants are now materially covered; order-side flat source fusion and a QueryFormer-style field-sequence interaction are closed under fitting-only gates; the strongest new result is an objective-specific 589-feature click/cart source-aware ranker on the stable 400-candidate pool at **+0.003535 weighted contribution, +186 click hits, +53 cart hits and 5/5 positive folds**. It is fitting-qualified only and is awaiting independent selection. [Current neural/objective snapshot](research/frontier/08_neural_objective_frontier.md) · [Submission provenance](docs/INFERENCE.md)

## Start here

| Review path | What it demonstrates |
| --- | --- |
| [Research case study](docs/PORTFOLIO.md) | Problem framing, validation, feature selection, failure slices and engineering tradeoffs |
| [09 · Controlled feature study](notebooks/09_controlled_feature_study.ipynb) | Executed comparisons, ablations, uncertainty and audit evidence |
| [Frontier research review](research/frontier/01_frontier_review.ipynb) | Five completed scoring studies, candidate-coverage diagnosis and decisions |
| [Training scale and recovery](research/frontier/02_training_scale_status.ipynb) | Larger training support, model sealing and checkpoint recovery; later completed and bridged to the established pipeline |
| [03 · Neural frontier](research/neural_stack/03_neural_stack_status.ipynb) | Dated neural-design snapshot and first-place sequence-retrieval adaptation |
| [04 · Competition frontier](research/frontier/04_competition_frontier.ipynb) | Executed September 23 scorecard: verified leaderboard progression and closed hypotheses at that snapshot |
| [05 · Similarity + attention frontier](research/frontier/05_similarity_attention_frontier.md) | Promoted 0.57586 similarity result and the first complementary-attention study |
| [06 · Transition + model frontier](research/frontier/06_transition_model_frontier.md) | Dense-interaction closeout, transition/source features and model-family diversification |
| [07 · Candidate + model frontier](research/frontier/07_candidate_model_frontier.md) | MF/W2V/Seq2Seq candidate work, broad third-place recreation and CPU closeouts |
| [08 · Neural + objective frontier](research/frontier/08_neural_objective_frontier.md) | Full first-place neural mechanism sweep, order-side source/sequence closeouts and the fitting-qualified click/cart challenger |
| [05 · Two-tower results](notebooks/05_two_tower_results.ipynb) and [06 · ANN benchmark](notebooks/06_ann_benchmark.ipynb) | Objective-conditioned neural retrieval and nearest-neighbor experiments |
| [10 · Competition inference](notebooks/10_competition_inference.ipynb) | Frozen native-model replay and verified official-prefix batch delivery |

## The task and architecture

Predict what a shopper will click, add to cart and order next, using only the observed session prefix and permitted history. Input consists of product IDs, timestamps and event types, not product descriptions or user demographics. Output is three ranked lists of 20 products per session.

The [organizer's metric](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md) weights pooled Recall@20 as **10% clicks + 30% carts + 60% orders**. Denominators include capped future targets missing from retrieval. Candidate coverage is an oracle ceiling, never an achieved recommendation score.

```mermaid
flowchart LR
    H[Permitted historical events] --> G[Co-visitation and item statistics]
    P[Observed session prefix] --> C[Up to 400 candidates]
    G --> C
    P --> F[Candidate-session features]
    C --> F
    F --> R[Three task-specific LambdaRank models]
    R --> O[20 clicks · 20 carts · 20 orders]
```

## Delivered system and controlled evidence

The original feature study processes **216.7 million training events**, engineers **1,482 feature formulas**, and selects **102 features**. On its **432,492-session reserved temporal cohort**, the selected model reaches **0.584392 weighted Recall@20**, versus **0.564904** for the compact ranker and **0.535244** for candidate fusion. These systems share candidates and evaluation sessions. These are offline results, not Kaggle scores. [Exact evaluation](reports/research/evaluation.json)

![Matched temporal evaluation](reports/portfolio/results.svg)

The selected representation improves on the compact ranker by **1.949 percentage points**, with a paired 95% session-bootstrap interval of **1.840 to 2.065 points**. Chronological fitting, selection and evaluation roles, frozen artifact identities, and a reconstruction audit support the comparison. Previously explored data and training-seed uncertainty remain limitations. [Validation and failure slices](docs/PORTFOLIO.md)

Full batch inference covers **1,671,803 official test sessions** and **5,015,409 validated prediction rows**. The frozen 102-feature reference scored **0.56842 private / 0.56862 public**. The objective router moved to **0.57100 / 0.57121**, and the long-session cart router reached **0.57140 / 0.57166**. The promoted similarity-stack submission **56542128** preserves the long-session router's click/cart lists and replaces orders with a task-specific 137-feature ranker that combines the established representation with learned candidate-to-session similarity and neural-query diagnostics; it reaches **0.57586 private / 0.57601 public**. All are post-deadline measurements; none establishes an official medal or rank. [Inference and provenance](docs/INFERENCE.md)

## Research: test mechanisms, measure outcomes, preserve decisions

The recent studies test timing/training interactions, objective-specific routing, multi-hop retrieval, candidate-aware ranking, explicit path features and learned graph affinities. Their matched controls generally use **134 features**, not the submitted 102-feature system.

The two-hop mechanism increased candidate coverage by **0.023199**, but achieved recall declined. Its tested retraining recipe also regressed. The later timing hybrid and latent-affinity experiments did not establish transferable gains. These findings remain visible in the [executed comparison notebook](research/frontier/01_frontier_review.ipynb), alongside counts, intervals and [selected method implementations](research/frontier/README.md). Different cohorts must not be read as a leaderboard progression.

**Ceiling-escape sequence:** after the promoted v42-derived similarity stack reached **0.57586 private / 0.57601 public**, the project systematically attacked the remaining gap instead of repeatedly tuning the incumbent. The transition/XGBoost family produced stable but sub-gate reserved gains; MF retrieval exposed large candidate-ceiling headroom; unrestricted and protected candidate integration regressed; Word2Vec and lag-Seq2Seq retrieval were closed; and a 785-feature third-place-inspired CPU frontier narrowly missed its fitting gate. The project then completed independent bounded adaptations of the remaining first-place neural variants v15/v18/v21/v23/v27/v29 alongside the earlier v31/v42 work. These models added varying candidate diversity but did not produce a new order-side promotion. Flat heterogeneous source fusion and a QueryFormer-style field-sequence interaction also failed to unlock the order candidate ceiling. The strongest current fitting result instead comes from **objective-specific click/cart source-aware ranking on the stable 400-candidate pool: +0.003535 combined official-metric contribution, +186 click hits, +53 cart hits and 5/5 positive folds**. This challenger is frozen for independent selection; no newer Kaggle submission is claimed. [Current snapshot](research/frontier/08_neural_objective_frontier.md) · [Reproduction matrix](research/neural_stack/reproduction_matrix.json)

## Engineering and publication boundaries

**AWS is the canonical private execution workspace; GitHub is the curated review surface.** The public update includes executed notebooks, aggregate evidence, readable selected implementations and tests. It does not copy raw events, per-session labels or predictions, cohort IDs, full models, embeddings, environments, credentials or account logs.

Content-addressed inputs, atomic receipts, native-model replay, resource bounds and return bundles make long workflows inspectable and restartable. Later frontier runs extend that discipline to reusable neural checkpoints, fitting caches, per-part evaluation receipts, one-upload submission ledgers and explicit stop decisions. Engineering failures are recorded separately from model-quality failures. The similarity deployment required several engineering recoveries before producing a fully validated score; the fixed XGBoost branch later completed selection and was rejected scientifically rather than presented as a gain. The public repository reports aggregate outcomes and reproducibility contracts, not private checkpoints or row-level predictions.

## Explore and inspect

Foundational notebooks cover [validation](notebooks/01_validation_protocol.ipynb), [retrieval](notebooks/02_retrieval_benchmarks.ipynb), [candidate budgets](notebooks/03_candidate_frontier.ipynb), [hard negatives](notebooks/04_hard_negative_quality.ipynb), [ranking features](notebooks/07_ranking_features.ipynb) and [ranking evaluation](notebooks/08_ranking_evaluation.ipynb). Earlier manual research remains in [research/manual](research/manual/README.md). [Environment and artifact requirements](docs/REPRODUCIBILITY.md)

Important findings are embedded in the saved notebooks as inline Plotly figures and static fallbacks. The [original portable report](https://github.com/alvaromendizabal/otto-recommender-system/raw/refs/heads/main/reports/portfolio/otto-research-report.zip) is supplementary, not a replacement for notebook evidence.

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python scripts/run_quality_gate.py
```

**Scope:** the delivered reference system and verified post-competition submission lineage are complete and inspectable; performance research remains open. The verified competition incumbent is **0.57586 private / 0.57601 public** and the historical private winning benchmark of **0.60503** has not been reached. Since the October 1 snapshot, the project has materially covered the remaining first-place neural variants, closed their direct order-side retrieval/similarity recipes under frozen gates, rejected flat order source fusion and the current QueryFormer-style order interaction, and produced a **fitting-qualified click/cart challenger at +0.003535 weighted contribution with 5/5 positive folds**. That result is selection-pending and is not a leaderboard claim. No official rank, medal, production-service deployment, or state-of-the-art claim is made.
