# System architecture · OTTO session recommender

## Design goal

Given only the observed prefix of an anonymous shopping session and permitted historical events, produce three ranked top-20 product lists: clicks, carts and orders.

The architecture separates **retrieval**, **representation**, **ranking**, **validation** and **delivery** so each layer can be tested independently.

## End-to-end flow

~~~mermaid
flowchart TD
    D[Historical event partitions] --> H[Point-in-time historical state]
    S[Observed session prefix] --> Q[Session representation]
    H --> R[Candidate retrieval]
    Q --> R
    R --> C[Candidate pool ≤ 400]
    C --> F[Candidate × session feature matrix]
    H --> F
    Q --> F
    Q --> N[Neural / sequence representations]
    N --> F
    F --> L1[Click ranker]
    F --> L2[Cart ranker]
    F --> L3[Order ranker]
    L1 --> P[Objective-specific policy / routing]
    L2 --> P
    L3 --> P
    P --> T[3 × top-20 lists]
    T --> B[Partitioned batch inference]
    B --> G[Validation + immutable receipts]
    G --> O[5.02M final rows]
~~~

## 1. Data and availability contracts

The raw task is event based: session ID, product ID, timestamp and event type.

The important design rule is not the schema—it is **availability**. Features and retrieval state must only use information available before the prediction point.

The project therefore records:

- source identities and file hashes;
- conversion manifests;
- temporal cutoffs;
- role-specific session ledgers;
- feature/model seals;
- inference input attestations.

Changing a fingerprinted input or implementation invalidates incompatible cached artifacts.

## 2. Candidate retrieval

Candidate generation limits the expensive ranking stage to a bounded pool.

Research channels include:

- co-visitation transitions;
- cart/order-weighted transitions;
- observed-session repeats;
- historical popularity;
- matrix-factorization retrieval;
- Word2Vec-style retrieval;
- neural sequence retrieval.

The production-style pipeline uses bounded candidate budgets, while research reports track **candidate ceiling** separately from achieved ranking quality.

That distinction matters: better coverage is useful only if downstream ranking can exploit it.

## 3. Feature layer

The controlled study generated **1,482 candidate/session formulas** across:

- historical frequency and recency;
- session context;
- repeated-item behavior;
- graph affinity;
- intent interactions;
- time-window statistics;
- candidate/source evidence.

Screening and controlled model selection produced a **102-feature** reference schema.

Later research adds specialized similarity/source/sequence evidence without silently changing the reference study.

## 4. Objective-specific ranking

The three objectives share candidates but do not share identical ranking behavior.

The controlled release uses native LightGBM LambdaRank models. Later research also evaluates:

- XGBoost ranking/classification variants;
- neural candidate/session interaction;
- sequence attention;
- learned similarity features;
- heterogeneous OOF combinations.

The release policy is objective specific. This is why deployment-parity tests compare final recommendation membership and routing behavior, not only raw model scores.

## 5. Neural representation layer

Neural work is treated as a representation source rather than assumed to outperform tree rankers directly.

A task-conditioned sequence encoder was tested in multiple roles:

- direct retrieval;
- neural-aware reranking;
- candidate/session similarity evidence;
- sequence interaction;
- OOF ensemble diversity.

One important project lesson is that a representation can fail as a standalone ranker yet still add value as stable similarity evidence.

## 6. Validation architecture

~~~mermaid
flowchart LR
    H[Permitted historical period] --> F[Fit]
    F --> S[Selection]
    S --> E[Reserved temporal evaluation]
    E --> D[Deployment parity]
    D --> R[Release]
~~~

The validation stack includes:

- chronological roles;
- disjoint session assignments;
- point-in-time feature generation;
- model seals before reserved evaluation;
- complete candidate pools on selection/evaluation;
- paired session-bootstrap uncertainty;
- chronological slices;
- OOF predictions for learned stacking;
- explicit promotion/kill thresholds;
- deployment-parity replay.

A later click/cart research path was intentionally blocked when parity auditing showed that its comparator did not exactly match the deployed routing policy.

## 7. Batch inference and recovery

Full official-prefix inference spans:

- **1,671,803 sessions**
- **5,015,409 output rows**

Large work is partitioned and resumable.

Each compatible part is tied to:

- input identity;
- model identity;
- implementation fingerprint;
- output digest;
- row/session coverage;
- completion receipt.

Interrupted jobs reuse sealed compatible parts. A delivery failure does not imply that the model must be retrained.

## 8. Observability and correctness

The private execution system records:

- stage transitions;
- progress counts;
- elapsed runtime;
- resource utilization;
- memory/disk headroom;
- checkpoint state;
- estimated compute cost;
- output hashes;
- failure classes.

The public repository exposes aggregate evidence and reproducibility contracts without publishing exact private orchestration or row-level predictions.

## 9. Software-quality architecture

GitHub Actions separates concerns into:

- project quality and regression tests;
- neural-contract tests;
- executed-notebook replay;
- portfolio rendering and verification.

The public notebooks are evidence-backed artifacts: figures are generated from checked research reports rather than manually typed chart values.

## 10. Public/private boundary

### Public

- selected implementations;
- aggregate experiment evidence;
- model/metric contracts;
- tests;
- executed notebooks;
- architecture and reproducibility documentation;
- source attribution.

### Private

- raw competition data;
- row-level labels/predictions;
- session/cohort identifiers;
- private execution runners;
- full model checkpoints;
- large embeddings;
- credentials;
- exact cloud orchestration.

This boundary keeps the repository useful for technical review while protecting data and competitive implementation details.
