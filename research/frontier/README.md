# Frontier research · evidence, mechanisms, and decisions

**Latest snapshot: October 3, 2026.**

The public research surface now covers the progression from the promoted neural-similarity order system through objective-specific click/cart modeling, contextual routing, candidate-conditioned sequence attention, heterogeneous OOF stacking, fresh independent selection, and the active sharded Final Reserve V2 protocol.

The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public** on submission **56542128**. Newer research has not yet produced a newer Kaggle submission; it is still moving through the offline promotion chain.

AWS/SageMaker remains the canonical private execution environment. GitHub publishes aggregate evidence, source attribution, validation contracts, readable selected implementations, and employer-facing decisions—not raw data, row-level predictions, private cohort IDs, checkpoints, credentials, or private orchestration.

## Read in order

1. [01_frontier_review.ipynb](01_frontier_review.ipynb) — early matched mechanism studies
2. [02_training_scale_status.ipynb](02_training_scale_status.ipynb) — scaling and checkpoint recovery
3. [04_competition_frontier.ipynb](04_competition_frontier.ipynb) — verified submission progression at the September snapshot
4. [05_similarity_attention_frontier.md](05_similarity_attention_frontier.md) — promoted neural-similarity transfer
5. [06_transition_model_frontier.md](06_transition_model_frontier.md) — transition/source features and model-family diversification
6. [07_candidate_model_frontier.md](07_candidate_model_frontier.md) — MF/W2V/Seq2Seq and broad candidate-model studies
7. [08_neural_objective_frontier.md](08_neural_objective_frontier.md) — full neural-family sweep and source-aware click/cart fitting
8. [09_click_cart_selection_protocol.md](09_click_cart_selection_protocol.md) — independent selection outcome for the source-aware joint recipe
9. [10_contextual_and_sequence_frontier.md](10_contextual_and_sequence_frontier.md) — CRAFT and sequence cross-attention
10. [11_heterogeneous_stack_and_fresh_selection.md](11_heterogeneous_stack_and_fresh_selection.md) — OOF stack promotion and fresh selection
11. [12_reserved_evaluation.md](12_reserved_evaluation.md) — active final reserve protocol and checkpoint state

## Current research state

### Verified submission

Submission **56542128** remains the verified public competition result at **0.57586 private / 0.57601 public**.

Its order model uses learned candidate-to-session similarity derived from a task-conditioned sequence representation. On reserved temporal evaluation, that model added **+0.004396 weighted Recall@20 and +508 order hits** while preserving the click/cart policy.

### Objective-specific source-aware ranking

A 589-feature source-aware click/cart ranker produced:

- +186 click hits
- +53 cart hits
- +0.003535 combined weighted contribution
- 5/5 positive fitting folds

Independent selection exposed asymmetric transfer: clicks improved while carts regressed. The joint recipe was closed without retuning on the opened selection cohort.

### Contextual routing

CRAFT-style contextual/reliability-aware transport tested four model families across five folds:

- flat MLP control
- contextual gate
- explicit cross interactions
- contextual mixture of experts

All **20 GPU fits** completed. No CRAFT arm passed its frozen gate, so the direction was scientifically closed.

### Sequence cross-attention

Four candidate-conditioned temporal sequence families were tested across five folds:

- residual MLP
- DIN-style cross-attention
- BST-style sequence attention
- behavior-specific cross-attention

All **20 GPU fits** completed.

The strongest arm, DIN-style long-session routing, produced **+0.001678 combined gain and +12 cart hits versus the deployment incumbent**, with stable 5-fold behavior. It did not clear the +0.003 combined promotion threshold, but its OOF predictions were preserved for ensemble research.

### Heterogeneous OOF stack

Nine heterogeneous OOF rankers were combined using:

- reciprocal-rank fusion
- within-session standardized-score averaging
- leakage-safe learned meta rankers

The selected fitting stack, **`zmean_seq__all`**, achieved:

- **+0.004203 combined deployment-aligned gain**
- **+64 cart hits**
- **5/5 nonnegative combined folds**
- **5/5 nonnegative cart folds**
- worst combined fold **+0.002071**

The result passed its frozen fitting gate.

### Fresh Selection V2

Because the earlier 20k selection cohort had already been opened, a new independent selection stage was frozen from previously unopened evaluation-role sessions using observed-prefix chronology only.

Fresh Selection V2:

- sessions: **20,000**
- combined gain: **+0.003194**
- clicks: **+136 hits**
- carts: **+47 hits**
- paired 95% interval: **[+0.001865, +0.004554]**
- both chronological halves positive
- decision: **PASS**

### Final Reserve V2 — active

The remaining reserve contains **412,492 sessions**.

Predictions are prepared in deterministic chronological shards before any reserve label is opened.

Current checkpoint:

- sessions predicted: **200,000 / 412,492**
- shards sealed: **4 / 9**
- remaining sessions: **212,492**
- reserve labels opened: **false**

The final reserve gate is frozen in advance and includes pooled gain, click/cart hit floors, block-bootstrap uncertainty, chronological halves, and quarter stability.

## Research principles demonstrated

- **OOF-first ensembling:** stackers learn only from leakage-safe OOF predictions.
- **Independent promotion stages:** fitting, selection, final reserve, and competition inference remain separate.
- **Fresh-cohort recovery:** when an earlier selection cohort was opened, a new chronological selection boundary was frozen instead of reusing it.
- **Negative-result retention:** failed hypotheses stay documented and reduce priority for near-duplicate experiments.
- **Role changes for representations:** a neural model may fail directly yet remain useful as a feature source or complementary stack component.
- **Infrastructure reuse:** large caches, prediction shards, and model checkpoints resume instead of restarting.
- **Explicit publication boundary:** GitHub shows aggregate scientific evidence without exposing private row-level competition artifacts.

## Public-source attribution

Mechanisms were independently adapted from strong public OTTO solution families and recommender research. Public repositories are used for mechanism inspiration and attribution; this project does not publish copied row-level predictions, private checkpoints, or private orchestration.

The public review distinguishes:

- recreated and validated
- adapted
- equivalent mechanism already covered
- scientifically closed
- blocked
- not yet implemented

## Reproduce the public review

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python -m pytest -q   tests/test_frontier_20261003_publication.py   tests/test_click_cart_selection_publication.py
```

See [frontier_status_20261003.json](../../reports/research/frontier_status_20261003.json) for the machine-readable current snapshot.