# Similarity promotion and v31 frontier · September 28, 2026

This is a public, aggregate status snapshot. AWS remains the canonical private execution workspace; raw events, labels, row-level predictions, embeddings, checkpoints, optimizer state, credentials and private runner bundles are intentionally excluded.

## Verified score movement

| System | Private | Public | Decision |
| --- | ---: | ---: | --- |
| Frozen 102-feature reference | 0.56842 | 0.56862 | Reference |
| Objective router | 0.57100 | 0.57121 | Superseded |
| Long-session cart router · 56504354 | 0.57140 | 0.57166 | Prior incumbent |
| **Neural-similarity orders · 56542128** | **0.57586** | **0.57601** | **Current incumbent** |

The current submission improves the prior incumbent by **+0.00446 private / +0.00435 public**. These are post-competition measurements, not official medal or rank claims.

## Why the similarity branch mattered

The first neural-aware downstream reranker was rejected on the 20,000-session selection cohort. Rather than treating the learned representation itself as failed, the next study changed its role: the trained task-conditioned representation became candidate-to-session similarity evidence inside the established ranking system.

The promoted order model uses **137 features**: 102 established historical/session/context features, 28 candidate-to-session similarity aggregates and 7 neural query/retrieval diagnostics.

It passed the untouched selection gate and then all **432,492 reserved temporal evaluation sessions**:

| Metric | Result |
| --- | ---: |
| Weighted Recall@20 gain | **+0.004396** |
| Additional order hits | **+508** |
| Paired 95% interval | **+0.003721 to +0.005046** |
| First chronological half | **+0.004444** |
| Second chronological half | **+0.004350** |

Official-prefix deployment preserved the incumbent click/cart lists and changed orders only. The resulting Kaggle movement closely matched the magnitude of the reserved temporal gain.

## Closed branch: fixed GPU XGBoost family

A materially different GPU XGBoost family completed its fixed selection study. Its strongest arm added **+4 order hits** and **+0.000712 weighted Recall@20**, below the frozen +6-hit and +0.001 promotion gates. Evaluation labels stayed closed. This family is stopped rather than expanded with more same-family tuning.

## Active branch: complementary v31 attention union

The current branch independently adapts a first-place-style attention encoder as a complement to the already evaluated v42-derived representation. It is not presented as a reproduction of the full winning ensemble.

Verified progress: five neural epochs completed; exact top-200 neural retrieval completed for fitting and selection; the 100,000-session candidate-union feature cache completed; and five fixed ranking arms entered fitting-only cross-validation. Selection and reserved evaluation remain closed until that gate passes.

The planned union keeps established candidates first, adds complementary neural candidates, and expands the candidate-aware representation to **184 aggregate features**. Exact private orchestration, weights, embeddings and per-session outputs remain outside the public repository.

## What remains from the leading public systems

The project still does **not** claim complete reproduction of the leading OTTO systems. Major open components include the remaining first-place neural encoders, a complete multi-neural candidate ensemble, matrix-factorization and sequence-to-sequence channels, broader position/time/action similarity families, and complementary objective-specific ranker ensembles.

## Reproduce the public review surface

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python -m pytest -q tests/test_similarity_frontier_publication.py
```

See the [machine-readable snapshot](../../reports/research/frontier_status_20260928.json), [submission receipt](../../reports/submissions/similarity_stack_20260925.json), [execution ledger](../../reports/research/execution_ledger_20260928.json), and [component reproduction matrix](../neural_stack/reproduction_matrix.json).
