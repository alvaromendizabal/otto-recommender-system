# Transition and model-family frontier · September 29, 2026

This is a public aggregate status snapshot. AWS remains the canonical private execution workspace. Raw events, labels, row-level predictions, embeddings, model binaries, optimizer state, credentials, private runner bundles and exact orchestration are intentionally excluded.

## Verified incumbent

The strongest verified post-competition submission remains **56542128 at 0.57586 private / 0.57601 public**. It preserves the previous click/cart lists and replaces orders with the promoted neural-similarity ranker. No experiment described below has produced or claimed a newer leaderboard score.

## What closed since the previous snapshot

| Study | Evaluation role | Best aggregate evidence | Decision |
| --- | --- | --- | --- |
| Complementary attention candidate union | Fitting-only cross-validation | Base representation **+7 order hits / +0.000235**; union arms regressed | Close tested union; preserve encoder as a feature source |
| Dense neural interaction + score stack | Fitting-only cross-validation | **+7 order hits / +0.000235** | Stop; below +0.0015 gain gate |
| Transition/source representation | Fitting-only cross-validation | **+37 order hits / +0.001241** across a 496-feature representation | Preserve representation; weighted-gain gate missed narrowly |

The transition/source study is the strongest fitting-only result since the promoted similarity model. It independently adapts broader leading-solution mechanisms—direct candidate-source evidence, position/time/action-weighted co-visitation and directional transition factors—without copying upstream weights or row-level predictions. It passed the hit and fold-stability requirements but missed the preregistered **+0.0015 weighted Recall@20** gate, so selection remained closed.

## Current active experiment

The stronger transition/source representation motivated a **different model-family test rather than more feature reshuffling**. A fixed study compares a gradient-boosted binary classifier, a query-group boosted ranker and fixed preregistered blends with the existing LightGBM ranker.

The study uses the same chronological roles and 496-feature representation. It passed fitting-only screening and then completed the **20,000-session selection cohort** successfully. Only after that gate passed did it enter the **432,492-session reserved temporal evaluation**.

At the last captured owner log, **134,144 / 432,492 evaluation sessions** had completed. This is progress evidence, not a final result. The selected arm identity, final evaluation gain and any deployment decision remain unpublished until the run finishes.

## Public reproducibility boundary

This repository exposes enough to review the scientific progression:

- dated aggregate metrics and stop/advance decisions;
- chronological cohort sizes and gate ordering;
- source attribution and a component-level reproduction matrix;
- tests that prevent an in-progress experiment from being presented as a completed gain;
- the verified incumbent submission identity and score lineage.

It intentionally does not publish the private feature cache, per-session predictions, fitted model binaries, embedding tables, checkpoint files, exact orchestration or current evaluation parts.

The public review contracts can be checked from the repository root:

```bash
uv sync --frozen --extra dev --extra ml
.venv/bin/python -m pytest -q tests/test_transition_frontier_publication.py
```

See the [machine-readable September 29 snapshot](../../reports/research/frontier_status_20260929.json), [execution ledger](../../reports/research/execution_ledger_20260929.json), [previous promoted similarity snapshot](05_similarity_attention_frontier.md), and [component reproduction matrix](../neural_stack/reproduction_matrix.json).
