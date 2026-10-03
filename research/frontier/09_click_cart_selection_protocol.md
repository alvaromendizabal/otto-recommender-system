# Click/cart selection · completed protocol and outcome

This document preserves the frozen protocol and final outcome of the first objective-specific click/cart source-aware challenger.

AWS remained the canonical private execution workspace. The public record contains aggregate metrics, gate definitions, and decisions only.

## Verified competition state

The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public**, submission **56542128**.

The official metric is pooled **Weighted Recall@20 = 0.10 clicks + 0.30 carts + 0.60 orders**, higher is better.

## Fitting result

The 589-feature source-aware challenger retained the stable 400-candidate pool.

| Metric | Clicks | Carts |
| --- | ---: | ---: |
| Baseline hits | 10,151 | 2,616 |
| Challenger hits | **10,337** | **2,669** |
| Net hits | **+186** | **+53** |
| Weighted contribution | **+0.000961** | **+0.002574** |

Combined fitting gain: **+0.003535**

Stability:

- 5/5 chronological folds positive
- worst fold **+0.001635**

Decision: **PROMOTE_TO_SELECTION**

## Frozen independent-selection gate

The 20,000-session selection comparison was frozen before label access.

Promotion required all of:

- combined click/cart gain ≥ +0.0025
- click hit gain ≥ 0
- cart hit gain ≥ +10
- paired-bootstrap 95% lower bound > 0
- chronological first half ≥ 0
- chronological second half ≥ 0

Bootstrap replicates: **2,000**

## Independent selection outcome

The challenger failed the frozen gate:

- clicks: **9,928 → 10,187**, **+259 hits**
- carts: **2,736 → 2,675**, **−61 hits**
- combined weighted gain: **−0.001738**
- paired 95% interval: **[−0.003914, +0.000415]**
- first half: **−0.001142**
- second half: **−0.002327**

Decision:

**STOP_CLICK_CART_SELECTION**

The opened selection cohort was permanently retired from tuning.

## Research consequence

The result exposed asymmetric transfer:

- source-aware evidence helped clicks;
- the same joint recipe damaged carts.

That evidence motivated two materially different follow-up families:

1. contextual/reliability-aware routing;
2. candidate-conditioned temporal sequence interaction.

Both were evaluated under fresh fitting-only OOF controls rather than retuning the rejected source-aware tree recipe.

See [10_contextual_and_sequence_frontier.md](10_contextual_and_sequence_frontier.md) for those experiments and [11_heterogeneous_stack_and_fresh_selection.md](11_heterogeneous_stack_and_fresh_selection.md) for the later stack promotion.

## Public/private boundary

Public:

- aggregate hit counts and metric deltas
- cohort sizes
- frozen gate
- confidence interval
- chronological stability
- decision

Private:

- session IDs
- row-level labels/predictions
- full embeddings/checkpoints
- private runners
- cloud paths and credentials
- exact competitive orchestration
