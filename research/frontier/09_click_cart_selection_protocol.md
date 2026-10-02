# Click/cart selection protocol · October 2, 2026

This public protocol freezes the next validation step for the strongest current fitting challenger. AWS remains the canonical private execution workspace. GitHub publishes aggregate evidence, source attribution, metric contracts, promotion gates and validation design; raw competition data, row-level labels/predictions, private runners, checkpoints, embeddings, optimizer state, credentials and exact cloud orchestration remain private.

## Verified competition state

The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public**, submission **56542128**. The recorded historical private winner is **0.60503**, leaving a **0.02917** private-score gap. No newer Kaggle submission is claimed by this protocol.

The primary metric is pooled **Weighted Recall@20 = 0.10 clicks + 0.30 carts + 0.60 orders**, higher is better.

## Fitting result that opened selection

The promoted fitting challenger keeps the established 400-candidate pool and augments the candidate/session representation with heterogeneous source evidence. The public result is aggregate only.

| Metric | Clicks | Carts |
| --- | ---: | ---: |
| Baseline hits | 10,151 | 2,616 |
| Challenger hits | **10,337** | **2,669** |
| Net hits | **+186** | **+53** |
| Official weighted contribution | **+0.000961** | **+0.002574** |

Combined official-metric contribution improved by **+0.003535**. All **5/5** chronological fitting folds were positive; the worst fold remained **+0.001635**.

The frozen fitting gate required:

- combined click/cart weighted gain >= **+0.0030**;
- click hit gain >= **0**;
- cart hit gain >= **+10**;
- at least **4/5** nonnegative folds;
- worst fold >= **-0.001**.

The challenger passed every fitting gate. The 1,200-candidate variants regressed, so the selected fitting policy is explicitly the **source-aware 400-candidate click/cart ranker**, not a broader candidate union.

## Frozen selection design

Selection uses the independent **20,000-session selection cohort**. The challenger is frozen before selection labels are opened.

The deployment-relevant comparator is the actual incumbent click/cart policy:

- **clicks:** the established incumbent click list;
- **carts:** the selected cart model for long observed prefixes and the incumbent fusion policy otherwise;
- **orders:** unchanged and not optimized in this selection milestone.

The selection stage advances only if all of the following hold:

- combined click/cart weighted gain >= **+0.0025**;
- click hit gain >= **0**;
- cart hit gain >= **+10**;
- paired session-bootstrap 95% interval has a lower bound **> 0**;
- chronological first-half gain >= **0**;
- chronological second-half gain >= **0**.

The paired bootstrap uses **2,000 replicates**. Reserved evaluation labels remain closed unless selection passes.

## Leakage and promotion discipline

The public contract is intentionally staged:

1. fit and freeze the final click/cart challenger using fitting data only;
2. generate all selection predictions with frozen models and frozen retrieval/source features;
3. open selection labels once for the predeclared comparison;
4. if selection passes, freeze the reserved-evaluation contract before opening reserved labels;
5. only after reserved evaluation passes may full competition inference be justified.

No blend weights, thresholds or feature choices are learned from selection or reserved evaluation results.

## Next decisions

**If selection passes:** freeze the exact click/cart challenger, open the predeclared reserved temporal evaluation on all **432,492 sessions**, preserve incumbent orders, and compare only the qualified click/cart replacement.

**If selection fails:** close this exact 589-feature source-aware tree recipe rather than retuning it on selection. The next research step should introduce a materially different capability such as contextual feature transport or a complementary nonlinear click/cart model.

**If only one objective transfers:** preserve the incumbent for the other objective and require the surviving objective to satisfy its own frozen weighted-contribution/stability requirements before promotion.

## Public/private boundary

Public artifacts intentionally expose:

- metric definitions and direction;
- aggregate fitting evidence;
- fitting and selection gates;
- cohort sizes and role separation;
- source/reproduction status;
- decision logic and failure criteria.

They intentionally exclude:

- raw event partitions;
- row-level targets or predictions;
- session/cohort identifiers;
- full embeddings or neural checkpoints;
- optimizer state;
- private handoff runners;
- private cloud paths and account metadata;
- exact private orchestration details.

See [08_neural_objective_frontier.md](08_neural_objective_frontier.md) for the fitting evidence that qualified this stage and [click_cart_selection_contract.json](click_cart_selection_contract.json) for the machine-readable frozen contract.
