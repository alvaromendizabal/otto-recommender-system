# Click/cart selection protocol and outcome · October 2, 2026

This document records the frozen independent-selection contract for the source-aware click/cart challenger and its completed outcome. AWS remains the canonical private execution workspace. GitHub publishes aggregate evidence, source attribution, metric contracts, promotion gates and validation design; raw competition data, row-level labels/predictions, private runners, checkpoints, embeddings, optimizer state, credentials and exact cloud orchestration remain private.

## Verified submission lineage

The strongest verified post-competition Kaggle result at this stage remained **0.57586 private / 0.57601 public**, submission **56542128**. No newer competition submission was produced by this experiment.

The primary metric is pooled **Weighted Recall@20 = 0.10 clicks + 0.30 carts + 0.60 orders**, higher is better.

## Fitting result that opened selection

The promoted fitting challenger kept the established 400-candidate pool and augmented the candidate/session representation with heterogeneous source evidence.

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

The challenger passed every fitting gate. The 1,200-candidate variants regressed, so selection evaluated the **source-aware 400-candidate click/cart ranker**, not a broader candidate union.

## Frozen selection design

Selection used an independent **20,000-session cohort**. The challenger was frozen before selection labels were opened.

Comparator:

- **clicks:** established incumbent click list;
- **carts:** selected cart model for long observed prefixes and incumbent fusion otherwise;
- **orders:** unchanged and not optimized in this selection milestone.

The preregistered gate required:

- combined click/cart weighted gain >= **+0.0025**;
- click hit gain >= **0**;
- cart hit gain >= **+10**;
- paired session-bootstrap 95% interval lower bound **> 0**;
- chronological first-half gain >= **0**;
- chronological second-half gain >= **0**.

The paired bootstrap used **2,000 replicates**.

## Independent selection outcome

The completed selection did **not** confirm the joint source-aware policy:

| Metric | Result |
| --- | ---: |
| Click hit gain | **+259** |
| Cart hit gain | **-61** |
| Combined weighted gain | **-0.001738** |
| Paired 95% bootstrap interval | **[-0.003914, +0.000415]** |
| First-half gain | **-0.001142** |
| Second-half gain | **-0.002327** |

Decision: **STOP_CLICK_CART_SELECTION**.

The result is informative rather than discarded: heterogeneous source evidence transferred strongly to clicks but not to carts. That asymmetry motivated the later contextual, sequence and heterogeneous-ensemble studies documented in [10_contextual_sequence_stack_validation.md](10_contextual_sequence_stack_validation.md).

## Leakage and promotion discipline

The selection cohort is now closed for tuning. No blend weights, thresholds, feature choices or model-family decisions may be learned from it.

Subsequent research returned to fitting-only OOF evidence and later created a new independent Fresh Selection V2 before further promotion. Reserved evaluation labels remained sealed throughout those fitting studies.

## Public/private boundary

Public artifacts expose:

- metric definitions and direction;
- aggregate fitting and selection evidence;
- promotion/kill gates;
- cohort sizes and stage roles;
- source/reproduction status;
- decision logic.

They exclude:

- raw event partitions;
- row-level targets or predictions;
- session/cohort identifiers;
- full embeddings or neural checkpoints;
- optimizer state;
- private handoff runners;
- private cloud paths and account metadata;
- exact private orchestration details.

See [08_neural_objective_frontier.md](08_neural_objective_frontier.md) for the fitting evidence and [10_contextual_sequence_stack_validation.md](10_contextual_sequence_stack_validation.md) for the subsequent research program.
