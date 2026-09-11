# Fitting-only OOF screen for the frozen 134-column shared representation

This stage follows the passed Aug-16 1,024-session engineering gate. It asks one supervised question **without accessing selection**: does the frozen 32-feature addition improve out-of-fold ranking on fitting sessions under fixed candidates, negative sampling, model capacity and seeds?

## Frozen protocol

- Data: exactly the four 256-session complete fitting parts from the passed scale gate.
- Three deterministic whole-session folds from `session_hash(session, 20260911) % 3`.
- Arms: 102-feature baseline and 134-feature shared representation only.
- Training rows: all positives plus the original `negative_budget=60` hard/random negative policy with candidate seed `20260908`.
- OOF validation: all 400 candidates for every held-out fitting session.
- Model: deterministic LightGBM LambdaRank, 75 fixed rounds, 31 leaves, learning rate 0.05, min leaf 100, truncation 25, four threads.
- No early stopping and no fold-specific hyperparameter tuning.
- Metrics: pooled official weighted Recall@20, objective Recall@20, per-fold stability and paired whole-session bootstrap.
- No feature formula edits, retention decisions, selection access or promotion.

## Advance rule

Proceed only to **fitting-only** family ablation when all are true:

1. pooled shared-minus-baseline weighted Recall@20 is at least +0.001;
2. pooled orders Recall@20 does not decline;
3. shared weighted Recall@20 is nonnegative versus baseline in at least two of the three folds.

Passing this gate does not authorize selection access. It only buys fitting-only leave-one-block-out testing of funnel, episode, raw graph, row-normalized graph and degree-normalized graph blocks on the same folds and artifacts.

The separate strict as-of demand/support family remains a distinct open hypothesis.
