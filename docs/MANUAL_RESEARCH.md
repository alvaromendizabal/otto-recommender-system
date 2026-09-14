# OTTO | Owner-run feature research

Exploratory internal fitting folds; NOT leaderboard scores.

Compare only within a matched round. Reused cohorts and adaptive hypothesis selection limit inference.

| Round | Representation | Weighted Recall@20 |
| --- | --- | ---: |
| 01 | control_shared | 0.484737 |
| 01 | plus_both | 0.482541 |
| 01 | plus_support | 0.480984 |
| 01 | plus_timing | 0.494230 |
| 02 | control_shared | 0.569231 |
| 02 | plus_timing | 0.566425 |
| 03 | control_shared | 0.569231 |
| 03 | mass_missing_aware | 0.564225 |
| 03 | mass_zero_ablation | 0.557076 |
| 04 | control_shared | 0.569231 |
| 04 | demand_rolling | 0.554024 |
| 04 | demand_static_ablation | 0.552895 |
| 05 | action_pairs | 0.556427 |
| 05 | collapsed_action_ablation | 0.562842 |
| 05 | control_shared | 0.569231 |
| 06 | control_shared | 0.569231 |
| 06 | global_context_ablation | 0.561671 |
| 06 | repeat_context | 0.558451 |
| 08 | control_shared | 0.569231 |
| 08 | equal_weight_ablation | 0.565877 |
| 08 | rarity_neighbors | 0.562598 |
| 09 | control_shared | 0.569231 |
| 09 | forward_continuation | 0.568002 |
| 09 | unsigned_window_ablation | 0.567680 |
| 10 | control_shared | 0.569231 |
| 10 | ordered_funnel | 0.567267 |
| 10 | unordered_funnel_ablation | 0.567057 |

## Source and evidence

Implementation and executed notebooks are public in `research/manual/`. Dataset-derived provenance, raw events, per-session labels, fitted arrays, model weights, credentials, and environments stay in AWS. See the exclusions inventory. Source snapshots include negative results and pending work; a commit does not certify an experiment or imply improvement. Restore exact local-only dependencies before executing a historical round from a fresh checkout.

[Source](../research/manual/README.md) · [Interactive report](../reports/manual_research/progress.html) · [Notebook](../reports/manual_research/summary.ipynb)

The runtime checkout remains pinned while a separate publication clone incorporates the public updates. This is a reviewed source publication, not a byte-for-byte data mirror or a runtime migration.
