# OTTO · Historical same-product action progression

A fixed-ranker experiment on clicks→carts, carts→orders and clicks→orders within similar historical shopping sessions. It asks whether original event ordering adds information beyond otherwise matched co-action evidence.

**24 ordered features + a separate 24-feature unordered ablation**, each independently added to the saved 134-feature control. Same 4,096 fitting sessions, 400 candidates, two chronological folds and ranker settings. Twelve maximum challenger fits; six control replays; zero control refits. No automatic feature promotion.

Start with **[the two-round guide](TWO_ROUND_GUIDE.md)**. Finish the pending Round 09 without redoing completed Round 08. The archive also contains Round 09 unchanged. Use the new installer/guide; its old installation prose is retained solely to preserve package identity.

| Notebook | Role |
|---|---|
| `00_round08_review.ipynb` | Executed uploaded-data review; eight Plotly charts |
| `01_build_features.ipynb` | Tests, read-only prior-data gates, feature checkpoints |
| `02_run_comparison.ipynb` | Exact control replay, matched comparison, report |
| `03_saved_results.ipynb` | Nine saved-result charts; no fitting |
| `04_feature_examples.ipynb` | Executed actual-formula examples on labeled synthetic events; five charts |

Read [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for all formulas, rationale, statistical limitations and stop/continue rules. The [feature catalog](feature_catalog.json) enumerates all 48 columns. Tests cover event ordering, ties, repeated actions, time windows, source denominators, identity-safe memoization, candidate-order invariance, temporal isolation, metric arithmetic, native model replay and preservation.

Historical source is a certified **retained tail** before a fixed cutoff, not complete customer history. Missing actions are not negatives. Selected-neighbor support ratios are not population conversion probabilities or causal effects. Internal fitting scores are not private leaderboard scores. A positive adaptive-screen result requires frozen independent temporal confirmation.

No AWS, GitHub or Kaggle write is performed by this toolkit. It reads the completed Round 08 cache; data/model outputs are local to this round. Never treat a failed or partial run as an authorization to retry indefinitely.
