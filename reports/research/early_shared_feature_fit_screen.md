# Fitting-only OOF shared-feature screen — measured result

The frozen 32-feature shared addition passed its preregistered **fitting-only** advancement gate on the 1,024-session Aug-16 cohort. The exact merged source commit was `95c24fb18ed69bc18648bdcf61a03b4812d23450`; selection/evaluation access was disabled.

| Metric | 102-feature baseline | 134-feature shared | Gain |
|---|---:|---:|---:|
| Weighted Recall@20 | 0.528096 | 0.537570 | +0.009473 |
| Clicks Recall@20 | 0.454914 | 0.460993 | +0.006079 |
| Carts Recall@20 | 0.370588 | 0.388235 | +0.017647 |
| Orders Recall@20 | 0.619048 | 0.625000 | +0.005952 |

Fold weighted gains were `+0.003886`, `+0.000923`, and `+0.027109`, so all three folds were nonnegative. The point-estimate rule therefore passed. However, the paired whole-session bootstrap 95% interval for weighted gain was `[-0.009634, +0.030733]`, which includes zero. This is evidence that the 32-feature block is worth carrying forward—not proof of a stable leaderboard improvement.

The run fitted exactly 18 fixed 75-round LambdaRank models, used the original fitting-negative policy (`negative_budget=60`, seed `20260908`), evaluated complete 400-candidate held-out fitting queries, took 15.406 seconds in the screen script, and peaked at 1,377.76 MiB RSS. No feature was retained or removed from these results.

## Feature-engineering decision

Feature engineering remains open. The shared block stays active while higher-leverage gaps are attacked. Existing research already shows the fixed 400-candidate frontier is a binding constraint: historical candidate-ceiling studies exceed current ranking performance materially. The next priority is broader, leakage-safe candidate generation and richer source/candidate×session aggregation—not treating this successful fitting screen as a transition to final modeling.

The strict as-of demand/support family also remains pending real-data execution.
