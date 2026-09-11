# Feature value: candidate-relative demand and controlled family ablations

## Measured outcome

The **54-model fitting-only pilot completed in 60.490 seconds**, with native reload parity and an independent complete replay that made **zero new fitting calls**. The experiment adds 12 candidate-relative historical popularity features and tests five leave-one-family-out alternatives. It does not establish a leaderboard improvement.

| Arm | Features | Pooled weighted Recall@20 | Earlier fold | Later fold |
|---|---:|---:|---:|---:|
| baseline | 102 | 0.470574 | 0.351240 | 0.645175 |
| shared | 134 | 0.484737 | 0.379714 | 0.637782 |
| baseline_relative | 114 | 0.468411 | 0.362149 | 0.624142 |
| shared_relative | 146 | 0.486757 | 0.368805 | 0.658047 |
| without_funnel | 123 | 0.473916 | 0.356786 | 0.644218 |
| without_episode | 128 | 0.478679 | 0.369611 | 0.637419 |
| without_graph_raw | 131 | 0.479461 | 0.373058 | 0.634180 |
| without_graph_row | 124 | 0.490712 | 0.389110 | 0.638592 |
| without_graph_degree | 132 | 0.486992 | 0.373058 | 0.652421 |

Scores use the official pooled metric, not the mean of fold/session scores. All arms use the same complete 400-candidate evaluation pools, outcomes and model capacity. The two temporal validation blocks contain 512 sessions in total. Training sets are 403 and 699 sessions after a six-hour query embargo; training positive labels are censored before negative sampling at the respective validation cutoff. Selection/evaluation roles remain excluded.

## Feature conclusions

The shared 32 additions improve the pooled point estimate over the matched 102-column baseline by **+0.014163**, but the second fold regresses and the descriptive 95% interval is **[-0.012741, +0.039994]**. The earlier random-session OOF result is not directly comparable with this temporal test.

Adding candidate-relative demand to the baseline produces **-0.002163** pooled change. Adding it to the shared schema gives **+0.002020**, entirely explained by 10 additional pooled click hits; net cart and order hits are unchanged. Both comparisons have mixed fold signs and intervals including zero. **Do not scale or promote this exact relative-demand variant from this evidence.** These transformations use fixed historical counts, not the separately proposed 75 rolling as-of features.

Dropping the ten row-normalized graph additions has the strongest point estimate (**0.490712**) and improves both temporal fold point estimates. Its advantage over the full shared arm is **+0.005975**, with a sign-reversed descriptive interval **[-0.004751, +0.019238]**. This warrants a larger fitting-only confirmation, not immediate deletion. Funnel, episode and raw-graph removals reduce the pooled point estimate; degree-normalized removal has mixed fold behavior. **No feature-retention decision was made.**

## Research target and missing signals

The historical private winning score is **0.60503**, but this pilot's reused fitting subset is not comparable with that leaderboard. The winner's [primary write-up](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution) emphasizes multiple co-visitation views, multi-step retrieval, action-conditioned neural similarities, multiple-window popularity ranks and about 1,200 candidates. These are a research coverage checklist, not a promise that another feature list guarantees a record.

The candidate-oracle ceiling on this exact 400-candidate subset is **0.635088** versus best measured ranking **0.490712**. Both ranking loss inside the pool and missed targets outside it remain relevant. Next, inspect the existing retrieval inventory and stage a bounded action-conditioned/multi-hop candidate coverage and resource test; independently confirm the row-normalized ablation on more fitting sessions. Do not enlarge training simply because one small arm is best.

## Reproducibility and artifacts

Source commit: `b9f53c29297a6df08cbeabf6890e0736047f9acf`. The original immutable result SHA-256 is `b6a4f9494c3dc7be5942524928675694cdacc0200e7c30d754811444f21b6d03`. Native models, model receipts, fold membership, per-session hits, replay and audit evidence remain in the existing private S3 prefix `ranking/research/feature-value-pilot/b9f53c29297a/`. The independent audit checked 18 arm reports and 4,608 session-arm entries; the worker verified 131 persisted files. Peak RSS was **1,390.68 MiB**. No graphs were rebuilt and no packages were installed for this pilot.

The earlier worker finished the experiment but timed out waiting for publication; do not mistake that synchronization issue for failed models. Preserve its state receipt. Publication and final workspace synchronization are separate operations and need their own verified receipts.

[Executed research notebook](../../notebooks/research/feature_value_pilot.ipynb). [Machine-readable evidence](feature_value_pilot.json).
