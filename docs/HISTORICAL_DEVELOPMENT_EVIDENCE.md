# Historical shared-feature development evidence

This page preserves the missing aggregate result from [PR #50](https://github.com/alvaromendizabal/otto-recommender-system/pull/50) without restoring its stale generated notebooks and figures. **This is historical fitting-stage development evidence. Final model selection and the accepted release are unchanged.** The [evidence map](EVIDENCE_MAP.md) describes the current project.

## Result and uncertainty

The [aggregate receipt](../reports/research/early_shared_feature_fit_screen.json) records **1,024 fitting sessions and three whole-session OOF folds**, with no selection access and no feature-retention decision. Metrics are pooled objective Recall@20 and the official 0.1 / 0.3 / 0.6 weighted combination.

| Objective | Baseline | Shared representation |
|---|---:|---:|
| Clicks Recall@20 | 0.45491388044579534 | 0.46099290780141844 |
| Carts Recall@20 | 0.37058823529411766 | 0.38823529411764707 |
| Orders Recall@20 | 0.6190476190476191 | 0.625 |
| Weighted Recall@20 | 0.5280964300613863 | 0.537569879015436 |

The recorded weighted gain is **0.009473448954049668**. Fold gains are **0.003885630498533743**, **0.0009230769230769154**, and **0.027109442784151172**. The paired-session 95% bootstrap interval is **[-0.009633767735509221, 0.03073267750860371]**, from 1,000 replicates. **The interval includes zero.** Passing the historical point-estimate advancement rule justified a bounded fitting-only investigation; it did not establish stable ranking lift or authorize model promotion.

The later forward-fold pilot used a different protocol, so its score is not a direct continuation of this OOF measurement. No models, event populations, predictions or bootstrap samples were replayed for this consolidation. Reported measurements and resource observations are copied exactly from the dated source.

## Byte provenance and publication scope

- Original record: [JSON at the exact PR commit](https://github.com/alvaromendizabal/otto-recommender-system/blob/1b646730cd92322a676b9222175b23cfdd730bf8/reports/research/early_shared_feature_fit_screen.json).
- PR commit: `1b646730cd92322a676b9222175b23cfdd730bf8`.
- Experiment source commit: `95c24fb18ed69bc18648bdcf61a03b4812d23450`.
- Original file SHA-256: `16ee88ceb2d5a2f42b4a395f654252a4f0da9eeed5e92d954a63141e04370a0d`.
- Published file: `reports/research/early_shared_feature_fit_screen.json`.
- Published file SHA-256: `0cca0637a999d2bfa6b51c8394bc8d3637a99b877999238ad801cf480236c2e9`.

This is a selective public receipt, not a byte-identical copy. Deployment identifiers, private artifact locations, model and sampling settings, and obsolete next-step plans were omitted. The PR's 20 generated notebook, execution and figure changes were not copied. Original hash identities preserve provenance without certifying access to the withheld experiment state or reproducing its results.
