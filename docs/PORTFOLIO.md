# Portfolio review guide

## A five-minute review

Start with the [README](../README.md), then
[Notebook 09](../notebooks/09_controlled_feature_study.ipynb). It shows the research
question, matched controls, the winning negative ablation, confidence intervals,
feature explanations, compute cost and failure slices. The headline result is
0.58439 weighted Recall@20 on 432,492 temporal evaluation sessions.

[Notebook 10](../notebooks/10_competition_inference.ipynb) demonstrates executable
native-model predictions. Its default mode verifies a compact example against the
full run; full-data mode calls the same inference implementation. Reviewing saved
outputs requires no dataset download or AWS account.

## Where to inspect each capability

| Capability | Concrete evidence |
|---|---|
| Scientific judgment | [Controlled feature study](../notebooks/09_controlled_feature_study.ipynb): source features hurt, fusion still wins two tasks, and uncertainty is scoped correctly |
| Feature engineering | [1,482 formula catalog](../reports/research/feature_catalog.csv): availability, ranges, correlations, grouped-fold stability and rejection reasons |
| Evaluation integrity | [Independent audit](../src/otto_recsys/research/audit.py): original events → targets, complete denominators, all native models and sampled prediction replay |
| Ranking | [Training](../src/otto_recsys/research/training.py) and [selection](../src/otto_recsys/research/study.py): task-specific LambdaRank, official-metric stopping, immutable selection seal |
| Deep learning | [Two-tower notebook](../notebooks/05_two_tower_results.ipynb) and [neural package](../gpu/two_tower): objective conditioning, hard negatives, mixed precision and checkpoint recovery |
| Retrieval and ANN | [ANN benchmark](../notebooks/06_ann_benchmark.ipynb): exact/approximate fidelity, candidate coverage, latency and paired comparisons |
| Data engineering | [Temporal corpus](../src/otto_recsys/research/protocol.py): 217 verified partitions, chronological roles, complete query ledgers and timestamped targets |
| MLOps | [Managed jobs](../src/otto_recsys/cloud), [checkpoint storage](../src/otto_recsys/cloud/research_checkpoints.py), and [CI](../.github/workflows/ci.yml) |
| Communication | [Model card](MODEL_CARD.md): clear intended use, limitations, lineage and deployment/evaluation distinction |

## Read deeper in order

Notebooks 01–04 explain validation, retrieval baselines, candidate budgets and
hard-negative quality. Notebooks 05–06 preserve the neural retrieval experiment.
Notebooks 07–08 cover the earlier exploratory ranking baseline. Notebook 09 is the
new controlled feature study; Notebook 10 closes the operational path to predictions.
The earlier scores retain their original cohorts and should not be compared directly
with the new temporal evaluation.

The project demonstrates measured ML work and reproducible engineering. It does not
claim novel model architecture, state of the art, production revenue lift, or completed
experiments that appear only as future ideas in the roadmap.
