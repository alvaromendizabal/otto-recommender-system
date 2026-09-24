# Frontier research · evidence, mechanisms and decisions

**Latest snapshot:** September 23, 2026. The strongest verified post-competition Kaggle result is **0.57140 private / 0.57166 public** on submission **56504354**. The recorded historical private winner is **0.60503**, leaving a **0.03363** gap. Competition scores and temporal research scores remain separate evaluation settings.

Start with [01_frontier_review.ipynb](01_frontier_review.ipynb) for earlier mechanism studies, [02_training_scale_status.ipynb](02_training_scale_status.ipynb) for scale/recovery engineering, [../neural_stack/03_neural_stack_status.ipynb](../neural_stack/03_neural_stack_status.ipynb) for the dated supervised-neural design snapshot, and [04_competition_frontier.ipynb](04_competition_frontier.ipynb) for the current executed scorecard.

## September 23 score-moving frontier

The post-competition submission lineage now has three verified milestones:

- Frozen official-prefix reference: **0.56842 private / 0.56862 public**.
- Objective router, submission 56472100: **0.57100 / 0.57121**.
- Long-session cart router, submission 56504354: **0.57140 / 0.57166**. It keeps fusion clicks and selected orders, and uses selected carts only when the observed prefix has at least 21 events.

A bounded exact-threshold study qualified thresholds 14–18 on historical ledgers and submitted threshold 14 only after freezing the rule. It scored **0.57130 private / 0.57168 public**. The slight public movement did not offset the private-score regression, so the family is closed and threshold 21 remains the incumbent.

## Closed neural and ensemble branches

The trained task-conditioned neural retriever increased order candidate coverage, but the first downstream neural-aware order ranker scored **0.593918** versus **0.600683** for the incumbent on the 20,000-session selection cohort, a **-0.006766** weighted difference with **-38 order hits**. The paired 95% interval was entirely below zero. A later residual-insertion grid then achieved at most **+2 fitting-only order hits**, below its +5 gate, without opening selection labels.

A separate three-seed LightGBM order ensemble tested individual seeds, equal-score means, standardized-score means, Borda and reciprocal-rank fusion. The best fixed arm, mean-of-three, tied the incumbent at **2,306 order hits**; every other arm lost hits. No arm qualified, evaluation labels remained closed, and the branch is stopped.

## Active frontier

Two materially different capabilities remain open in this snapshot:

- **Candidate-to-session similarity stack:** adapts the third-place-style aggregation idea to the completed learned representation. It reached owner-run reserved evaluation; no final result is published here yet.
- **GPU XGBoost order stack:** introduces a different boosted-tree family and ranking objective rather than another LightGBM seed. Its first owner run completed one GPU fit but stopped at an engineering serialization-parity gate before selection. No predictive gain is claimed until the corrected continuation passes selection and reserved evaluation.

The complete first-place eight-model neural candidate ensemble and the full third-place matrix-factorization / sequence-to-sequence / broad similarity-feature stack are still not claimed as reproduced.

## Training scale: completed and bridged

The fixed 8,192→32,768-session comparison completed on 16,384 matched evaluation sessions. Weighted Recall@20 moved from **0.546284 to 0.550513**: **+0.004229**, with a paired 95% interval of **[-0.000352, +0.008903]**. Both time halves and all objectives improved in point estimates, but the interval crossed zero, so the preregistered promotion gate did not pass.

A retrospective same-session bridge then compared the 32,768-session challenger with an archived, established 100,000-session pipeline from the same corpus lineage. The established pipeline scored **0.563622** versus **0.550513** for the challenger, a **-0.013109** difference for the pilot with a paired interval entirely below zero. The research decision is therefore **stop pilot expansion and return to the established pipeline**.

## Current frontier: supervised sequence retrieval

The active experiment independently adapts a first-place-style task-conditioned sequence encoder and hard-negative contrastive objective. It preserves the incumbent 400 candidates, appends up to 200 neural order candidates, and fits one fixed neural-aware order ranker on the established 100,000-session fitting universe. Click/cart routing stays fixed. The frozen downstream roles are 100,000 fit, 20,000 selection and 432,492 evaluation sessions.

The public [reproduction matrix](../neural_stack/reproduction_matrix.json) distinguishes mechanisms that are adapted and evaluated from those merely prepared or still missing. In particular, the complete eight-model winning neural candidate ensemble and TheoViel's full matrix-factorization / sequence-to-sequence / XGBoost stack are **not** claimed as reproduced. The real neural run has no published selection/evaluation gain yet.

## Completed mechanism studies

| Mechanism | Measured evidence | Decision |
| --- | --- | --- |
| Timing with stronger task-specific ranking | Exploratory positive effect; later frozen hybrid did not confirm | Do not promote the hybrid |
| Three two-hop graph paths | Candidate coverage improved; achieved recommendations regressed | Do not submit unchanged rankers |
| Candidate-aware training and 18 path features | Primary recipe regressed | Stop tested recipe |
| 64-dimensional graph factorization and 16 affinity summaries | Small uncertain negative effect; order recall declined | No demonstrated improvement |
| Nested training scale | +0.004229 point gain over pilot control; interval crossed zero; established 100k bridge stronger | Stop pilot expansion |
| Neural-aware order ranking | -0.006766 weighted selection gain; -38 order hits | Stop tested reranker |
| Sparse neural residual insertion | Best +2 fitting-only order hits vs +5 gate | Stop residual policy family |
| Cart threshold refinement | Threshold 14: 0.57130 private / 0.57168 public vs threshold-21 incumbent 0.57140 / 0.57166 | Close threshold family; keep threshold 21 |
| Three-seed LightGBM order ensemble | Best fixed arm tied baseline; all others lost hits | Stop same-family seed bagging |

Compare point scores only within each matched study. Reported intervals are descriptive paired session-bootstrap intervals, not corrections for adaptive research or training-seed variability. Public totals reproduce point scores but not bootstrap distributions. The corpus was previously explored. The pilot control is not the accepted 102-feature submission model.

## Readable implementations and attribution

[retrieval.py](retrieval.py) contains label-blind propagation and fixed-budget replacement. [path_features.py](path_features.py) contains 18 path signals. [latent_features.py](latent_features.py) contains the independent approximate factorization, staged array checkpoints and 16 affinity summaries. [contracts.py](contracts.py) supplies local integrity helpers. [source_manifest.json](source_manifest.json) records original and published hashes.

These are selected executed method kernels with package-relative imports, not the entire private orchestration environment. Their substantive function definitions are unchanged from the recorded sources. Point-in-time correctness also depends on the historical inputs and runner contracts, not just a matrix passed to a function.

[TheoViel's third-place writeup](https://github.com/TheoViel/kaggle_otto_rs) motivates item similarities, event/position weighting and aggregation. The staged first-place contract informed ranking schedules and the bounded multi-hop adaptation. These references motivate mechanisms; their scores do not transfer to this project. The independent graph factorization is not Word2Vec, implicit ALS or NetMF. A complete winning ensemble and learned-candidate stack are not established by these studies.

## Training-scale repair and recovery

The initial attempt passed only three arguments to a five-argument replay function. The [recorded patch](training_scale_replay.patch) passes the complete resource tuple and output path. A signature-enforcing regression catches the old call. The latest owner run passed the real eight-session candidate/feature replay and completed all three model fits, establishing that the original integration blocker was cleared.

Its later stop was `PAUSED_CHECKPOINTED` at the predefined time boundary. A new local synthetic interruption test pauses this same worker after one evaluation chunk, resumes with fitting forbidden, and verifies that only the missing chunk is generated while existing binary checkpoints remain unchanged. The unchanged handoff also passes all 47 local tests. These engineering tests do not replace the pending real-data evaluation.

## Inspect and reproduce the public review

The notebooks contain their executed outputs, inline Plotly payloads, static fallbacks and post-figure sentinels. Open them in the documented [analysis environment](../../docs/REPRODUCIBILITY.md). The review workflow re-executes both notebooks without private data or model fitting. The original review can also be replayed from the repository root:

```bash
python research/frontier/review.py --execute
python -m pytest tests/test_frontier_publication.py tests/test_training_scale_publication.py -q
```

## Publication boundary and next decision

Only aggregate evidence, source identities, selected code and executed interpretation are public. Raw events, row-level targets and predictions, cohort IDs, full models, embeddings, environments and account logs remain private. Publication does not pull, reset or migrate the pinned AWS runtime.

The next decisions come from the still-open similarity-stack and XGBoost branches, not from reopening stopped neural residual, threshold, or same-family seed-bagging recipes. The strongest verified post-competition submission is **0.57140 private / 0.57166 public**, reference **56504354**. Any new model must pass its frozen local gates before competition inference or upload.
