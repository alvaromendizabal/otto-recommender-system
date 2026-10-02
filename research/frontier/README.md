# Frontier research · evidence, mechanisms and decisions

**Latest snapshot:** October 2, 2026. The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public** on submission **56542128**. The recorded historical private winner is **0.60503**, leaving a **0.02917** gap. The major first-place neural variants are now materially covered, order-side source/QueryFormer recipes are closed under fitting-only gates, and the strongest new result is a **+0.003535 click/cart fitting gain with 5/5 positive folds**, frozen for independent selection. No newer submission is claimed.

Start with [01_frontier_review.ipynb](01_frontier_review.ipynb) for earlier mechanism studies, [02_training_scale_status.ipynb](02_training_scale_status.ipynb) for scale/recovery engineering, [../neural_stack/03_neural_stack_status.ipynb](../neural_stack/03_neural_stack_status.ipynb) for the dated supervised-neural design snapshot, [04_competition_frontier.ipynb](04_competition_frontier.ipynb) for the September 23 scorecard, [05_similarity_attention_frontier.md](05_similarity_attention_frontier.md) for the promoted similarity result, and [06_transition_model_frontier.md](06_transition_model_frontier.md) for the September 29 transition/model-family frontier, and [07_candidate_model_frontier.md](07_candidate_model_frontier.md) for the October 1 candidate/model closeouts, then [08_neural_objective_frontier.md](08_neural_objective_frontier.md) for the completed neural sweep, source/sequence closeouts and fitting-qualified click/cart challenger.

## September 28 score-moving frontier

The post-competition submission lineage now has three verified milestones:

- Frozen official-prefix reference: **0.56842 private / 0.56862 public**.
- Objective router, submission 56472100: **0.57100 / 0.57121**.
- Long-session cart router, submission 56504354: **0.57140 / 0.57166**. It keeps fusion clicks and selected orders, and uses selected carts only when the observed prefix has at least 21 events.
- **Neural-similarity orders, submission 56542128: 0.57586 / 0.57601.** It preserves the prior click/cart lists and replaces orders with the promoted 137-feature similarity ranker.

A bounded exact-threshold study qualified thresholds 14–18 on historical ledgers and submitted threshold 14 only after freezing the rule. It scored **0.57130 private / 0.57168 public**. The slight public movement did not offset the private-score regression, so the family is closed and threshold 21 remains the incumbent.

## Closed neural and ensemble branches

The trained task-conditioned neural retriever increased order candidate coverage, but the first downstream neural-aware order ranker scored **0.593918** versus **0.600683** for the incumbent on the 20,000-session selection cohort, a **-0.006766** weighted difference with **-38 order hits**. The paired 95% interval was entirely below zero. A later residual-insertion grid then achieved at most **+2 fitting-only order hits**, below its +5 gate, without opening selection labels.

A separate three-seed LightGBM order ensemble tested individual seeds, equal-score means, standardized-score means, Borda and reciprocal-rank fusion. The best fixed arm, mean-of-three, tied the incumbent at **2,306 order hits**; every other arm lost hits. No arm qualified, evaluation labels remained closed, and the branch is stopped.

## Current frontier

The candidate-to-session similarity branch remains the **promoted competition incumbent**: its 137-feature order ranker added **+0.004396 weighted Recall@20 and +508 order targets** on all 432,492 reserved temporal sessions, then scored **0.57586 private / 0.57601 public** on submission 56542128.

Post-incumbent research has now materially covered the remaining first-place neural mechanisms. v27/v29 produced useful candidate diversity but missed the frozen order gate; their stable-pool similarity stack also failed. v21/v23 added smaller incremental coverage, while v15/v18 regressed in the equal-budget pool. These models remain documented representation studies rather than promoted components.

The strongest 1,200-candidate order pool contains **2,628 fitting order targets versus 2,238 achieved baseline hits**, yet neither flat source-aware fusion nor a QueryFormer-style field-sequence interaction converted that oracle headroom into stable order gains. The source400 arm tied aggregate order hits; source1200 lost 40. A nonlinear field control gained +8 orders on 400 candidates but missed its stability/gain gate, while the explicit cross-attention arms regressed. These order-side recipes are closed.

The current promotion candidate comes from shifting attention to clicks and carts. A 589-feature source-aware ranker on the established 400-candidate pool improved clicks by **+186 hits / +0.000961 weighted contribution** and carts by **+53 hits / +0.002574**, for **+0.003535 combined official-metric contribution**. All five chronological folds were positive and the worst fold remained +0.001635. The expanded 1,200-candidate variants regressed.

Decision: **freeze the source400 click/cart winner and open the independent selection stage once.** Selection and reserved-evaluation labels remain unopened in this snapshot. See [08_neural_objective_frontier.md](08_neural_objective_frontier.md).

## Training scale: completed and bridged

The fixed 8,192→32,768-session comparison completed on 16,384 matched evaluation sessions. Weighted Recall@20 moved from **0.546284 to 0.550513**: **+0.004229**, with a paired 95% interval of **[-0.000352, +0.008903]**. Both time halves and all objectives improved in point estimates, but the interval crossed zero, so the preregistered promotion gate did not pass.

A retrospective same-session bridge then compared the 32,768-session challenger with an archived, established 100,000-session pipeline from the same corpus lineage. The established pipeline scored **0.563622** versus **0.550513** for the challenger, a **-0.013109** difference for the pilot with a paired interval entirely below zero. The research decision is therefore **stop pilot expansion and return to the established pipeline**.

## Sequence-retrieval lineage

The first sequence experiment independently adapted a first-place-style task-conditioned encoder and hard-negative contrastive objective. Its v42-derived representation ultimately became useful as similarity evidence and was promoted. A later complementary attention encoder was trained and evaluated under fitting-only candidate-aware cross-validation; its candidate union did not qualify and is closed. The frozen downstream roles remain 100,000 fit, 20,000 selection and 432,492 evaluation sessions.

The public [reproduction matrix](../neural_stack/reproduction_matrix.json) distinguishes mechanisms that are adapted and evaluated from those still missing. Two complementary neural encoder families have now been independently adapted; broader source/transition features and an XGBoost ranking family are also tested. The complete eight-model winning neural candidate ensemble and TheoViel's full matrix-factorization / sequence-to-sequence candidate stack are **not** claimed as reproduced.

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
| Complementary attention candidate union | Best base arm +7 order hits / +0.000235; union arms regressed | Stop tested union; preserve encoder as feature source |
| Dense neural interaction + score stack | Best arm +7 order hits / +0.000235; below +0.0015 gain gate | Stop dense-stack family |
| Transition/source representation | Best arm +37 order hits / +0.001241; stable but below +0.0015 gain gate | Preserve representation; test complementary model family |

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

The transition-source XGBoost reserved evaluation is complete and closed; the later MF/W2V/Seq2Seq/broad-feature CPU studies are also frozen under their recorded decisions. The next decision comes from the bounded v29/v27 GPU frontier, not from reopening stopped tree, retrieval or threshold recipes. The strongest verified post-competition submission remains **0.57586 private / 0.57601 public**, reference **56542128**. No new competition inference or upload is justified until a new model clears fitting, selection and reserved evaluation gates.
