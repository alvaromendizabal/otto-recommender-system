# Frontier research · evidence, mechanisms and decisions

**Latest snapshot:** September 21, 2026, 8:09 PM Pacific / September 22, 2026, 03:09 UTC.

Start with [02_training_scale_status.ipynb](02_training_scale_status.ipynb). Three larger-training rankers are complete and sealed; 249 of 256 evaluation chunks are committed. A bounded deadline pause leaves 448 sessions to score. There is no final scale-up result or promotion decision. [Aggregate status](training_scale_status.json)

[01_frontier_review.ipynb](01_frontier_review.ipynb) and [evidence.json](evidence.json) preserve the earlier dated snapshot: five completed scoring studies and the initial pre-training failure. Their references to a pending rerun describe that earlier snapshot, not the current checkpoint state. Both notebooks are readable without AWS or private competition data.

## Current training-scale comparison

The original 8,192 fitting sessions are nested in 32,768 sessions. All 498,404 original sampled rows are retained inside 1,993,202 rows. Candidate generation, 134 features, negative sampling, seed and task-specific training schedules are unchanged. Three pilot control models are reused and three larger-training models are complete.

Order queries containing both positive and negative candidates increase from 685 to 2,644. This passes the predefined training-support requirement; it does not establish a recall gain. Evaluation completed 15,936 of the fixed 16,384 sessions before the worker reached its deadline. All 249 returned scoring chunks passed identity, hash and count-bound checks. All 922 included inventory entries passed checksum verification. Private model receipts agree with the returned seals and omitted-artifact inventory; native weights are reverified by the AWS runner rather than downloaded for this review.

The next owner invocation uses the exact same handoff and science ID. It reuses completed models and feature chunks, finishes seven evaluation chunks, aggregates the entire cohort, calculates the fixed paired interval and decision, and generates the final result notebook. Cached training arrays may be reread for integrity checks. No new hyperparameters or cohort selection are introduced.

## Completed mechanism studies

| Mechanism | Measured evidence | Decision |
| --- | --- | --- |
| Timing with stronger task-specific ranking | Exploratory positive effect; later frozen hybrid did not confirm | Do not promote the hybrid |
| Three two-hop graph paths | Candidate coverage improved; achieved recommendations regressed | Do not submit unchanged rankers |
| Candidate-aware training and 18 path features | Primary recipe regressed | Stop tested recipe |
| 64-dimensional graph factorization and 16 affinity summaries | Small uncertain negative effect; order recall declined | No demonstrated improvement |
| Nested training scale | Models complete; full evaluation pending | Resume seven remaining chunks |

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

Finish the original fixed evaluation. Advancement requires at least +0.003 weighted recall, nonnegative time-half changes, no pooled order loss, sufficient support and a positive lower paired confidence bound. A pass still requires a matched representative comparison with the actual submitted pipeline before a new competition-improvement claim. The authenticated submission history remains 0.56842 private / 0.56862 public, reference 56132573. No new submission was made.
