# Frontier research · evidence, mechanisms and decisions

**Snapshot:** September 21, 2026 Pacific / September 22, 2026 02:00 UTC. This is a review of completed owner-run experiments, not a new training run.

Open [01_frontier_review.ipynb](01_frontier_review.ipynb) for the primary reading path. It reconstructs all reported point scores from aggregate hit counts and complete denominators in [evidence.json](evidence.json), then visualizes matched gains and the retrieval/ranking distinction. Paired intervals come from the returned per-session analyses; aggregate totals cannot reproduce their bootstrap distribution. Different cohorts must not be compared as a longitudinal score curve.

## Research sequence

**Timing × training:** a fixed winner-informed training package changed how timing features behaved on an exploratory fitting cohort. The point gain was positive but uncertain. An objective-specific hybrid was frozen afterward and failed the later test. This is evidence against promoting that hybrid, not against all timing features.

**Two-hop retrieval:** three normalized graph paths (time→time, time→cart, cart→order) improve target coverage while preserving the 400-candidate budget. Frozen rankers nevertheless lose achieved recall. Candidate-aware retraining and 18 explicit path signals also regress in the tested setup. The candidate-replacement recipe was stopped rather than submitted.

**Latent affinity:** a 64-dimensional approximate factorization of the historical time graph adds 16 candidate/session similarity summaries on the unchanged original pool. It produces an inconclusive small negative difference and lower order recall. It is not an implementation of Word2Vec, implicit ALS or NetMF.

**Training support:** the next fixed comparison nests 8,192 fitting sessions inside 32,768, retains the original 498,404 sampled rows, reuses three control models and allows three new models. No scale-up score exists yet: the first attempt stopped at the replay interface.

## Leading-method attribution and scope

The [third-place first-hand description by TheoViel](https://github.com/TheoViel/kaggle_otto_rs) motivates item-to-item similarities, event/position weighting and candidate/session aggregation. The staged first-place contract informed ranking schedules and a bounded multi-hop adaptation. These references motivate mechanisms; their leaderboard scores do not transfer to this implementation.

| Mechanism | Current evidence | Scope |
| --- | --- | --- |
| Stronger task-specific ranking schedule | Tested in gold-ranker study | Adapted, not full winning-pipeline reproduction |
| Multi-hop co-visitation | Coverage gain; achieved ranking loss | Independently implemented bounded adaptation |
| Candidate-aware sampling and path features | Tested; primary recipe rejected | Not deployed |
| Learned graph affinity | Tested; no robust gain | Independent graph factorization, not Word2Vec/ALS |
| Full winning ensemble and learned candidate stack | Not established by these returned studies | Do not describe staged repositories as validated reproduction |
| Training-scale comparison | Source corrected; AWS rerun pending | No result or promotion |

## Readable implementation snapshots

[retrieval.py](retrieval.py) contains label-blind propagation, fixed-budget replacement and feature reconstruction. [path_features.py](path_features.py) contains all 18 path features. [latent_features.py](latent_features.py) contains the approximate factorization, staged array checkpoints and all 16 affinity features. [contracts.py](contracts.py) contains their small local integrity helpers.

The substantive function definitions are identical to those in the executed handoff sources; package imports were made relative. [source_manifest.json](source_manifest.json) records original and published hashes. These are **selected method snapshots, not the complete owner-run orchestration environment**. They accept historical graphs or prefix arrays supplied by the caller. Correct prediction-time cutoffs are enforced by the private runner and are not inferred from a naked matrix. Original pipeline code and earlier manual research remain elsewhere in this repository.

## Reproduce the public review

No AWS account or competition data is needed to read the saved notebook. Its code uses the existing analysis stack documented in [the reproducibility guide](../../docs/REPRODUCIBILITY.md). From the repository root, using an interpreter with that analysis stack:

```bash
python research/frontier/review.py --execute
python -m pytest tests/test_frontier_publication.py -q
```

Alternatively open the notebook with the analysis kernel and run all cells. The charts use inline Plotly MIME payloads and SVG fallbacks built from the same validated values. The review does not refit models or access account data. The original research bootstrap intervals are reported, not newly inferred from aggregate counts.

## Training-scale repair

The returned call passed three arguments to `replay(d, engine, matrix, external, out)`. `setup_features(d)` actually returns all three resources: engine, matrix function and external-candidate adapter. The correction is:

```python
resources_cache = setup_features(d)
replay(d, *resources_cache, out)
```

The real replay writes the receipt only after exact candidate, source-rank, graph and 134-feature checks. The old caller's synthetic success receipt is removed. A signature-enforcing `create_autospec` replaces the permissive `lambda *args` test double. The strengthened existing end-to-end test fails on the old call and all 47 tests pass after correction. The [patch](training_scale_replay.patch) records the change; the owner-run executable has separate immutable source identity.

The failed attempt lasted 59.573 seconds, verified 498,404 old rows and fitted zero new models. It does not measure whether larger training support helps. The corrected handoff has passed local synthetic and notebook tests; a real AWS rerun remains pending. Local LightGBM was 4.6.0; the pinned owner environment reports 4.7.0, so real input/model replay remains mandatory.

## Publication boundary

Only aggregate metrics, compact source identities, documentation, selected implementations and the executed review are published here. Raw events, labels, session-level predictions, cohort IDs, full checkpoints, embeddings, environments, credentials and account logs remain in AWS. Public source publication does not pull, reset, migrate or overwrite the separately pinned runtime.

## Next decision

Run the corrected, bounded training-scale comparison using preserved caches. Require at least +0.003 weighted recall, nonnegative gains in both time halves, no pooled order loss, adequate order support and a positive lower paired confidence bound. Even a pass establishes a gain over the pilot, not the submitted 102-feature system. A matched representative comparison with that submitted pipeline and full official-prefix inference are still required before claiming a new competition improvement.
