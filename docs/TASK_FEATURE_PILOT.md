# Task-specific feature screening pilot

Feature engineering remains open. This small development experiment reuses the
completed domain study's immutable candidate-feature caches.

## Hypothesis and domain rationale

Predicting the next click differs from predicting all future cart additions and
orders. A shared weighted feature shortlist can favor order signals and suppress
click/cart signals even with separate rankers. The prior domain study's opposing
task point estimates make this a concrete question, not evidence of a gain.

The third-place OTTO implementation combines candidate-to-session similarities with
position, time and action weights and fits different targets. This motivates testing
task-sensitive representations here; it does not validate our shortlist rule.
[Official task](https://github.com/otto-de/recsys-dataset/blob/main/KAGGLE.md),
[third-place implementation](https://github.com/TheoViel/kaggle_otto_rs).

## Preregistered comparison

[Exact configuration](../configs/task_feature_pilot.json). Twenty evenly spaced full
fitting partitions provide 5,120 whole sessions; eight selection partitions provide
2,048. The potentially partial final partition is omitted. Selection preserves all
400 candidates per session. This systematic partition sample spans the periods; it
is not a simple random sample. Every arm preserves candidate IDs, targets, sampled
fitting negatives and the original 102 baseline feature columns.

| Arm | Added columns | Question |
|---|---|---|
| Baseline | None | Matched small-data reference |
| Shared shortlist | At most 32, common to all tasks | Does supervised pruning help? |
| Task shortlist | At most 32 per task | Does task-specific selection help at the same budget? |

All 172 cached additions are eligible: progression (40), episodes (24), raw graph
pools (36) and normalized graph pools (72). Quality screening removes constants and
near-duplicates against columns present in the experiment. Nine binary pilots use
three session-grouped fitting folds and three targets. A task-specific column needs
positive gain in two folds. Shared screening weights normalized gains 0.1/0.3/0.6
and requires two-fold support in at least one task. Ties use the feature name.

Gain provides a shortlist, not proof of incremental utility. Pilot log loss is
conditional on sampled fitting negatives, not an official ranking metric. Three
matched LambdaRank arms measure actual pooled Recall@20 on complete selection
queries, including targets missed by retrieval. Shortlists freeze before selection
candidate values/targets are loaded. Models share sample, seed, capacity and stopping.

## Bounds and decisions

Use four CPU threads, 50 rounds per screening model, 150 rounds per ranker, 30-round
patience, 15-second heartbeats and a 900-second deadline. Models, checksums, per-session
statistics and contracts are saved incrementally. Resume must reuse completed models
without training. Altered data or contracts fail before another model fit.

Advance this exact procedure only if its point estimate beats both controls. Otherwise
do not scale it merely because execution succeeded. A negative small pilot does not
reject an entire feature family. A gain still needs a larger matched comparison, block
ablations and separately preregistered temporal confirmation. This subset of repeatedly
used development data is not a fresh test. Paired intervals are descriptive and do not
correct for repeated selection or systematic partition sampling. No automatic promotion.

## Reproduction

Download the configuration's source S3 cache manifests/contracts and selected
partitions into artifacts/task_feature_inputs/{fit_cache,selection_cache}. Recover
the verified corpus manifest.json and queries.parquet into its corpus directory.
Manifest, corpus and partition checksums are verified before training.

Run: `.venv/bin/python scripts/run_task_feature_pilot.py`.

The same command recovers completed model checkpoints. Source data remain durable in
the project bucket. The result bundle must be checkpointed before declaring this
milestone complete. The accepted submission remains unchanged.

## Completed pilot — September 10, 2026

The managed job `otto-task-feature-e57d55d8f858` is **Completed**. It ran for
269.738 processing seconds (4.50 minutes), approximately **$0.257 in instance compute**
at the previously recorded $3.4272/hour rate. The 900-second cap was $0.8568;
storage, requests, logs and transfer are excluded. The experiment itself took
83.32 seconds and training-free replay took 15.86 seconds.

| Matched arm | Click Recall@20 | Cart Recall@20 | Order Recall@20 | Weighted Recall@20 | Change vs baseline |
|---|---:|---:|---:|---:|---:|
| 102-column baseline | 0.519065 | 0.422292 | 0.665816 | 0.578084 | — |
| Shared 32 additions | 0.517539 | 0.437991 | 0.701531 | **0.604069** | **+2.599 pp** |
| Per-task 32 additions | 0.511947 | 0.441130 | 0.693878 | 0.599860 | +2.178 pp |

All arms used the same 5,120 fitting and 2,048 selection sessions. The weighted
candidate ceiling was 0.718717. The pooled denominators were 1,967 click targets,
637 cart targets and 392 order targets. The shared arm found 14 more order targets
and 10 more cart targets, but three fewer click targets than the matched baseline.

The shared-minus-baseline descriptive paired 95% gain interval is +0.837 to +4.531 pp.
Per-task minus shared is **−0.421 pp**, interval −1.448 to +0.666 pp.
These intervals condition on fitted models and this small, systematically sampled,
repeatedly inspected development cohort. They do not include training variability or
correct for repeated research selection. **0.604069 is not a Kaggle score**, and
cannot be compared with the historical winning private score as evidence of parity.

### Decision and feature attribution

The exact per-task selection hypothesis failed its requirement to beat both controls.
Do not scale that procedure. Shared pruning is promising enough for a larger matched
development comparison. It is not yet a promoted feature set.

Quality screening retained 150 of the 172 additions; the shared shortlist includes
11 funnel/state columns, six episode columns and 15 graph affinities. The graph
group includes raw, row-normalized and degree-normalized values. The aggregate gain
does not identify the individual contribution of these blocks. Fitting-fold gain
support is a screening heuristic, not an ablation or temporal stability result.

The [frozen next-stage schema](../configs/shared_feature_validation.json) preserves
the exact 102 baseline and 32 added columns. First compare only baseline and shared
on 12,800 fitting / 5,120 selection sessions, with unchanged candidates and negatives,
four threads and a 900-second hard cap. Advance to separately specified block ablations
only if the weighted point gain is at least 0.001 and order recall does not decline.
A passing point gate justifies another experiment; it is not confirmation. Then
measure funnel, episode, raw, row-normalized and degree-normalized blocks, including
their interactions, before preregistering temporal/seed confirmation.

### Verification, recovery and evidence

- Four smoke tests passed, including an end-to-end synthetic run and restart.
- Nine native rankers passed checksum/schema checks and independent pooled-metric arithmetic.
- A second real-data run disabled training: **zero training calls**, all model hashes
  and results identical. Checkpoint contracts reject changed inputs/configuration.
- Source, 34 input hashes, fitting-only screening, per-session statistics, native
  models, UTC heartbeats and results are preserved in the project S3 checkpoints.
- The local execution transport disconnected before producing useful rankers. The
  managed retry changed the execution environment and imposed a hard timeout; it
  did not start another unbounded local run.

[Results](../reports/research/task_feature_results.json) ·
[Audit and replay](../reports/research/task_feature_audit.json) ·
[Fitting-only screening](../reports/research/task_feature_screening.json) ·
[Completed run and cost receipt](../reports/research/task_feature_run.json) ·
[Launch/input contract](../reports/research/task_feature_launch.json).

Feature engineering remains open: 14 inventory families still require work, alongside
six previously covered scopes and two data-based exclusions. No new full-scale
training, final-holdout access or Kaggle submission follows from this pilot.
