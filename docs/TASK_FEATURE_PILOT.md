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
