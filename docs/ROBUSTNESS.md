# Does the feature gain survive time and training randomness?

The original study establishes a gain on one temporal cohort: weighted Recall@20
is 0.584392 for the selected representation, compared with 0.564904 for the
compact control. This extension tests how consistently that result repeats.
The protocol is frozen in [configs/robustness.toml](../configs/robustness.toml)
before any new replication is trained. **Replication results are pending.**

## What stays fixed

The comparison repeats the same research procedure: historical retrieval,
training-only feature screening, eight matched feature configurations, three
task-specific rankers, complete-query model selection, and reserved evaluation.
It preserves the 1,482-feature catalog, 128-feature screening cap, 400-candidate
budget, fitting-negative policy, model settings, and stopping rule. Source
hashes identify the exact research implementation.

For each temporal window, all three model seeds share the same fitting,
selection and evaluation sessions, prefix cuts, candidates, negative samples,
and screened feature schema. Only the ranker's seed changes. The cohort seed
stays at 20260908; model seeds are 20260908, 20260909 and 20260910.

LightGBM's deterministic CPU mode makes a run reproducible for a fixed seed;
it does not establish robustness to other seeds. No new bagging or feature
subsampling setting is introduced to manufacture seed diversity. Native model
hashes and predictions will show whether different seeds actually change the
learned ranking. Identical predictions count as evidence of determinism, not
additional independent results. See the [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html).

## The nine planned comparisons

All times are UTC. Boundaries are left inclusive and right exclusive.

| Window | Historical data ends | Fitting ends | Selection ends | Evaluation ends |
|---|---|---|---|---|
| Early | Aug 16, 22:00 | Aug 19, 22:00 | Aug 20, 22:00 | Aug 22, 22:00 |
| Middle | Aug 18, 22:00 | Aug 21, 22:00 | Aug 22, 22:00 | Aug 24, 22:00 |
| Reference | Aug 20, 22:00 | Aug 23, 22:00 | Aug 24, 22:00 | Aug 26, 22:00 |

These dates are in 2022. Each window has three model seeds, giving nine cells
and up to 216 native model fits: eight configurations times three objectives
times nine cells. The 24 completed reference-seed models remain reusable after
their contracts and bytes are verified.

The first batch repeats the reference window with two new seeds. Its existing
corpus, retrieval artifacts and feature caches can be reused exactly. The
earlier windows require their own historical retrievers, training-only screens
and feature caches. They must never reuse a retriever or feature screen fitted
on a later window. The selected feature names and retained family counts may
therefore differ across windows; the **selection procedure** is frozen.

All data has been available to earlier project work, and this extension was
motivated by the original result. The early and middle windows are retrospective
robustness checks, not previously unseen test sets. Their evaluation intervals
do not overlap, but their training periods share historical data. This is not
nine independent datasets or evidence of online business lift.

## What will be reported

For every cell, publish weighted and per-objective Recall@20 for the selected
pipeline, compact control and fixed fusion. Keep all eight selection-stage
ablation results, chosen families, feature counts, best iterations, model
digests, candidate coverage and runtime. Report the selected-minus-core and
selected-minus-fusion gains, including negative results.

Per-cell paired bootstrap intervals describe query-sampling uncertainty
conditional on that cell's fitted models. The bootstrap seed stays fixed across
replications. Seed ranges and temporal-window differences are reported
separately. Do not pool repeated predictions of the same session as independent
observations, or call a nine-row standard error a population confidence interval.

The portability claim is supported only if the selected-minus-core weighted
gain is positive in every completed planned cell. If it is not, identify the
specific window, objective or seed where it fails. This criterion determines
the wording of the conclusion; it must not be used to discard unfavorable
cells, alter this protocol, or retune against their evaluation labels.

Completion requires all nine cells, verified input and output identities,
resumability checks, and a readable Plotly comparison in the canonical research
notebook. A first successful job does not close this study.

## Durable execution and cost bounds

Run one new managed job at a time. The first reference replications each have a
two-hour runtime limit. Each cell owns a separate checkpoint namespace derived
from the protocol identity and cell name. Completed native checkpoints and
evaluation parts are verified before reuse. The previous study and submission
remain intact. Job status must identify the source commit, protocol, cohort,
model seed and output locations.

## Path to 50 submission files

The existing full submission is one verified artifact. The target is eventually
50 distinct, validated full-size prediction files, each traceable to a model or
ensemble recipe. Robustness evaluation comes first; there are **not 50 completed
submissions** at this stage.

After the research comparison is complete, define candidate model and ensemble
recipes using fitting/selection evidence, record their scope, and generate the
competition predictions from reusable inference caches. Validate row coverage,
three objectives per session, item IDs, top-20 uniqueness and checksums. Detect
duplicate prediction contents rather than counting renamed copies toward 50.
Retain every file's recipe, model identities, validation metrics and provenance
in a submission index. Large CSV files belong in durable object storage; GitHub
holds the compact index and verification evidence. Producing files and uploading
them to Kaggle are separate actions.
