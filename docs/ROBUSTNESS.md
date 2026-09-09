# Does the feature gain survive time and training randomness?

The original study establishes a gain on one temporal cohort: weighted Recall@20
is 0.584392 for the selected representation, compared with 0.564904 for the
compact control. This extension tests how consistently that result repeats.
The protocol is frozen in [configs/robustness.toml](../configs/robustness.toml)
before any new replication is trained.

## First verified replication

Two of the nine planned cells are verified: the original reference and a new
model seed on the same reference window. Each evaluates all **432,492 reserved
sessions**, after fitting eight feature configurations and three task rankers.

| Model seed | Selected | Compact control | Candidate fusion | Gain over compact | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| 20260908 · original | 0.584392 | 0.564904 | 0.535244 | +1.949 pp | +1.840 to +2.065 pp |
| 20260909 · replication | 0.584430 | 0.564207 | 0.535244 | +2.022 pp | +1.913 to +2.135 pp |

**What this means:** the feature gain remains positive when only the model seed
changes. Both seeds select the 102-feature configuration that excludes direct
source features for all three tasks. The selected model's absolute score changes
by just 0.0038 percentage points. This is evidence of consistency on this time
window; it does not establish stability across different periods.

The intervals describe session-sampling uncertainty for each fitted model pair.
They are not confidence intervals over training seeds. Both runs use the same
reserved sessions, so their predictions must not be treated as independent
observations. The original model remains the headline result; this comparison
does not select a new model using evaluation labels.

[Notebook 09](../notebooks/09_controlled_feature_study.ipynb) plots the scores and
paired gains in Plotly, reports all three objectives, and compares actual
sampled top-20 predictions. The [comparison data](../reports/robustness/comparison.json)
links every number to checked source files. The new seed's
[verification receipt](../reports/robustness/cells/reference_seed_20260909/robustness_audit/report.json)
records the full metric audit, 24 native models, eight ablations, 256-session
prediction replay and both reproduced 1,000-sample bootstrap comparisons.
The probe found different ordered top-20 lists in all 256 sampled sessions for
each task. The sets of recommended items changed in 242 click, 230 cart and
243 order cases. Thus the similar aggregate scores do not imply identical
recommendations. These counts describe the fixed probe sample.

The [nine-cell progress snapshot](../reports/robustness/progress.json) records
which jobs have finished verification and which scores remain unmeasured.
Runtime, source commit, launch settings and durable artifact locations are saved
in the [run receipts](../reports/robustness/runs).
Seed 20260910 was launched after the first seed's verification job completed;
its [execution receipt](../reports/robustness/runs/reference_seed_20260910.json)
records the separate checkpoint namespace and two-hour runtime limit.

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

## How a completed replication becomes a verified result

`scripts/verify_robustness.py` checks the completed training identity, frozen data,
all eight selection comparisons and all 24 native models. It recomputes the
official metrics from every reserved-session statistic, replays 256 deterministic
sessions, and reproduces both 1,000-sample paired bootstrap comparisons. A separate
prediction probe compares actual top-20 item IDs with the original models;
different model-file headers alone do not demonstrate prediction diversity.

The original event reconstruction is reusable only because the reference corpus
and historical retrieval identities remain unchanged. This verification records
that reuse explicitly. Earlier temporal windows require their own source audit.
An AWS verification job has a separate status receipt and a 30-minute runtime
limit; it preserves the original training status. Verified evidence can be reused
after its inputs and integrity are checked again.

To prepare this managed verification, `scripts/prepare_robustness.py` accepts
`--verify-launch` pointing to the completed training launch, plus the exact
verifier source commit and archive checksum. The original model-source identity
remains separately recorded inside the verification launch.

The compact comparison can be rebuilt without cloud access or model training:

```bash
uv run --frozen python scripts/publish_robustness_report.py
uv run --frozen python scripts/publish_robustness_report.py --check
```

This command checks the saved audit and input identities before extracting the
metrics for Notebook 09. It does not substitute for the full managed verification.

## Remaining milestones

1. Complete and verify seed 20260910 on the reference window, giving three
   reference seeds with fixed cohorts, candidates and screened features.
2. Build historically valid corpus, retrieval and screening artifacts for the
   early and middle windows. Run all three seeds on each, then audit every cell.
3. Publish the complete nine-cell comparison, including unfavorable outcomes,
   per-objective tradeoffs, seed ranges and temporal differences. Update the
   model card and portfolio conclusions to match the full evidence.
4. Produce the distinct submission artifacts described below, then publish a
   release with executed notebooks, reproduction commands and a concise review path.

The project already has an audited reference experiment and a complete batch
submission. The robustness extension and 50-file submission collection are
separate unfinished milestones. No portfolio rating substitutes for those checks.

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
