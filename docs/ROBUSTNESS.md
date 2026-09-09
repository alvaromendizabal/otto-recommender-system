# Does the feature gain survive time and training randomness?

The original study establishes a gain on one temporal cohort: weighted Recall@20
is 0.584392 for the selected representation, compared with 0.564904 for the
compact control. This extension tests how consistently that result repeats.
The protocol is frozen in [configs/robustness.toml](../configs/robustness.toml)
before any new replication is trained.

## Three verified reference seeds

All three reference-window seeds are verified: the original and both planned
new model seeds. Each evaluates all **432,492 reserved sessions**, after fitting
eight feature configurations and three task rankers.

| Model seed | Selected | Compact control | Candidate fusion | Gain over compact | Paired 95% interval |
|---|---:|---:|---:|---:|---:|
| 20260908 · original | 0.584392 | 0.564904 | 0.535244 | +1.949 pp | +1.840 to +2.065 pp |
| 20260909 · replication | 0.584430 | 0.564207 | 0.535244 | +2.022 pp | +1.913 to +2.135 pp |
| 20260910 · replication | 0.584988 | 0.564675 | 0.535244 | +2.031 pp | +1.921 to +2.141 pp |

**What this means:** the feature gain remains positive when only the model seed
changes. All three seeds select the 102-feature configuration that excludes direct
source features for all three tasks. The selected score ranges from 0.584392 to
0.584988, a spread of **0.060 percentage points**. The feature gain ranges from
**+1.949 to +2.031 percentage points**. These are descriptive ranges over the three
planned seeds, not population confidence intervals. They support consistency on
this time window; they do not establish stability across different periods.

The intervals describe session-sampling uncertainty for each fitted model pair.
They are not confidence intervals over training seeds. All three runs use the same
reserved sessions, so their predictions must not be treated as independent
observations. The original model remains the headline result; this comparison
does not select a new model using evaluation labels.

[Notebook 09](../notebooks/09_controlled_feature_study.ipynb) plots the scores and
paired gains in Plotly, reports all three objectives, and compares actual
sampled top-20 predictions. The [comparison data](../reports/robustness/comparison.json)
links every number to checked source files. Verification receipts for
[seed 20260909](../reports/robustness/cells/reference_seed_20260909/robustness_audit/report.json)
and [seed 20260910](../reports/robustness/cells/reference_seed_20260910/robustness_audit/report.json)
each record the full metric audit, 24 native models, eight ablations, 256-session
prediction replay and both reproduced 1,000-sample bootstrap comparisons.
The notebook reports actual ordered top-20 prediction changes against the original
model on the same fixed probe sample. Similar aggregate scores need not imply
identical recommendations; sample-level changes are kept separate from full-cohort metrics.

The [nine-cell progress snapshot](../reports/robustness/progress.json) records
which jobs have finished verification and which scores remain unmeasured.
Runtime, source commit, launch settings and durable artifact locations are saved
in the [run receipts](../reports/robustness/runs).
The final reference seed's
[execution receipt](../reports/robustness/runs/reference_seed_20260910.json)
records completed training and the separate completed verification job. Verification
reused the saved checkpoints without retraining. This closes the reference-seed
milestone. The first early-window result is described below.

## First early-window result

The early window evaluates **562,504 reserved sessions** from August 20–22, 2022
(22:00 UTC boundaries), using history ending August 16 at 22:00 UTC. Its fitting
and selection cohorts contain 100,000 and 20,000 sessions. The historical
retrieval and fitting-only feature screen were rebuilt for this window.

| Early window · seed 20260908 | Weighted Recall@20 | Gain over comparator | Paired 95% gain interval |
|---|---:|---:|---:|
| Selected representation | **0.566897** | — | — |
| Compact control | 0.545094 | **+2.180 pp** | **+2.082 to +2.286 pp** |
| Candidate fusion | 0.516428 | +5.047 pp | +4.848 to +5.234 pp |

The gain is the selected model's score minus the comparator's score. For example,
0.566897 − 0.545094 ≈ 0.021804, or **2.180 percentage points**. This is an offline
recommendation metric; it does not estimate conversion or revenue lift.

Per-action Recall@20 makes the tradeoff visible:

| Action | Selected | Compact control | Candidate fusion | Selected minus compact | Selected minus fusion |
|---|---:|---:|---:|---:|---:|
| Clicks | 0.507496 | 0.455046 | 0.526940 | +5.245 pp | −1.944 pp |
| Carts | 0.417017 | 0.396951 | 0.418439 | +2.007 pp | −0.142 pp |
| Orders | 0.651737 | 0.634173 | 0.563670 | +1.756 pp | +8.807 pp |

**What this means:** the feature gain also appears in this earlier period, and
all three objectives improve against their matched compact controls. Candidate
fusion retains higher click and cart recall. Stronger order recall drives the
selected pipeline's overall advantage over fusion under the competition's
0.1/0.3/0.6 action weights. These tradeoffs remain part of the result.

The earlier score must not be compared directly with the reference score as if
both used the same sessions. Each window has its own matched controls. All three early seeds are independently audited. Seed 20260909 reaches
**0.566784**, versus **0.545668** for its compact control, a **+2.112 percentage-point
gain** (paired 95% interval **+2.013 to +2.212**). Both use the same 562,504-session
cohort and the same fitting-only preparation; only the model seed changes. The
observed early-seed scores range from 0.566465 to 0.566897. Their intervals are
conditional on each fitted model and should not be pooled as independent cohorts.

## Completed nine-cell comparison

The selected feature procedure beats the matched compact ranker in **all nine audited runs**. Gains range from **+1.791 to +2.180 percentage points**, with all paired 95% session-bootstrap intervals above zero. Every window and seed chooses the variant without direct source-score features for all three actions.

The highest absolute offline score is **0.590759**, middle window seed **20260908**. On the reference cohort, seed **20260910** is highest at **0.584988**. Different windows use different sessions, so these are descriptive maxima. The original reference seed **20260908** remains the submission model, with **0.584392** on its reserved evaluation; no new seed is chosen using evaluation results.

| Window | Seed | Selected Recall@20 | Compact Recall@20 | Gain (pp) | Paired 95% interval (pp) |
|---|---:|---:|---:|---:|---|
| Early | 20260908 | 0.566897 | 0.545094 | +2.180 | +2.082 to +2.286 |
| Early | 20260909 | 0.566784 | 0.545668 | +2.112 | +2.013 to +2.212 |
| Early | 20260910 | 0.566465 | 0.547613 | +1.885 | +1.782 to +1.989 |
| Middle | 20260908 | 0.590759 | 0.571302 | +1.946 | +1.839 to +2.061 |
| Middle | 20260909 | 0.590745 | 0.572832 | +1.791 | +1.678 to +1.902 |
| Middle | 20260910 | 0.590465 | 0.571880 | +1.858 | +1.748 to +1.969 |
| Reference | 20260908 | 0.584392 | 0.564904 | +1.949 | +1.840 to +2.065 |
| Reference | 20260909 | 0.584430 | 0.564207 | +2.022 | +1.913 to +2.135 |
| Reference | 20260910 | 0.584988 | 0.564675 | +2.031 | +1.921 to +2.141 |

The full comparison is independently checked against the archived audit payloads, metric counts and exact input hashes. All nine planned cells are retained. The third early seed scores 0.566465; the middle seeds range from 0.590465 to 0.590759. No training or audit jobs remain for this protocol.

The original competition score was invalidated separately for future test events. Corrected official-prefix inference is complete; its full file and refreshed native replay are verified. Kaggle accepted the frozen reference model's file and returned **0.56842 private / 0.56862 public**. This separate competition evaluation does not change any temporal model-selection decision.

### How this window chose its features

The screen evaluated all **1,482 candidate formulas** on **513,889 candidate rows
from 8,448 fitting sessions**, using nine grouped fitting-only pilots. It retained
128 features before the eight controlled ablations. Selection then chose
`without_source` for clicks, carts and orders, using **101 features** for each.
The compact control used 28 retained core features.

| Family | Retained by the fitting-only screen | Used by the selected rankers |
|---|---:|---:|
| History | 50 | 50 |
| Context | 19 | 19 |
| Repeat behavior | 21 | 21 |
| Graph structure | 7 | 7 |
| Interactions | 4 | 4 |
| Retrieval source signals | 27 | 0 |
| **Total** | **128** | **101** |

The reference window selected 102 features. This difference is expected: the
protocol freezes the feature catalog and selection procedure, while each
window's fitting data determines its retained schema. The source feature family
is removed at ranking time; the historical retrieval sources still generate
the candidate pool.

| Selection-stage configuration | Features | Weighted Recall@20 on 20,000 selection sessions |
|---|---:|---:|
| Compact control | 28 | 0.546766 |
| All screened features | 128 | 0.552680 |
| Without context | 109 | 0.548917 |
| Without graph | 121 | 0.551309 |
| Without history | 78 | 0.546704 |
| Without interactions | 124 | 0.551530 |
| Without repeat behavior | 107 | 0.548005 |
| **Without source signals** | **101** | **0.570206** |

These are **selection scores**, kept separate from the reserved evaluation
above. Model choice was sealed before evaluation labels were opened. All eight
configurations and 24 native rankers are retained, including less successful
ablations.

The [early-window audit](../reports/robustness/cells/early_seed_20260908/robustness_audit/report.json)
and [execution receipt](../reports/robustness/runs/early_seed_20260908.json) link
the complete evaluation to its source bytes, feature screen, model selection and
saved native models. The separate audit reconstructs prefixes and labels from
all 217 original Parquet partitions, checks every evaluation part, replays 256
sessions, and reproduces both 1,000-resample paired intervals. The original raw
JSONL-to-Parquet conversion is not rerun by this audit.

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
times nine cells. The 72 completed reference-window models remain reusable after
their contracts and bytes are verified.

The completed first batch repeats the reference window with two new seeds. Its existing
corpus, retrieval artifacts and feature caches were reused exactly. The
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

### Preparing an earlier window

Earlier-window jobs start from the 217 original training Parquet files. The
bootstrap checks every file's size and SHA-256 and verifies the retained
conversion manifest. It does not accept the reference window's corpus,
retrieval bundle, or selected features.

For the early window, history ends on **August 16, 2022, at 22:00 UTC**. The
runner then builds the query ledger, 32 historical graph partitions, the
1,482-feature fitting cache, and nine grouped fitting-only screening pilots.
It materializes the retained features for the fitting and selection cohorts
before fitting the eight configurations and three task rankers. Evaluation
starts after model selection has been sealed.

Preparation is shared **within a window**. Model seeds get separate model and
evaluation checkpoints. A later seed can reuse the preparation only after its
contracts and all data hashes pass verification. Graph and feature-cache parts
are uploaded after their receipts close; screening pilots are published when
the screening stage finishes. The published preparation receipt is preserved
on reuse, including its original creation time and source commit.

Generate the launch from an exact committed source archive:

```bash
uv run --frozen python scripts/prepare_robustness.py \
  --cell early_seed_20260908 \
  --source-commit "$OTTO_SOURCE_COMMIT" \
  --source-sha256 "$OTTO_SOURCE_ARCHIVE_SHA256" \
  --output artifacts/early_launch/launch.json
```

The variables identify the Git commit and SHA-256 of its source archive. This
command validates and writes a launch; it does not start a cloud job. The
managed request uses one `ml.c7i.16xlarge` instance, a 100 GiB volume, and a
7,200-second limit. Source, bootstrap, and launch uploads are checked before
execution. Existing [run requests](../reports/robustness/runs) record the exact
SageMaker inputs and role.

The separate earlier-window verifier requires the original event directory:

```bash
uv run --frozen --extra ml python scripts/verify_robustness.py \
  --root artifacts/research \
  --launch artifacts/early_launch/launch.json \
  --source artifacts/train
```

It rebuilds all fitting, selection, and evaluation prefixes and labels from
those events, checks every prepared data part and screening pilot, verifies
the 24 rankers and complete evaluation counts, replays 256 sessions, and
reproduces both paired bootstrap comparisons. The original training receipt
and training log remain separate from the audit receipt and log. The auditor
cannot reuse the reference window's original-event reconstruction.

## How a completed replication becomes a verified result

`scripts/verify_robustness.py` checks the completed training identity, frozen data,
all eight selection comparisons and all 24 native models. It recomputes the
official metrics from every reserved-session statistic, replays 256 deterministic
sessions, and reproduces both 1,000-sample paired bootstrap comparisons. For additional reference-window seeds, a separate
prediction probe compares actual top-20 item IDs with the original models;
different model-file headers alone do not demonstrate prediction diversity.

For the reference window, the original event reconstruction is reusable because
its corpus and historical retrieval identities remain unchanged. Those verification
receipts record that reuse explicitly. Earlier temporal windows require their own source audit.
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

## Delivery state

All nine planned training and independent audit jobs completed successfully. Their compact results, full metric arithmetic, feature choices and paired intervals are published together. Notebook 09 and the Plotly report include every outcome.

One official-prefix full submission has passed independent validation. The final Kaggle Submit action is awaiting the approval required by automatic review. A late score, when obtained, will be recorded independently of the offline evaluation.

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
