# Experiment ledger

## Measured research stages

| Stage | Data / comparison | Result and evidence |
|---|---|---|
| Earlier exploratory ranker | 30 features; 100 candidates; 103,468 Fold 0 sessions | 0.373086 → 0.497317 weighted Recall@20; [notebook 08](../notebooks/08_ranking_evaluation.ipynb) |
| New temporal corpus | 217 verified event partitions; 100,000 fit / 20,000 selection / 432,492 evaluation sessions | Explicit future-label timestamps and event indices; [contract](../reports/research/temporal_contract.json) |
| Broad feature screen | 1,482 formulas; 514,013 rows; three grouped fitting folds | 128 shortlisted; every rejection and diagnostic recorded in the [catalog](../reports/research/feature_catalog.csv) |
| Matched model selection | Eight configurations × three objectives | `without_source` wins all tasks; 102 final features; [24 model records](../reports/research/ablation_models.csv) |
| Reserved evaluation | Same complete 400-candidate pools and 432,492 sessions | Selected 0.584392, compact 0.564904, fusion 0.535244; [evaluation](../reports/research/evaluation.json) |
| Independent reconstruction | All fit/selection/evaluation prefixes and targets | 788,883 labels, zero differences; 4,608 primary and 6,912 supporting metric checks match; [audit](../reports/research/audit.json) |
| Frozen seed replications | Three model seeds; same reference cohort, candidates, screen and stopping rule | Selected-minus-compact gains +1.949 / +2.022 / +2.031 pp; all retain 102 features; [comparison and audit links](ROBUSTNESS.md#three-verified-reference-seeds) |
| Interpretation and feature cost | 1,000 selection sessions; 4,096 SHAP rows; matched warm query timings | SHAP additivity passes; selected p95 6.42 ms versus broad 22.10 ms; [report](../reports/research/interpretation.json) |
| Competition inference | Frozen selected weights, refreshed training history, actual full notebook execution | 1,671,803 sessions; 5,015,409 rows; 1,633 parts; [full notebook receipt](../reports/research/competition_notebook_execution.json) |

No earlier score is silently reused as a matched baseline for the new temporal cohort.
The feature-family comparisons select a model; the reserved evaluation measures it.

## Execution and recovery

The controlled-study source is commit `3325a369801dd43f240f908f7d12e997fd4da706`,
with source bundle SHA-256
`cc4acc5a7fcad37c1b2faab2e79f1abcbeadce15e4bca652440eb0b6ad65e978`.
The [managed study record](../reports/research/managed_study.json) preserves interpreter,
lockfile, seed, limits, worker/thread counts, source identity and exact stage timestamps.

The initial processing launch failed during bootstrap because the container supplied
an inherited virtual-environment path. No model fit ran in that launch. The bootstrap
now explicitly creates the project environment and clears the inherited environment;
regression tests cover both study and delivery jobs. The corrected study completed.

Local historical aggregation initially exceeded available memory/spill space. Completed
artifacts were retained, resource settings were adjusted, and the graph completed. The
managed delivery workflow recomputes its deployment graph directly from existing owned
S3 training/test inputs, with 16 history threads and a 64 GiB aggregation memory limit.
It keeps deployment and research histories separately identified.

Transferred inputs are checked against size and SHA-256 before use. One set of local
read-only verification downloads contained incomplete responses; size validation and
fresh downloads repaired them before the audit. No incomplete file was certified.
Native models, graph parts, metrics and prediction parts are checkpointed to S3.

## Compute accounting

[Machine-readable job accounting](../reports/research/execution_ledger.json) records
observed processing start/end times. The verified us-west-2 processing price was
**$3.4272/hour** for `ml.c7i.16xlarge`, effective September 1, 2026. The 4-hour maximum
runtime gives a **$13.7088 instance-compute limit per launched job**, before storage,
requests and transfer. These are estimates, not AWS invoice totals.

The failed bootstrap used 189.641 reported processing seconds; the successful study
used 2,135.336. The internal study pipeline took 1,965.692 seconds, including fitting,
evaluation and durable publication; provisioning and bootstrap explain the difference.
The completed delivery job used 2,821.022 reported processing seconds, about
$2.686 in instance compute. These three launches total approximately
$4.899; this is not the lifetime cost of the earlier neural experiments.

## Publication contract

Source changes and experimental artifacts are committed through reviewable Git history.
CI executes the canonical notebooks, rejects errors and warnings, verifies dependency
and input fingerprints, proves reuse, and publishes actual outputs with receipts.
The final release is ready only when its exact published head passes the project,
neural and notebook gates and the managed prediction run has a verified completion.
