# Project scope and completion

The reference experiment is complete. The first full batch output is invalidated
because its source included future test events; replacement inference is required. The project is
now in **temporal validation and release preparation**. The active work repeats
the frozen research procedure, then delivers a traceable submission collection
and a final employer-facing release.

The [research progress snapshot](../reports/robustness/progress.json) and
[verified comparison](../reports/robustness/comparison.json) are the sources for
completion counts. A launched job, generated model, or attractive chart does
not count as a verified research result.

| Milestone | Evidence |
|---|---|
| Event processing and temporal protocol | 216.7M training events; 217 verified partitions; disjoint chronological roles |
| Retrieval baselines | Executed notebooks 02–03 and published source/budget measurements |
| Neural retrieval and ANN | Executed notebooks 05–06 with original scope and exact/approximate comparisons |
| Exploratory task-specific ranking | Executed notebook 08; matched Fold 0 baseline and recovery evidence |
| Broad feature engineering | 1,482 explicit formulas and fitting-only screening diagnostics |
| Controlled feature comparisons | Eight configurations, 24 model fits, immutable pre-evaluation model selection |
| Reserved temporal evaluation | 432,492 sessions; complete denominators; paired intervals and failure slices |
| Independent audit | Exact event reconstruction, native model hashes and 4,608 sampled replay checks |
| Interpretation and feature cost | TreeSHAP, grouped permutation, matched warm computation measurements |
| Employer-facing research narrative | Notebook 09, README, model card, catalog and experiment ledger |
| Full competition prediction | Historical execution is retained as incident evidence; official-prefix replacement is pending |
| Notebook publication | CI commits verified executed outputs and a matching execution manifest |

## Remaining release work

| Workstream | Current evidence | Completion requirement |
|---|---|---|
| Frozen robustness study | Four of nine cells audited: all reference seeds and early seed 20260908 | All nine planned window/seed cells audited, with every outcome retained |
| Valid Kaggle submission | First score invalidated by confirmed test-target contamination | Regenerate using official truncated prefixes, validate, submit and record the replacement score |
| Submission collection | Zero valid competition files after source invalidation | 50 distinct validated files, each linked to its model or ensemble recipe and content hash |
| Research narrative | Executed notebooks, Plotly figures, reference model card and case study | Update conclusions using the completed temporal comparisons and show the remaining limitations |
| Final release | Passing CI and a reproducible reference pipeline | Publish an identified release with consistent executed notebooks, artifact index, reproduction commands and a concise review path |

The early window's **seed 20260908 is complete and audited**. Early seed
20260909 started at 20:17 UTC on September 9. At 20:36 UTC the managed remaining
queue started early seed 20260910 and middle seed 20260908 concurrently. The
other middle seeds depend on the first middle seed's independent audit; each
training run has its own audit step. Shared preparation is reused without
simultaneous writers. A cloud completion counts only after its audit is checked.

### Concurrency and terminal monitoring

The user requested concurrent execution on September 9. This operational
amendment permits three simultaneous pipeline steps, within the verified AWS
quota of five instances. It leaves the frozen data, feature, model, selection,
seed and evaluation contracts unchanged. The original early job and replacement
inference are separate bounded jobs. The [saved execution](../reports/robustness/batch/execution.json)
and [plan](../reports/robustness/batch/plan.json) identify the actual queue.

From an updated clone with the project AWS credentials configured:

```bash
uv run --frozen --extra cloud python scripts/robustness_status.py --watch
```

The command is read-only, reports actual SageMaker job names, and refreshes every
30 seconds. It does not launch duplicate jobs. To inspect the saved observation
offline, use `--snapshot reports/robustness/batch/execution.json`.

### Remaining time and bounded execution

The remaining five validation cells were estimated at **5–7 hours sequentially**.
The managed queue should reduce the remaining validation and audit work to
approximately **2–3 hours from its 20:36 UTC start**, allowing for the middle
window preparation dependency. This is an estimate based on completed runs,
not a completion guarantee. Source-corrected inference should take roughly
**45–60 minutes once launched**, based on the previous full execution, and can
run alongside validation. Notebook publication and release checks add time.

The original 0.93583 private score cannot measure competitiveness: its queries
contained future events. [The source audit and replacement workflow](INFERENCE.md)
explain the correction. None of the validated training-only research changes.

The 50-file collection has no measured end-to-end runtime yet. Its next bounded
step is a small distinct-output pilot, followed by a measured batch estimate. The
existing full inference took about 41 minutes; multiplying that by 50 would ignore
the intended reuse of retrieval and feature work. Completion of the current product
scope means nine audited cells, 50 distinct validated files, and a tagged release
with consistent executed notebooks and artifact links.

## Optional work after the release

New neural sources within this protocol, sequence models, alternative rankers,
new blending studies and online experiments remain possible extensions. They
are not required to finish the currently defined research and delivery scope.
