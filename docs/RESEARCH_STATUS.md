# Release and research status

## Confirmed release

The retained post-competition release is **0.57586 private / 0.57601 public**,
submission **56542128**. The [recorded receipt](../reports/submissions/similarity_stack_20260925.json)
binds that result to the released artifact. The controlled temporal study and the
synthetic public demo are separate evaluation populations and cannot be substituted
for this score.

Batch delivery is complete. Feature engineering and retrieval research remain open
under the [standing rules](EXECUTION_RULES.md); untested or deferred hypotheses are
not counted as completed findings.

## Inspected v27 owner return — 10 October 2026

The inspected owner return records **SUCCESS** and a completed scientific comparison,
with decision **CLOSED_SUPPORT_RANKER_NEGATIVE**. It reused **two completed ranker
checkpoints**, with **zero new fits in this invocation**. Successful execution did
not produce a qualified candidate.

The comparison used **20,000 previously exposed selection sessions**:

| Comparison | Weighted Recall@20 difference |
|---|---:|
| Candidate versus incumbent | +0.00652504 |
| Candidate versus matched control | +0.000859889 |
| Descriptive 95% interval for the matched-control difference | −0.000448933 to +0.002193071 |

The candidate failed the frozen matched-control gain and lower-bound checks. Its
improvement against the incumbent did not isolate a sufficient contribution over
the matched control. These selection sessions are not a fresh independent holdout.

Later independent confirmation and submission were **SKIPPED_SCIENTIFIC_GATE**.
No submission was created or left pending, and no new external score was produced.
The retained release above remains supported by its earlier receipt, not reverified
by this experiment.

The [sanitized execution summary](../reports/latest_execution/summary.json) records
the inspected return and evidence scope. Archive integrity was checked against
111 manifested members. Verify the committed summary with
`python tools/summarize_execution.py --check`; this validates the public evidence
contract without running private models. No completion timestamp is inferred from the dated run ID,
and this is not a claim to have inventoried every live cloud process or artifact.
Private models and full predictions are not published or independently replayed by
this summary.

## Keep historical decisions visible

- The [controlled study](../notebooks/09_controlled_feature_study.ipynb) measures
  feature value on a declared temporal population.
- The [deployment-parity correction](../research/frontier/11_validation_integrity_reconciliation.md)
  rejected a click/cart challenger after reconstructing the actual comparator.
- The [candidate-coverage diagnosis](../research/frontier/12_corrected_comparator_candidate_coverage.md)
  separates available-but-missed targets from missing candidates.
- The [earlier feature roadmap](ROADMAP.md) records its original baseline and study
  milestones; its dated experiments are not a live execution inventory.

A new release requires a distinct hypothesis, valid source/comparator contracts,
matched controls, a predeclared promotion decision, defensible independent
confirmation and verified inference. Online serving, revenue lift and unrestricted
feature-research completion are not established by this portfolio release.
