# Review the completed OTTO delivery

**The delivered system is complete:** a controlled offline study, a validated full-population
batch prediction, a scored post-competition release, executed notebooks, and automated
quality checks. Further candidate-retrieval research remains open and does not change the
released model. The published aggregate evidence is dated **October 6, 2026, 02:15 UTC**.

## Start with working software

From the repository root, using Python 3.11+:

```bash
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo --check
```

Open `/tmp/otto-public-demo/report.html`. Select a session and objective to inspect the observed prefix, ranked products, score components and held-out targets. The report is self-contained and works offline. Repeating the first command checks and reuses completed outputs; `--check` only verifies them.

This synthetic example demonstrates the software and evaluation contracts with transparent public rules. It does not reproduce the private research model or its measured performance. [Walkthrough and artifact contracts](PUBLIC_DEMO.md).

## Five-minute review

| Time | Inspect | What it demonstrates |
| --- | --- | --- |
| 1 minute | [Employer overview](EMPLOYER_OVERVIEW.md) | Problem, scale, ownership and delivered outcome |
| 1 minute | [Architecture](ARCHITECTURE.md) | Retrieval, task-specific ranking, temporal contracts and recovery |
| 2 minutes | [Executed study notebook](../notebooks/09_controlled_feature_study.ipynb) | Measured feature value, uncertainty, ablations and error slices |
| 1 minute | [Executed inference notebook](../notebooks/10_competition_inference.ipynb) | Small native-model replay and full-batch provenance |

For one deeper decision, read the [corrected comparator case](../research/frontier/12_corrected_comparator_candidate_coverage.md).
A challenger with positive aggregate gain still failed the click non-regression gate.
Preserving that rejection is part of the delivered validation design.

## Check the evidence without installing the ML stack

From a checkout, use Python 3.11 or newer:

```bash
python3 -S scripts/review_portfolio.py
```

The command reads six committed reports, checks the recorded model and prediction
identities, recomputes pooled weighted recall from integer counts, reconciles three
release records, and verifies the candidate-error decomposition. It requires no AWS,
Kaggle account, competition data, model download or third-party Python packages.
It is read-only and also accepts `--json` for machine-readable results and input hashes.
The expected final marker is `OTTO_PUBLIC_REVIEW_PASSED`.

This verifies the consistency of published evidence. It does not independently retrain
the models, reconstruct bootstrap intervals from private rows, or query a live leaderboard.
For locked tests, model replay and notebook execution, use [Reproducibility](REPRODUCIBILITY.md).

## Keep the three result types separate

| Evidence | Result | Interpretation |
| --- | --- | --- |
| Controlled temporal study | 0.564904 → **0.584392** on 432,492 sessions | +1.949 percentage points from the selected representation over the compact control |
| Scored release, submission 56542128 | **0.57586 private / 0.57601 public** | Post-competition evaluation; no official medal/rank claim |
| Archived time-controlled cart diagnostic | 2,181 ranked hits; 2,824 targets available in the candidate pool; 4,778 capped targets | Different cohort and objective; 643 within-pool misses and 1,954 outside-pool misses |

The [original official-prefix receipt](../reports/submissions/kaggle_submission.json)
records the older 0.56842 / 0.56862 reference. It remains immutable historical evidence.
The [promoted release receipt](../reports/submissions/similarity_stack_20260925.json)
records the newer score. `project_status.py` labels both and dates its archived research
question. Earlier full-session input scores remain invalidated.

## Delivery boundaries

| Complete and reviewable | Evidence |
| --- | --- |
| Controlled feature study and separate programmatic audit | [Study](../notebooks/09_controlled_feature_study.ipynb) and [audit](../reports/research/audit.json) |
| Full batch delivery | 1,671,803 sessions and 5,015,409 validated output rows |
| Scored post-competition release | [Recorded submission](../reports/submissions/similarity_stack_20260925.json) |
| Small public native-model replay | [Notebook 10](../notebooks/10_competition_inference.ipynb); deliberately limited review sample |
| Re-execution and recovery checks | [CI](../.github/workflows/ci.yml) and [notebook execution receipts](../notebooks/execution.json) |
| Dependency-free hands-on demo | [Demo source](../src/otto_recsys/public_demo.py), [tests](../tests/test_public_demo.py) and [dedicated CI](../.github/workflows/public-demo.yml) |
| Attribution and adaptation status | [Source reproduction matrix](../research/neural_stack/reproduction_matrix.json) |

The complete winning ensemble has not been reproduced, and no newer research challenger
has been promoted. Broader candidate coverage needs fresh independent confirmation and
scored transfer. Online deployment and revenue lift were not evaluated.

Public code includes selected method implementations and an intentionally small native-model
replay. Full event data, full prediction populations, larger checkpoints, embeddings,
credentials and private run orchestration remain outside this publication. The project
uses AWS for canonical private research; running this review does not access a space or
start any cloud resources.

See [Publication scope](PUBLICATION_SCOPE.md) for the exact distinction between synthetic data, existing historical reference artifacts and current private research assets.
