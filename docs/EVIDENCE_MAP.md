# Engineering evidence map

[Overview](../README.md) · [Five-minute review](REVIEWER_GUIDE.md) · [Design decisions](DECISIONS.md)

## A system to inspect, not just a score

This project connects temporal data preparation, candidate retrieval, task-specific ranking and verified batch output. The recorded research processed **216.7 million events**; the delivered artifact covers **1,671,803 official sessions** and **5,015,409 task/session rows**. The public examples are intentionally smaller and do not require access to the private workspace.

| Engineering question | Inspect the implementation | Inspect the evidence |
|:--|:--|:--|
| **What was knowable at prediction time?** | [Temporal protocol](../src/otto_recsys/research/protocol.py) and [query construction](../src/otto_recsys/research/dataset.py) | [Validation notebook](../notebooks/01_validation_protocol.ipynb) and [model card](MODEL_CARD.md) |
| **Can retrieval find the relevant items?** | [Historical retrievers](../src/otto_recsys/research/retrievers.py) and [candidate/feature engine](../src/otto_recsys/research/features.py) | [Candidate-frontier notebook](../notebooks/03_candidate_frontier.ipynb); candidate availability is reported separately from ranked recall |
| **Which features actually help?** | [Fitting-only screening](../src/otto_recsys/research/screening.py) and [controlled study](../src/otto_recsys/research/study.py) | [Executed 102-feature comparison](../notebooks/09_controlled_feature_study.ipynb), ablations and paired uncertainty |
| **How are model choices kept honest?** | [Training](../src/otto_recsys/research/training.py), [evaluation](../src/otto_recsys/research/evaluation.py) and [protocol regression tests](../tests/test_research_robustness.py) | [Comparator correction](../research/frontier/11_validation_integrity_reconciliation.md) and [final research decisions](RESEARCH_STATUS.md) |
| **Did the batch result really ship?** | [Bounded native replay](../notebooks/10_competition_inference.ipynb) and [inference contracts](INFERENCE.md) | [Scored release receipt](../reports/submissions/similarity_stack_20260925.json) and [October 12 closeout](../reports/closeout/summary.json) |
| **Can another engineer inspect working software safely?** | [Public launcher](../reproduce.py), [native reference workflow](../scripts/reproduce_reference.py) and [publication checks](../tools/check_public_boundary.py) | [Reproduction CI](../.github/workflows/reproduce.yml), [quality CI](../.github/workflows/ci.yml) and [publication scope](PUBLICATION_SCOPE.md) |

## Three proof levels

**Runnable public software.** The dependency-free walkthrough generates an interactive report from invented sessions. The native reference profile uses the actual conversion, retrieval, feature and LightGBM training modules on a small invented dataset. Repeating the run verifies and reuses completed artifacts. These are executable software checks, not competition-performance measurements.

**Recorded research.** The selected reference scores **0.584392 weighted Recall@20** versus **0.564904** for its compact comparator on **432,492 temporal sessions**. The notebooks show the controlled comparison and its interpretation. Recomputing published aggregate arithmetic is not the same as independently reconstructing the private experiment.

**Recorded delivery.** The later neural-similarity release scores **0.57586 private / 0.57601 public**, with a complete artifact identity and validated output population. The October 12 audit matched the submitted CSV's complete checksum. This is post-competition offline scoring, not evidence of production traffic or online business lift.

## The technical conversation

For an **ML engineering** review, start with how candidate pools, feature contracts and model interfaces stay aligned across training and inference. Then inspect interrupted-run recovery and output validation.

For a **data science** review, start with the temporal protocol and controlled feature study. Ask how target availability, sampling, matched controls, uncertainty and rejected candidates affected decisions.

For an **applied AI** review, start with [neural retrieval](../gpu/two_tower) and [representation research](../research/neural_stack), then the integration of learned similarities into ranking. This is a recommendation and representation-learning project; it does not claim an LLM or agent deployment.

## What stays private

Full event and prediction populations, later private checkpoints and embeddings, unpublished feature recipes, private execution orchestration and credentials are not part of the public handoff. Existing source notices, historical aggregates and the deliberately bounded native replay remain intact. [Exact publication boundary](PUBLICATION_SCOPE.md).

**Status:** the verified release is retained and active experimentation is concluded. The repository is a reviewable engineering artifact, not a promise of future model gains.
