# Engineering decisions

The most important design choices are about correctness and recoverability, not model names. This document links each choice to executable implementation and evidence.

## 1. Retrieve broadly, rank separately

**Decision.** Candidate generation is label-blind; objective-specific rankers select the final recommendations. Measure candidate availability as an oracle ceiling separately from achieved top-20 hits.

**Tradeoff.** Larger pools increase feature and inference cost and can make ranking harder. The later orders study added 44 reachable targets but its first integrated ranker lost 24 ranked hits, so that policy was rejected.

**Inspect:** [candidate/feature engine](../src/otto_recsys/research/features.py), [ranked metric](../src/otto_recsys/research/metrics.py), [closeout](RESEARCH_STATUS.md).

## 2. Separate time roles and seal decisions

**Decision.** Historical retrieval data precede fitting queries. Feature screening uses fitting data; selection chooses the model before the evaluation phase opens future targets.

**Tradeoff.** Smaller early cohorts and strict temporal eligibility can reduce apparent performance, but prevent retrospective information from contaminating the prediction problem. Previously inspected populations remain labeled as such rather than being renamed independent tests.

**Inspect:** [temporal construction](../src/otto_recsys/research/protocol.py), [matched ablations](../src/otto_recsys/research/study.py), [seal verification](../src/otto_recsys/research/evaluation.py).

## 3. Make recovery content-based

**Decision.** Complete data writes precede checksum receipts. Source/data/configuration identities bind cached stages; locks prevent conflicting writers. A mismatched cache fails rather than silently loading.

**Tradeoff.** Full hashing costs I/O. A checkpoint is useful only when it survives and is verified; temporary local storage is not treated as a backup. The closeout documents missing private experimental caches without claiming the entire private system can be restored.

**Inspect:** [conversion](../src/otto_recsys/data/convert.py), [cache construction](../src/otto_recsys/research/materialize.py), [training checkpoints](../src/otto_recsys/research/training.py), [public launcher](../reproduce.py).

## 4. Validate the delivered population, not a sample of the file

**Decision.** Validate every task/session row, exactly 20 unique nonnegative item IDs and complete session coverage. Record the resulting artifact hash separately from external scoring.

**Tradeoff.** Full validation adds delivery time but catches missing, duplicated and malformed recommendations before publication. A valid CSV alone does not establish recommendation quality or a new score.

**Inspect:** [inference and validator](../src/otto_recsys/research/inference.py), [recorded release](../reports/submissions/similarity_stack_20260925.json).

## 5. Publish runnable scope, not ambiguous promises

**Decision.** Keep the public demo, native integration profile, raw-data reference pipeline and later neural release explicitly separate. Make the first three accessible without an owner-specific account; state the latter's remaining reproduction prerequisites.

**Tradeoff.** The public reference exercises genuine research components, but does not claim that small synthetic training recreates historical numerical results. Restricted data, credentials and large private checkpoints are not shipped in Git.

**Inspect:** [clone-to-results guide](QUICKSTART.md), [publication scope](PUBLICATION_SCOPE.md), [CI](../.github/workflows/reproduce.yml).
