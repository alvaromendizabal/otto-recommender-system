# Supervised neural retrieval · current frontier

**Status:** real project execution is in progress. This directory records the design, source attribution, reproducibility boundaries and latest verified state. It does **not** claim a new neural validation score.

The strongest verified competition result is **0.57100 private / 0.57121 public** (submission `56472100`, scored after the competition deadline). The historical private winning benchmark recorded by the project is **0.60503**, leaving a **0.03403** absolute gap. The current neural experiment addresses a capability present in leading OTTO systems that the submitted router does not yet reproduce: supervised, task-conditioned sequence retrieval integrated with downstream ranking.

Start with [03_neural_stack_status.ipynb](03_neural_stack_status.ipynb). It is an executed, data-free review notebook with inline Plotly output and static fallbacks. [status.json](status.json) is the machine-readable snapshot; [reproduction_matrix.json](reproduction_matrix.json) states exactly which leading-solution mechanisms are adapted, validated, pending or still missing.

## Why this round exists

Recent small-pilot research did not establish a path to the winning score. Increasing the pilot from 8,192 to 32,768 fitting sessions produced a **+0.004229** matched point gain but an interval crossing zero. More importantly, a same-corpus bridge showed the established 100,000-session pipeline at **0.563622** on the same 16,384 sessions, versus **0.550513** for the 32,768-session pilot. That evidence ended pilot expansion.

The next capability therefore comes from the leading-solution gap analysis rather than another shallow feature variant.

## Adapted first-place mechanism

The inspected first-place v42 source uses a task-conditioned sequence encoder with item embeddings, time context and event-type information, plus a contrastive objective that combines a weighted positive mean with the hardest positive and mines hard negatives. This project independently implements that mechanism with explicit methodological changes:

- 500-dimensional item embeddings and 50-dimensional time context.
- Last 10 observed events as the sequence input, with up to 10 future events as supervised targets inside the permitted historical period.
- Hard-negative contrastive training with positive/negative collision masking and stable log-sum-exp arithmetic.
- Cyclic hour-of-week context to support temporal extrapolation.
- One reproducible historical cut per eligible session/epoch rather than every possible anchor.
- One encoder in this experiment, not the winner's eight-model neural ensemble.

No upstream executable code, weights or private artifacts are vendored here. The source references are recorded in the reproduction matrix.

## Integration design

The experiment does **not** replace established retrieval. It preserves all 400 incumbent candidates and appends up to 200 neural order candidates, creating a maximum order pool of 600. One fixed order ranker then combines the established selected feature schema with eight neural signals. Click and cart routing remain fixed.

The fixed scale is **100,000 fitting sessions → 20,000 selection sessions → 432,492 evaluation sessions**, with **110,057,408** permitted historical events feeding sequence preparation. A nonpositive selection gain stops the direction. If selection is positive, the sealed pipeline proceeds to the full evaluation cohort.

Final advancement requires at least **+0.003 weighted Recall@20**, a positive lower paired 95% interval bound, nonnegative gains in both chronological halves and no order-hit loss. The temporal result would still be distinct from a Kaggle score.

## Current execution state

The owner-run single-GPU workflow has passed the engineering tests and native ranker integration, verified the real reference implementation, reused the downloaded reference corpus and reused prepared sequence data. The prior run committed a complete model/optimizer/RNG checkpoint at **at least step 6,000**. The latest supplied rerun entered `supervised_sequence_training`; the supplied excerpt did not yet contain a selection or evaluation result.

Checkpoint reuse is intentional. The expensive historical preparation and committed optimizer/model state remain in AWS. This public repository does not copy the 11+ GiB checkpoint, raw data, embeddings, labels, per-session predictions or optimizer tensors.

## Leading-solution reproduction state

The project should not describe itself as having recreated the winning system merely because individual mechanisms have been discussed or partially adapted. The current matrix records, among other items:

- Baseline co-visitation and task ranking: **adapted and evaluated**.
- Bounded multi-hop retrieval: **partially adapted; tested replacement recipe regressed**.
- Word2Vec candidate/session similarity: **adapted and previously tested; no robust overall gain established**.
- First-place v42-style supervised sequence retrieval: **independently implemented; project evaluation in progress**.
- Eight-model winning neural candidate ensemble: **not yet reproduced end to end**.
- TheoViel-style full MF / sequence-to-sequence / 744-feature XGBoost stack: **integrated validated reproduction not established**.
- Full complementary winning ranker ensemble: **not yet reproduced**.

See [reproduction_matrix.json](reproduction_matrix.json) for the auditable component-by-component record.

## Reproducibility and publication boundary

The public notebook exposes the research question, current score gap, scale decision, architecture, protocol and stopping logic. The private AWS runner remains content-addressed, checkpointed and restartable. There is no elapsed-time cutoff in this neural experiment; integrity, memory, disk and scientific progression gates remain active.

This publication is intentionally a review surface, not an AWS mirror. Raw competition data, full model checkpoints, embeddings, environments, account logs and credentials remain private.
