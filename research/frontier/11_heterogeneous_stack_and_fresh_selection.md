# Heterogeneous OOF stacking and Fresh Selection V2

This snapshot records the strongest fitting-stage promotion after the click/cart source-aware selection rejection.

## Heterogeneous base models

The stack combines nine OOF rankers spanning three model families:

- source400 LightGBM
- sequence residual MLP
- DIN-style cross-attention
- BST-style sequence attention
- behavior-specific cross-attention
- CRAFT flat MLP
- CRAFT contextual gate
- CRAFT cross-interaction model
- CRAFT mixture-of-experts model

## Stack families

The experiment evaluated twelve stack/policy combinations using:

- reciprocal-rank fusion
- within-session standardized-score averaging
- leakage-safe learned meta rankers

Learned meta models used strict OOF discipline: each held-out fold was scored by a meta model trained only on the other four folds.

## Fitting promotion

Selected stack:

**`zmean_seq__all`**

Aggregate fitting evidence:

- deployment-aligned combined gain: **+0.004203**
- cart hits: **2,712 → 2,776**
- cart gain: **+64**
- recovered / lost cart hits: **118 / 54**
- combined folds nonnegative: **5/5**
- cart folds nonnegative: **5/5**
- worst combined fold: **+0.002071**
- worst cart fold: **+0.001474**

Decision: **PASS FITTING GATE**

## Fresh Selection V2

The original selection cohort had already been opened, so it was not reused.

A new independent 20,000-session selection stage was frozen from previously unopened evaluation-role sessions using observed-prefix chronology only.

The remaining sessions were preserved as Final Reserve V2.

Fresh Selection V2 result:

- combined weighted gain: **+0.003194**
- click hit gain: **+136**
- cart hit gain: **+47**
- paired 95% interval: **[+0.001865, +0.004554]**
- first half: **+0.004021**
- second half: **+0.002373**

Decision: **PASS**

The remaining Final Reserve V2 labels stayed unopened.

## Validation lesson

The project deliberately paid the cost of creating a fresh selection boundary rather than reusing a cohort that had already influenced research decisions.

This keeps the promotion claim interpretable despite extensive adaptive experimentation.

See [12_reserved_evaluation.md](12_reserved_evaluation.md) for the active final reserve stage.
