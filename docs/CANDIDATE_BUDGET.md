# Candidate-budget optimization

This stage turns retrieval-source ablations into an objective-aware candidate budget.
It does **not** tune final ranking scores. It measures candidate-set recall versus the
number of unique candidates added by Item2Vec beyond the full co-visitation family.

## Why this stage exists

The full retrieval audit showed that Item2Vec adds unique hidden-target coverage beyond
co-visitation, but 200 ANN neighbors per objective also expands the candidate set
substantially. Before candidate materialization, hard-negative mining, or GPU retrieval,
we measure the exact marginal-recall curve at multiple Item2Vec depths.

The default grid is `0, 10, 20, 50, 100, 150, 200`. For each objective the evaluator
reports:

- co-visitation recall ceiling;
- Item2Vec standalone recall at each depth;
- Item2Vec-only marginal recall at each depth;
- union recall at each depth;
- exact average unique candidate count;
- marginal recall per 100 additional unique candidates.

The default recommendation is the **smallest Item2Vec quota that preserves at least
95% of the maximum observed Item2Vec-only marginal recall** for that objective. The
capture fraction is explicit and configurable.

## Engineering contract

- frozen leakage-safe validation inputs only;
- full co-visitation family retained while Item2Vec depth is swept;
- one validation bucket at a time;
- bounded DuckDB memory and isolated spill directories;
- atomic `state.json` after every completed bucket;
- exact input-identity hash prevents incompatible resume;
- UTC structured logging, 30-second heartbeat, RSS/CPU telemetry, stage and total time;
- deterministic quota grid and deterministic recommendation rule;
- unit/integration tests for rank histograms, exact union counts, and quota selection.

## Output

`artifacts/candidate_budget/metrics.json` contains the full Pareto curve and
objective-aware recommended Item2Vec quotas. `state.json` makes the run resumable.

These quotas become inputs to the next canonical candidate-materialization and
hard-negative-mining stage. They are not hidden defaults in downstream code.
