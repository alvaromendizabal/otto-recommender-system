# Final Reserve V2 · active label-blind evaluation

This document records the final offline evaluation protocol for the currently promoted click/cart stack.

## Candidate

- **Clicks:** full source400 click ranker
- **Carts:** `zmean_seq__all`
- **Orders:** unchanged incumbent order policy

The cart stack combines source400 with four sequence rankers using within-session standardized scores.

## Final reserve

Population:

**412,492 sessions**

Predictions are generated in deterministic chronological shards before any reserve label is opened.

Current checkpoint:

- sessions predicted: **200,000 / 412,492**
- shards sealed: **4 / 9**
- remaining sessions: **212,492**
- reserve labels opened: **false**

Prediction preparation is content-addressed and resumable. Completed shards are never regenerated unless an integrity check fails.

## Frozen final gate

After all 9/9 shards are sealed, the complete prediction set is frozen before labels are read.

Promotion requires:

- combined weighted gain ≥ **+0.0025**
- click hit gain ≥ **0**
- cart hit gain ≥ **+200**
- chronological-block bootstrap 95% lower bound **> 0**
- first half ≥ **0**
- second half ≥ **0**
- at least **3/4 quarters** nonnegative
- worst quarter ≥ **−0.0005**

Bootstrap replicates: **2,000**

## Shard execution state

| Shard | Cumulative sessions | Status |
| ---: | ---: | --- |
| 1 | 50,000 | sealed |
| 2 | 100,000 | sealed |
| 3 | 150,000 | sealed |
| 4 | **200,000** | **sealed** |
| 5 | 250,000 | pending |
| 6 | 300,000 | pending |
| 7 | 350,000 | pending |
| 8 | 400,000 | pending |
| 9 | 412,492 | pending + final evaluation |

Recent 50k shards complete in roughly 28 minutes on `ml.g6.4xlarge`. CPU feature construction is the dominant wall-clock stage; GPU memory is not the bottleneck.

## Publication boundary

The public repository reports:

- aggregate progress
- cohort sizes
- frozen gates
- lifecycle decisions
- measured runtime/resource behavior

It does not publish:

- reserve session IDs
- row-level predictions
- reserve labels
- private model checkpoints
- private runners
- cloud account details

The next public scientific result should be the completed reserve decision, not an intermediate score.
