# Ranking and neural-retrieval training data

This subsystem creates leakage-safe supervised prefixes from the already-frozen
pre-validation training universe and mines retrieval-hard negatives without ever
using the final validation labels for training.

## Design

1. `training_cache.py` streams `data/processed/validation/train_sessions.jsonl`.
   A deterministic session hash selects a reproducible training subset. A second
   deterministic hash chooses an observed-prefix cut that always leaves hidden
   future events. Only the last `max_prefix_events` observed events are retained.
2. Future targets follow the OTTO objective contract: the first hidden event is
   the click target; all unique hidden cart and order items are multi-positive
   targets.
3. The cache writes full observed events, most-recent unique items, labels, and
   example metadata as Zstandard-compressed Parquet plus a content-addressed
   manifest.
4. `hard_negatives.py` reuses the validated co-visitation and Item2Vec candidate
   generators. Every future positive for a session/objective is excluded before
   negative ranking, preventing false-negative contamination.
5. Negatives are ranked by source agreement and reciprocal source rank. The
   output keeps compact lists of hard-negative aids instead of exploding one
   row per negative.
6. Both stages emit UTC structured logs, elapsed times, resource heartbeats,
   atomic state/manifest files, deterministic configuration identities, and
   explicit runtime guards.

## Recommended first full training configuration

Use a deterministic 1/8 sample of the 12.19M pre-validation sessions. This is a
large enough first corpus to train and benchmark the initial target-conditioned
two-tower while keeping CPU candidate mining and GPU iteration practical.

Candidate mining should use the measured high-recall Item2Vec diagnostic pool
(`item2vec_k=800`) but retain only 64 hardest non-positive candidates per
session/objective. In-batch negatives on GPU provide additional diversity.

The 800-neighbor setting is **not** treated as a final serving budget. The final
frontier remained boundary-limited, so deeper Item2Vec search is best regarded
as a hard-negative discovery pool. Future two-tower and sequential retrievers
must earn their own candidate quotas by incremental-recall testing.
