# Ranking and neural-retrieval training data

This subsystem materializes the **frozen official-local validation prefixes and
labels** as supervised training data and mines retrieval-hard negatives from
artifacts fitted only on the earlier pre-validation universe.

## Leakage contract

The ranking cache does **not** synthesize hidden futures inside the same sessions
used to fit co-visitation or Item2Vec. Instead it consumes
`data/processed/validation/test_sessions.jsonl` and the aligned
`test_labels.jsonl`. Those prefixes occur after the training cutoff used by the
validated co-visitation matrices and Item2Vec model.

This separation is critical: candidate-generation artifacts must not have been
fit on the hidden future of a supervised example. The hard-negative miner checks
that the ranking cache and Item2Vec artifacts share the same frozen validation
manifest identity before it runs.

## OOF design

Every session receives a deterministic hash-based fold. The default is five
folds. Downstream rankers and neural rerankers must produce out-of-fold
predictions for local evaluation: a session is scored only by a model that was
not trained on that session's labels.

For a final Kaggle submission, after OOF model selection is complete, the chosen
model may be refit on all frozen local-validation examples because those examples
are historical relative to the competition test sessions.

## Stored artifacts

`training_cache.py` writes Zstandard-compressed Parquet:

- `events.parquet`: every observed prefix event in order;
- `items.parquet`: most-recent unique observed items for retrieval;
- `labels.parquet`: click/cart/order targets with OOF fold IDs;
- `examples.parquet`: session-level counts, timestamps, folds, and buckets;
- `manifest.json`: source hashes, validation identity, configuration, output
  hashes, counts, and elapsed time.

`hard_negatives.py` reuses the validated co-visitation and Item2Vec candidate
generators. It excludes **every positive label for the same session/objective**
before ranking negatives. Each positive row retains its OOF fold and a compact
list of hard-negative item IDs plus source-agreement diagnostics.

## Runtime engineering

Both stages use canonical filenames, explicit resource guards, UTC structured
logging, elapsed-time reporting, progress heartbeats, deterministic input
identities, atomic writes, and content hashes. Hard-negative mining is resumable
bucket by bucket through `state.json`.

## Recommended first neural corpus

Use all 515,702 frozen local-validation prefixes for the first target-conditioned
two-tower experiment. Mine from the measured high-recall Item2Vec discovery pool
(`item2vec_k=800`) and retain 64 hardest non-positive items per
session/objective. GPU training should combine these with in-batch negatives.

The 800-neighbor setting is a **discovery pool**, not a serving budget. Every
learned retriever must earn its final quota through incremental-recall and
Recall@20 evaluation.
