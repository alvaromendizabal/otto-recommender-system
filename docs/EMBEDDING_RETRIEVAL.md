# Embedding retrieval

This stage adds a learned retrieval family that is intentionally complementary to
co-visitation rather than a replacement for it.

## Item2Vec

Training uses leakage-safe validation-training sessions and streaming skip-gram
with negative sampling. The corpus is read from JSONL on demand; the full 199M
events are never loaded into Python memory. Defaults are 128 dimensions, window
10, 10 negative samples, five epochs, and seed 42.

The model and KeyedVectors are persisted using Gensim's separate-array format so
large vector arrays can be memory-mapped for indexing and evaluation.

## FAISS

The ANN layer uses cosine-normalized HNSW with explicit item IDs. Vectors are
added in bounded batches and the completed index is verified after persistence.
Default HNSW settings are M=32, efConstruction=200, and efSearch=256.

## Evaluation

Validation is bucketed exactly like the co-visitation benchmark. For each target,
observed item embeddings are pooled using recency and action-type weights before
ANN search. Recall@20/50/100/200 and the weighted OTTO objective are recorded.

The next stage compares source-aware unions of revisit, co-visitation, Item2Vec,
and later neural retrieval. Low-K fusion is not assumed to be optimal: the
co-visitation ablation already shows that type/buy sources add high-K diversity
while naive reciprocal-rank fusion can degrade top-20 ordering.
