# Roadmap

## 0. Foundation
- deterministic runtime
- CI
- structured logging
- heartbeat
- tests
- environment preflight

## 1. Data
- official OTTO data
- immutable manifest and hashes
- streaming JSONL to typed Parquet
- resumable conversion

## 2. Validation
- temporal validation reconstruction
- future-event masking
- clicks/carts/orders ground truth
- leakage tests

## 3. Retrieval baselines
- session revisit
- popularity
- time-weighted co-visitation
- type-weighted co-visitation
- buy-to-buy co-visitation
- candidate provenance

## 4. Embedding retrieval
- Item2Vec / skip-gram
- FAISS ANN
- hard-negative candidate diagnostics

## 5. Neural retrieval
- target-conditioned two-tower
- TorchRec embeddings where useful
- mixed precision
- hard negatives
- checkpoint/resume

## 6. Sequential retrieval
- Transformer / SASRec-style
- relative-time features
- event-type embeddings
- optional HSTU-inspired or state-space challenger

## 7. Ranking
- LightGBM LambdaRank
- XGBoost ranker
- CatBoost YetiRank
- neural reranker challenger
- separate clicks/carts/orders pipelines

## 8. Ensemble
- OOF predictions
- rank-normalized blending
- constrained weight search
- ablations and seed robustness

## 9. AWS portfolio layer
- S3 lineage
- SageMaker processing/training
- experiment records
- cost and throughput reports
- CI
- model card
