# Retrieval evaluation

This stage evaluates the multi-source OTTO candidate system before learned
retrievers and task-specific rankers are added.

## Design

The evaluator is deliberately external-memory and resumable.

1. Flatten the 515,702-session validation set into compact Parquet tables.
2. Assign each validation session to one of 32 deterministic buckets.
3. Evaluate one session bucket at a time with DuckDB and an 8 GiB memory cap.
4. Join the bucket against the existing time, type, and buy co-visitation
   matrices directly from Parquet.
5. Build source-specific ranks for revisit, time, type, and buy candidates.
6. Fuse sources with reciprocal-rank fusion.
7. Measure Recall@20/50/100/200/500/1200 for clicks, carts, and orders.
8. Persist `state.json` after every completed bucket so the exact same command
   resumes after a browser disconnect, Jupyter restart, or instance restart.

## Ablations

The benchmark reports four cumulative systems:

- `revisit`
- `revisit_time`
- `revisit_time_type`
- `full_covisit`

Incremental target hits are reported between adjacent ablations. This tests
whether each retriever contributes new ground-truth items instead of only
reordering candidates already supplied by earlier sources.

## Resource policy

Full evaluation requires at least 16 GiB RAM. On the current 32-GiB-class
SageMaker instance, DuckDB is capped at 8 GiB. Each session bucket gets an
isolated spill directory that is removed after the bucket completes.

The 2.42-GiB co-visitation graph is intentionally scanned once per session
bucket. This trades extra sequential NVMe reads for a much smaller and more
predictable peak join state, which is appropriate for the current instance.

No raw data, processed data, validation data, or completed co-visitation matrix
is modified by this stage.

## Run sequence

From the repository root:

```bash
uv run python scripts/run_quality_gate.py
```

Build the compact validation cache:

```bash
uv run python scripts/build_retrieval_validation.py \
  --validation-dir data/processed/validation \
  --output-dir artifacts/retrieval_validation \
  --buckets 32 \
  --flush-sessions 10000 \
  --heartbeat-seconds 30
```

Run the resumable co-visitation ablation benchmark:

```bash
uv run python scripts/evaluate_covisit.py \
  --validation-cache artifacts/retrieval_validation \
  --covisit-dir artifacts/covisit \
  --output-dir artifacts/retrieval_evaluation \
  --buckets 32 \
  --ks 20 50 100 200 500 1200 \
  --rrf-k 60 \
  --threads 4 \
  --memory-limit 8GB \
  --heartbeat-seconds 30
```

If the process or Jupyter session stops, run the exact same evaluation command.
Completed session buckets are recorded in
`artifacts/retrieval_evaluation/state.json` and are skipped on restart.

The final report is written to
`artifacts/retrieval_evaluation/metrics.json`.
