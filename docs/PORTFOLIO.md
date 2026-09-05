# OTTO Recommender System — Portfolio Guide

This repository is organized as an engineering project first and an analytical portfolio second.

## What to review first

1. `README.md` — project entry point and reproduction commands.
2. `notebooks/01_validation_protocol.ipynb` — temporal validation and multi-objective metric design.
3. `notebooks/02_retrieval_benchmarks.ipynb` — retrieval-source evidence and measured recall.
4. `notebooks/03_candidate_frontier.ipynb` — candidate-budget / recall trade-off.
5. `notebooks/04_hard_negative_quality.ipynb` — OOF data construction and hard-negative integrity.
6. `src/otto_recsys/` — typed CPU retrieval and data pipeline implementation.
7. `gpu/two_tower/` — objective-conditioned neural retrieval implementation.
8. `tests/` — unit, integration, and contract tests.

## Engineering signals the project is designed to demonstrate

- leakage-aware temporal validation rather than random splits;
- objective-specific retrieval for clicks, carts, and orders;
- explicit ablations and incremental-recall analysis;
- memory-aware large-scale processing of hundreds of millions of events;
- deterministic manifests and artifact hashes;
- false-negative-safe hard-negative mining;
- OOF fold discipline;
- GPU checkpoint/resume semantics;
- separation of control-plane CPU code from paid GPU training;
- static analysis, unit tests, smoke tests, structured timestamps, runtime telemetry, and heartbeats;
- Git for source history and S3 for large immutable artifacts.

The notebooks are intentionally thin. They explain and visualize canonical result artifacts; the implementation remains in importable modules and scripts so that production logic is not duplicated inside notebooks.
