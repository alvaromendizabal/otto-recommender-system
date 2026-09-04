# OTTO Recommender System

Research-grade session-based recommendation project for the OTTO multi-objective
recommendation dataset.

## Engineering principles

1. Canonical filenames only. Corrections edit or replace the existing file.
2. Git stores history; filenames do not.
3. Raw data is immutable.
4. Long-running jobs emit UTC timestamps, stage timing, total elapsed time,
   resource telemetry, and periodic heartbeats.
5. Every material component receives unit tests and an end-to-end smoke test.
6. Validation is temporal and leakage-safe.
7. CPU/data and GPU/recommender runtimes are deliberately isolated to reduce
   dependency conflicts.
8. Experiment artifacts record the git commit, data manifest, configuration,
   random seed, package versions, hardware, timing, metrics, and output paths.

## Runtime strategy

### Studio development environment

Python 3.13.15 with a tightly pinned CPU/data/tabular stack:

- NumPy 2.5.2
- Polars 1.44.1
- PyArrow 25.0.1
- scikit-learn 1.9.0
- LightGBM 4.7.0
- XGBoost 3.4.1
- CatBoost 1.2.10
- FAISS CPU 1.15.0

### SageMaker GPU training environment

GPU training is isolated from Studio. The canonical compatibility contract is:

- Python 3.13.x
- PyTorch 2.13.0
- TorchRec 1.8.0
- FBGEMM-GPU 1.8.0
- CUDA 12.6

The GPU stack is not installed into the Studio development environment.

## First commands

```bash
uv python install 3.13.15
uv sync --extra dev --extra ml
uv run python scripts/run_quality_gate.py
```

Do not proceed to data download until the quality gate passes.
