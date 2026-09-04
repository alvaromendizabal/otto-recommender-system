# Runtime and dependency policy

The project uses two intentionally separate environments.

## Development / CPU environment

Used for:

- source development
- tests
- dataset conversion
- temporal validation
- co-visitation candidate generation
- tabular rankers
- FAISS CPU retrieval
- experiment inspection

Runtime: CPython 3.13.15.

Direct dependencies are pinned in `pyproject.toml`. `uv.lock` becomes the exact
transitive environment contract after the first successful resolution and is committed
to Git.

## GPU environment

Used only for SageMaker training jobs involving PyTorch/TorchRec.

Compatibility contract:

- Python 3.13.x
- PyTorch 2.13.0
- TorchRec 1.8.0
- FBGEMM-GPU 1.8.0
- CUDA 12.6

The GPU runtime is intentionally not merged into the development environment.
This avoids CUDA/PyTorch/FBGEMM constraints destabilizing the CPU analytics stack.

## Version policy

We use current stable releases, not preview/nightly builds, unless a documented
experiment specifically requires one. Upgrades require:

1. a clean dependency resolution,
2. the full quality gate,
3. smoke tests,
4. an explicit compatibility record,
5. a Git commit.

No filename is changed to indicate an upgrade.
