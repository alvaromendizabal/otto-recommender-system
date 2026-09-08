# Runtime and dependency policy

Three environments have separate responsibilities and dependency contracts.

| Profile | Runtime | Contract and use |
|---|---|---|
| Project / CPU | CPython 3.13.15 | `pyproject.toml` and committed `uv.lock`; conversion, retrieval, feature research, ranking, inference and project tests |
| Analysis | CPython 3.12.13 | `notebooks/requirements.in` plus the full transitive hash lock in `notebooks/requirements.txt`; notebook kernels, Plotly/PNG output and native-model replay |
| Neural | Separate PyTorch environment | The recorded SageMaker image/runtime and `gpu/two_tower/` requirements; objective-conditioned retrieval and checkpoint tests |

The measured tabular study uses LightGBM 4.7.0, DuckDB 1.5.5, Polars 1.44.1 and
PyArrow 25.0.1. Its exact source, lockfile hash, Python version, seed and resources
are retained in [the managed record](../reports/research/managed_study.json).
The analysis environment has a separate NumPy version; native-model replay verifies
prediction agreement across the project and analysis runtimes.

The implemented two-tower uses PyTorch sparse item embeddings, dense/sparse optimizers,
objective-conditioned attention pooling and BF16 GPU autocast. TorchRec/FBGEMM were
possible extensions, not required dependencies of the measured implementation.
Neural CPU contract CI uses Python 3.12.13 and the official PyTorch 2.13.0 CPU wheels.
It does not certify a CUDA runtime. The GPU validation script verifies actual CUDA,
BF16, a training step and checkpoint round-trip inside the selected managed image.
Historical neural runs retain their own source and runtime provenance; the project
CPU lock must not be substituted for their environment.

Plotly's static export requires Chrome in addition to the pinned Kaleido package.
The GitHub runner executes both interactive and PNG representations. Notebook execution
binds the kernel to the explicitly selected interpreter, rather than the user's default
Jupyter kernel. Production inference invoked by Notebook 10 uses the locked project
interpreter; the analysis kernel reads its validated output.

Upgrades require an explicit dependency resolution, relevant compatibility checks,
quality gates and a commit. Source and artifact identities remain tied to the versions
that actually produced them; changing a version does not retroactively validate an old
experiment under a new environment.
