# Two-Tower Neural Retrieval Experiments

## Experimental objective

The two-tower retriever is retained only if it contributes measurable held-out
recall beyond the frozen non-neural retrieval system. Neural complexity is not
counted as a success by itself.

The first authorized experiment is **OOF Fold 0 only**. Folds 1-4 remain blocked
until Fold 0 produces complementary retrieval evidence.

## Leakage contract

Fold 0 is excluded from gradient updates and used for checkpoint selection. Its metrics are exploratory validation evidence, not an untouched test estimate. Training consumes
only rows assigned to folds 1-4. The frozen ranking cache, hard-negative family,
and Item2Vec initialization are identified by immutable manifests and become
part of the run ID. Source code is hashed, archived deterministically, uploaded
to S3, downloaded again, and byte-verified before paid GPU compute can start.

## Managed training contract

The Fold 0 training pipeline is a server-side SageMaker Pipeline with one
resumable training step. It uses the same checkpoint implementation that passed
the cross-worker resume proof:

- model state;
- dense and sparse optimizer state;
- scheduler state;
- NumPy/PyTorch CPU and CUDA RNG state;
- epoch, next batch, and global step;
- immutable input ID and configuration.

The worker starts with `resume-if-available=true`. A retry or deliberate rerun
with the identical run contract resumes only from a compatible checkpoint.
Changed source, data manifests, image, or configuration generates a different
run ID and therefore a different checkpoint namespace.

## Fold 0 configuration

The canonical configuration lives in `configs/two_tower.toml`. The first run
uses the already-proven `ml.g6.xlarge` path and full fold data. The configuration
sets a maximum of eight epochs with early stopping handled by the trainer,
BF16 inside the GPU package, periodic heartbeats, and durable checkpoints.

No larger GPU is justified until telemetry shows that the L4 is materially
constraining throughput or memory.

## Evaluation contract after training

The learned retriever must be evaluated at candidate depths 20, 50, 100, 200,
400, and 800 for clicks, carts, orders, and the weighted OTTO objective.

For candidate pools larger than 20, `candidate recall ceiling@K` is defined as
the maximum final Recall@20 an ideal downstream ranker could recover from that
pool. Per session, hits and ground-truth denominator are both capped at 20,
matching the competition's final top-20 constraint.

The report must include:

1. neural standalone candidate-recall ceiling;
2. frozen base-retrieval candidate-recall ceiling;
3. base + neural union recall;
4. incremental recall from the neural source;
5. neural-only positive hits by objective;
6. paired session-level uncertainty estimates;
7. ANN approximation quality and latency;
8. GPU utilization, VRAM, throughput, wall time, and billable time.

## Go/no-go rule

Do not run folds 1-4 simply because Fold 0 completed. Continue to full OOF only
when the learned source contributes non-trivial complementary positives and the
incremental gain is stable enough to justify the additional compute.

If Fold 0 is complementary, the next milestone is five-fold OOF candidate
generation followed by objective-specific learning-to-rank. If it is not, tune
or replace the neural retriever before scaling.

## Employer-visible evidence

Large checkpoints remain in S3. GitHub contains the evidence required to audit
the experiment without AWS access:

- compact run manifests and metrics under `reports/metrics/`;
- deterministic configuration under `configs/`;
- typed source and tests;
- `notebooks/05_two_tower_results.ipynb` as an executed analytical narrative;
- README tables summarizing measured results and decision rationale.

## Hermetic source-integration gate

Fold-training source is integrated through `scripts/validate_two_tower_fold.py`.
The validator does not inherit assumptions from the configured Studio `.venv`.
It creates a detached worktree from the expected base commit, overlays the exact
bundle, and runs `uv sync --frozen --extra dev --extra ml` so the complete locked
CPU development and ML dependency graph is recreated. It verifies that
`otto_recsys` resolves from that worktree, executes the full root gate there,
checks Ruff/mypy against the GPU package's exact pins, and runs the GPU pytest
contract in an isolated exact-version environment without installing Torch into
the CPU control plane. Only a clean-room pass permits application to the real tree.

The first retrieval pass uses exhaustive FP32 inner-product search with ascending-item-ID tie breaks. This establishes a reference without ANN approximation. A separate ANN fidelity and serving-latency benchmark remains required before scaling. See [FOLD_EVALUATION.md](FOLD_EVALUATION.md).
