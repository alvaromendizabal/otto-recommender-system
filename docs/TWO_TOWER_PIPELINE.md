# Managed Two-Tower Resume-Proof Pipeline

## Goal

Prove that the neural retriever can continue optimization on a fresh SageMaker GPU worker without relying on an interactive Studio terminal.

## Managed execution

The workflow is a SageMaker Pipeline with two serial training steps:

1. **CreateDurableCheckpoint**
   - starts from frozen S3 inputs;
   - trains to a configured checkpoint boundary;
   - writes a complete atomic checkpoint to `/opt/ml/checkpoints`;
   - exits successfully after the checkpoint is durable.

2. **ResumeAndAdvance**
   - starts on a fresh SageMaker training worker;
   - uses the same checkpoint S3 URI;
   - requires `--resume` and fails if no compatible checkpoint is restored;
   - advances beyond the restored `global_step`;
   - writes `resume_proof.json` with the restored and final step counts.

The pipeline succeeds only if both SageMaker Training steps succeed. The second training script additionally fails closed unless the restored checkpoint has a positive `global_step` and training advances beyond it.

## Durable state

Each run is identified by a hash of:

- Git commit;
- deterministic source archive hash;
- frozen input-manifest identities;
- training image;
- training and resume-proof configuration.

Run state is stored under:

```text
s3://<bucket>/retrieval/two-tower/runs/resume-proof/<run-id>/
    checkpoints/
        checkpoint.pt
        progress.json
        resume_event.json
        resume_proof.json
        training_manifest.json
    control/
        pipeline_definition.json
        run_manifest.json
        execution.json
    output/
        job-a/
        job-b/
```

## Leave-and-return behavior

After `scripts/launch_two_tower_pipeline.py --start` returns a pipeline execution ARN, the AWS-managed execution no longer depends on the Studio terminal. The browser can be closed.

Check status later with:

```bash
uv run python scripts/two_tower_pipeline_status.py
```

The status command reads the durable S3 run pointer and SageMaker Pipeline execution state.

## Cost discipline

Registration creates/updates the pipeline definition but starts no GPU compute. The paid run uses bounded GPU steps purely to prove checkpoint recovery before a complete fold is authorized.

## Registration contract

The launcher emits retry policies using the current SageMaker Pipelines service shape, where each policy stores `ExceptionType` as an array. Registration itself is a no-GPU live service validation: the pipeline must be accepted by `CreatePipeline` or `UpdatePipeline` before any execution can be started.

## Exact-source preflight

Before a managed GPU execution can be registered or started, the launcher now
fails closed on the exact `gpu/two_tower/` source tree. It runs:

1. Python compilation;
2. Ruff using the GPU package configuration;
3. mypy against the GPU package and entry points;
4. CPU-safe resume/entrypoint contract tests;
5. deterministic source-archive creation;
6. file-by-file SHA-256 parity between the working tree and the tarball;
7. an S3 upload/download round trip followed by the same byte/content parity
   verification.

The pipeline is not submitted if any of those checks fail. The static-analysis
toolchain is exactly pinned (`ruff==0.16.6`, `mypy==2.3.1`, `pytest==9.0.3`) and
the launcher executes those exact versions in an isolated `uvx`/`uv tool run` environment from inside the GPU source
root. Ruff also has explicit first-party classification for both
`otto_two_tower` and `sagemaker_entrypoint`, eliminating environment-dependent
import sorting. This prevents a locally approved tree from diverging from the
exact source bytes SageMaker executes.

The paid GPU worker does **not** reinstall the developer lint/type-check stack or
repeat static analysis. It installs only `requirements.txt`, compiles the runtime
source with the container's Python interpreter, validates CUDA/GPU availability,
and then trains. Static analysis belongs on the CPU control plane; GPU time is
reserved for runtime validation and model execution.

The run manifest records the source archive SHA-256 plus local and S3-roundtrip
verification metadata, making the training source independently auditable.

## Failure observability

The SageMaker entrypoint records stage-aware failures in two places before it
returns a nonzero exit code:

```text
/opt/ml/output/failure
/opt/ml/output/data/failure.json
```

`failure.json` contains the failed stage, return code, command, elapsed time,
Git commit, run ID, Python/runtime metadata, and GPU information when available.
Unexpected exceptions also include a traceback. Successful jobs write
`entrypoint_summary.json` to `/opt/ml/output/data`.

`scripts/two_tower_pipeline_status.py` now reports pipeline steps, checkpoint
object counts, the failed SageMaker training job, instance type, billable time,
and native failure reason. When a persisted `failure.json` is available in the
training output artifact, the status command surfaces its stage and message.
Use `--show-logs` only when the recent CloudWatch tail is needed.
Pinned test execution uses `python -m pytest` inside the isolated `uv` environment so the exact GPU source directory remains on Python's import path.
