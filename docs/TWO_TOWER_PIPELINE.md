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
