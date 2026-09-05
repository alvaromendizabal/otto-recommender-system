# SageMaker two-tower training

The SageMaker GPU worker is disposable; training state is not.

The canonical execution path is the managed SageMaker Pipeline described in
[`TWO_TOWER_PIPELINE.md`](TWO_TOWER_PIPELINE.md). Source code is committed to GitHub, the exact
training source archive is deterministically hashed and frozen to S3, all large inputs are S3-backed,
and `/opt/ml/checkpoints` is synchronized to a run-specific S3 checkpoint prefix.

The two-step resume proof uses a bounded first training job to create a durable checkpoint and a
fresh second training job that is required to restore that checkpoint and advance optimization.
The second job fails closed if the checkpoint is absent, has a mismatched immutable input identity,
or does not advance beyond the restored `global_step`.

Long-running orchestration is owned by SageMaker after the pipeline execution starts. An interactive
Studio terminal is used only to register/start the managed execution and to inspect status later.
