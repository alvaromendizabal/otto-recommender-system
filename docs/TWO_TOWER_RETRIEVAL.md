# Two-tower neural retrieval

The two-tower model is the first learned retrieval challenger after the frozen co-visitation + Item2Vec candidate system. It is trained only on the frozen out-of-time ranking corpus and its false-negative-safe 64-item hard-negative lists.

## Evaluation discipline

The first paid GPU execution is a bounded smoke on one validation fold. A full single-fold training run is justified only after the smoke passes. Five-fold OOF training is justified only after the learned retriever demonstrates measurable retrieval value. The final retention criterion is incremental Recall@K against the existing co-visitation + Item2Vec union, not training loss alone.

## Isolation

GPU code lives under `gpu/two_tower/` so the CPU Studio dependency environment remains unchanged. No Torch, TorchRec, or CUDA dependencies are added to the root project.

## Durability and resume contract

GPU compute is replaceable; training state is not. Production SageMaker jobs must configure
`CheckpointConfig` with local path `/opt/ml/checkpoints` and a run-specific S3 URI. The training
entry point writes `checkpoint.pt`, `progress.json`, `metrics.json`, `training_manifest.json`,
`best_model.pt`, and JSONL logs under that checkpoint directory using atomic local replacement.
SageMaker continuously synchronizes the checkpoint directory to S3. A replacement training job
must reuse the same checkpoint S3 URI and pass `--resume`; the restored checkpoint is rejected if
its canonical input/configuration ID does not match the current run.

Final model artifacts are copied to `SM_MODEL_DIR` only after successful training. The frozen
ranking cache, hard-negative corpus, item initialization, source commit, checkpoint S3 prefix,
training-job metadata, and final model artifact therefore remain independently recoverable after
the GPU instance is terminated.
