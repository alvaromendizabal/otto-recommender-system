# OTTO objective-conditioned two-tower retriever

This package is intentionally isolated from the CPU Studio environment. The base OTTO repository does not need PyTorch installed. The GPU job receives three frozen inputs: ranking cache, hard negatives, and exported Item2Vec item vectors.

The model uses a shared sparse item embedding initialized from Item2Vec, event-type/time/position embeddings, objective-conditioned attention pooling for the session tower, an objective-conditioned candidate tower, explicit retrieval-hard negatives, false-negative-masked in-batch negatives, BF16 autocast, dense and sparse optimizers, warmup+cosine scheduling, gradient clipping, deterministic fold isolation, atomic checkpoints, mid-epoch resume, JSONL metrics, and NVIDIA telemetry heartbeats.

Production training uses `/opt/ml/checkpoints` for durable intermediate state. Configure the
SageMaker training job with a run-specific checkpoint S3 URI so periodic checkpoints are synced
during training and restored into a replacement job. Use `--resume` to continue from the restored
mid-epoch state; input/configuration mismatches fail closed.
