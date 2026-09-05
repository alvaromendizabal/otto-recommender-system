# Durable SageMaker two-tower training

The SageMaker GPU worker is disposable. The training state is not.

The resume-proof workflow freezes the training source under an immutable Git commit and source
SHA-256, uses only S3-backed data channels, and configures SageMaker checkpoint synchronization
from `/opt/ml/checkpoints` to a dedicated S3 run prefix. Job A is intentionally stopped only after
both `checkpoint.pt` and `progress.json` are visible in S3. Job B starts on a new GPU with the same
checkpoint S3 prefix and `--resume`; the trainer fails closed if the checkpoint is missing or its
input identity differs.

A resume proof passes only when Job B advances `global_step` beyond the checkpoint observed from
Job A. Both jobs are stopped after the proof to control GPU cost. The requests, source archive,
progress snapshots, and final proof manifest are retained under the immutable S3 run prefix.

The current GPU image contract is the AWS SageMaker PyTorch 2.13 Amazon Linux 2023 DLC with
Python 3.12 and CUDA 13.3. The CPU Studio environment remains unchanged.
