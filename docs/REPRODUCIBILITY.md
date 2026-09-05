# Reproducibility

## Principle

Source code and compact public result summaries belong in Git. Large datasets, model artifacts, checkpoints, and raw experiment outputs belong in durable object storage and are identified by manifests/hashes.

## Local engineering gate

```bash
uv run python scripts/run_quality_gate.py
uv pip check
git diff --check
```

## Result notebooks

The notebooks read only compact files under `reports/metrics/`; they do not contain hidden production logic or require access to private S3 data.

## Expensive jobs

Long-running jobs should be launched as managed AWS jobs. A browser or terminal disconnect must not be required for the job to continue. Checkpointed training must write recovery state to S3 and validate input identity before resuming.

## Provenance expected for each learned model

- Git commit
- validation manifest ID
- input artifact hashes
- configuration
- random seed / fold
- training metrics
- checkpoint state
- evaluation metrics
- model artifact hash
