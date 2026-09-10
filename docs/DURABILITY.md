# Durability and Recovery Contract

The project treats compute as replaceable and state as durable.

## Git

Canonical source, tests, documentation, notebooks, and small reports are versioned in GitHub.

## S3

Large immutable/recomputable artifacts are frozen to deterministic prefixes and accompanied by manifests.

## Training checkpoints

Neural training checkpoints must include enough state to continue optimization rather than merely reload weights:

- model state;
- optimizer state;
- learning-rate scheduler state;
- epoch / batch / global step;
- best metric and early-stopping state;
- RNG state;
- immutable input identity.

A resume request must fail closed if its checkpoint is missing or incompatible.

## Operational visibility

All long jobs should emit:

- UTC timestamps;
- stage start/end events;
- total elapsed time;
- periodic heartbeat;
- CPU/RAM telemetry;
- GPU utilization and VRAM telemetry for GPU work;
- durable metrics and progress state.

A managed AWS job should continue if the interactive Studio browser is closed.

## Recover the domain-feature study

The immutable launch is `reports/research/domain_feature_launch.json`; the managed
job and source identities are recorded in `reports/research/domain_feature_run.json`.
The launch includes all corpus, baseline-model and historical-graph checksums. Its
checkpoint URI is sufficient to locate completed feature parts, model iterations,
selection statistics and UTC logs after an interrupted chat session.

For a completed job, collect its evidence in an authenticated project environment:

```bash
uv run --frozen --extra ml --extra cloud python scripts/collect_domain_study.py
```

The collector requires managed completion, checks the source input and account,
downloads only the small model/audit evidence and observed query corpus, verifies
file lengths and SHA-256, and reuses valid local downloads. It recomputes the
independent aggregate audit, descriptive query slices and instance-compute estimate.
It launches no jobs and refits no models. Candidate feature caches remain remote.
The resulting small reports are the inputs to canonical notebook 09.

If processing actually fails or is stopped, retain the existing checkpoint prefix.
After verifying the original worker has terminated, a replacement job can use the
same eight immutable processing inputs and launch JSON with a unique job name.
Keep the source archive and study contract identical: feature parts and model
iterations will then resume after checksum checks. A changed contract is a new
experiment, not a continuation of an old one. A completed study needs collection,
not a replacement training job.
