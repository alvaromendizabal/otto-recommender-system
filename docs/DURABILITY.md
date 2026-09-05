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
