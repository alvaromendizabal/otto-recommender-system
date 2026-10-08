# Public review and private research boundaries

This repository is an employer-facing record of a completed offline recommendation
study and batch delivery. Public review does not require access to the private
research workspace, cloud account or complete competition data.

| Public material | What it lets a reviewer verify |
| --- | --- |
| Selected reference implementations, tests and data contracts | Metric arithmetic, temporal checks, model interfaces and recovery behavior |
| Aggregate reports, recorded submission receipts and executed notebooks | The published measurements and their stated scope |
| Dependency-free evidence checker | Consistency of aggregate counts and recorded artifact identities; it does not independently authenticate the original observations |
| [Synthetic demonstration](PUBLIC_DEMO.md) | A small, inspectable recommendation workflow using invented data; its results are not competition measurements |
| Bounded historical native-model replay | Three already-public controlled-reference LightGBM models and candidate examples from eight official-prefix sessions, reproducing 24 historical output rows |

The existing replay is a deliberate, limited exception to the otherwise aggregate
publication boundary. It includes reference-model files and example row-level inputs
and outputs under `reports/research/inference_replay/`. It is not the complete later
release, a full dataset, or a way to reproduce the competition score from scratch.

Full event data and label/prediction populations, current private checkpoints and
embeddings, unpublished feature recipes, private run orchestration and credentials
remain outside this publication. New portfolio work must not copy those assets into
code, notebook outputs, HTML, logs or download bundles. Synthetic examples must stay
clearly labeled and separate from measured research evidence.

Recorded scores are post-competition results. Programmatic audits are project checks,
not third-party certification. No official medal/rank, online deployment, revenue lift,
or improvement from unpublished experiments is claimed. Historical evidence retains
its original date and scope.

The [demo launcher](../scripts/run_public_demo.py) uses the Python standard library.
See [Reproducibility](REPRODUCIBILITY.md) for the supported public commands and
[Model card](MODEL_CARD.md) for the controlled reference's limitations. Upstream method
attribution is recorded in the [component audit](../research/neural_stack/reproduction_matrix.json).
