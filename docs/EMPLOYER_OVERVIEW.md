# OTTO · Retrieval, ranking and reliable batch delivery

**Alvaro Mendizabal · Machine Learning Engineering · Recommendation Systems**

## Result and responsibility

I built a recommendation research and batch-delivery system around **216.7 million
anonymous shopping events**. I designed the pipeline, developed its representations
and ranking experiments, established temporal evaluation contracts, and engineered
recoverable AWS execution through full-population inference.

The verified post-competition release scored **0.57586 private / 0.57601 public**
and generated **5,015,409 validated rows for 1,671,803 official sessions**. These
are offline delivery and research measurements; online revenue lift and an official
competition placement were not established.

**[Explore the public synthetic demo](https://alvaro-otto-session-lab.tartmacaw2.chatgpt.site)** · [Case study](PORTFOLIO.md) ·
[Five-minute guide](REVIEWER_GUIDE.md) · [Current research status](RESEARCH_STATUS.md)

## Engineering contributions

| Layer | What I designed, implemented or evaluated |
|---|---|
| Data | Typed event partitions, source hashes and point-in-time feature contracts |
| Retrieval | Co-visitation, session revisits and popularity, with neural/latent retrieval studies; up to 400 candidates per session |
| Feature research | 1,482 formulas screened to a 102-feature controlled reference, followed by measured specialized representations |
| Ranking | Task-specific LambdaRank, neural-similarity and sequence research, routing and leakage-aware OOF ensembles |
| Evaluation | Chronological roles, matched controls, complete denominators, paired uncertainty and frozen promotion gates |
| Execution | Checkpoint recovery, immutable receipts, resource telemetry and deterministic full-batch outputs |
| Review | Tests, executed notebooks, public evidence audits and disclosure controls |

## Three decisions that matter

**Measure retrieval separately from ranking.** A model cannot rank a missing target.
On a separate archived cart diagnosis, 2,181 targets were recovered out of a 4,778
capped denominator, while the pool contained 2,824. The 643 within-pool misses and
1,954 outside-pool misses call for different hypotheses; neither difference is a
promised model gain.

**Let complexity earn its place.** On 432,492 matched temporal sessions, the selected
ranker scored **0.584392** versus **0.564904** for the compact control. The gain is
**1.949 percentage points**, with a paired 95% session-bootstrap interval of
**1.840 to 2.065 points**. This local result is separate from the external score.

**Correct the comparator before promoting a candidate.** A deployment-parity audit
found that a newer click/cart validation path did not implement the deployed policy.
Corrected replay matched the incumbent on 4,096 preserved official prefixes and
rejected the challenger because click hits regressed, despite a positive weighted
point estimate. I retained the failure and fitting evidence. [Case study](../research/frontier/11_validation_integrity_reconciliation.md).

## Try the engineering locally

```bash
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo
python3 -S scripts/run_public_demo.py --output /tmp/otto-public-demo --check
```

Open `/tmp/otto-public-demo/report.html` to trace invented sessions through retrieval,
ranking and separate future-target evaluation. Python 3.11+ is the only requirement.
The report's explanations and artifacts come from the actual small pipeline; its
synthetic metrics are not research-performance evidence. [Demo guide](PUBLIC_DEMO.md).

## What a reviewer can verify

| Path | What it verifies | Boundary |
|---|---|---|
| Synthetic demo | Local retrieval, ranking, evaluation, explanations and deterministic reuse | Transparent teaching rules, invented data |
| Evidence review | Aggregate arithmetic, release identities and error decomposition | No model retraining or new external evaluation |
| Historical native replay | Three public reference models on eight example sessions | Limited historical sample, not the current private release |
| Full research workflows | Code, protocols, tests and executed evidence | Official data and larger private artifacts required |

The [publication scope](PUBLICATION_SCOPE.md) records the source notices and the
existing public model-replay exception. Private event populations, checkpoints and
active research recipes remain outside Git.

The delivered release and current experiments have separate states. The
[research status](RESEARCH_STATUS.md) records the inspected v27 negative decision
without changing the confirmed score or treating ongoing feature research as complete.

[Architecture](ARCHITECTURE.md) · [Reproducibility](REPRODUCIBILITY.md) ·
[Controlled study](../notebooks/09_controlled_feature_study.ipynb) · [Research archive](../research/frontier/README.md)
