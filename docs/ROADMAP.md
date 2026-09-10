# Project status and research direction

**Feature engineering remains open.** The accepted baseline scores **0.56842 private /
0.56862 public**. Its engineering release is complete, and all nine original temporal
validation cells are independently audited. This does not complete the current request
for broad feature research or establish the historical winning score of 0.60503.

## Verified experiment state

| Work | Evidence-backed status |
|---|---|
| Baseline feature study | 1,482 formulas, 102 selected features; 0.584392 offline weighted Recall@20 |
| Original temporal/seed confirmation | 9/9 audited; +1.791 to +2.180 percentage points over matched compact controls |
| Official-prefix submission | Accepted; 0.56842 private / 0.56862 public |
| Item2Vec feature pilot | Completed; no supported weighted gain, no promotion |
| Wider retrieval pilot | Completed; both challengers improved coverage but reduced final ranking |
| Complementary graph feature study | **Completed and audited**: 0.595269 / 0.595923 / 0.596443 versus baseline 0.599523; no promotion |
| Domain feature study | **Completed and audited**: 172 eligible formulas, five arms, 12 new models; best weighted point 0.599941 versus 0.599523, interval spans zero; no promotion |
| Broader feature completion gate | **Open**; [coverage inventory](FEATURE_RESEARCH.md) records unresolved work |
| New-feature temporal confirmation | Pending; the original inspected cohorts are not fresh holdouts |
| Earlier 50-file collection | Not executed; one accepted baseline file is verified |

The graph-feature study completed at **03:47:56 UTC on September 10** after 34.25 managed minutes. The [run receipt](../reports/research/graph_feature_run.json) and [audit](../reports/research/graph_feature_audit.json) retain the source identity, all four results and native-model verification. That job is no longer running.

The study reused both historical graphs, all original candidate
rows, labels, negative samples and 102 baseline features. It tested 144 affinities, retained 120 after fitting-only redundancy screening,
and measured four arms: baseline, symmetric, forward and both. A successful execution
is not itself evidence of improvement. A positive development result still needs
separate temporal confirmation before it can replace the accepted baseline.

The [complete coverage assessment](FEATURE_RESEARCH.md) distinguishes tested original
families from unresolved as-of demand, sequence/funnel context, smoothed encodings,
normalized and multi-hop graphs, factorization/session neighbors, task-conditioned
neural retrieval, multiple intents, repeat/new behavior and candidate frontiers.
The available inputs do not support invented product semantics or persistent-user
profiles. Feature counts are not a completion percentage.

## Exact status and recovery

```bash
uv run --frozen --extra dev --extra ml python scripts/project_status.py
uv run --frozen --extra dev --extra ml python scripts/project_status.py --require-feature-gate
uv run --frozen --extra cloud python scripts/retrieval_status.py --receipt reports/research/domain_feature_run.json
```

The project-status command reads and verifies committed evidence; it does not query
live AWS state. Its optional gate check exits 2 while feature coverage or confirmation
is incomplete. The retrieval-status command checks the actual managed job and logs.
Neither command launches paid compute. A saved running observation must not be treated
as proof that the job is still running now.

A duplicate job is unnecessary when a matching study is active. Source, corpus,
candidate cache and graph hashes bind the experiment to its inputs. Native models,
feature partitions and UTC heartbeats persist in the project AWS account. The domain launch and completion receipts identify the owned S3 checkpoints and immutable source; credentials and signed download URLs are not published.
The completed managed job used one instance with a two-hour runtime cap.

## Latest completed study and next work

The domain job `otto-domain-features-09629193067f` completed at **06:22:01 UTC on
September 10**. Independent metric/model verification passed after all 64 evidence
files were recovered and checksum verified. It used **50.22 processing minutes**,
estimated **$2.87 in instance compute**, excluding storage, requests, logging and
transfer. The job is completed; it does not need another training launch.
[Completed receipt](../reports/research/domain_feature_run.json) ·
[Measured outcomes and limitations](FEATURE_RESEARCH.md#completed-domain-comparison-september-10-2026).

Sequence features have the highest click/cart point estimates among these arms;
normalized graph features have the highest order point estimate. Selecting those
existing models after inspecting this cohort gives 0.601446, an optimistic diagnostic,
not an independently confirmed improvement. Reuse the feature caches to separate
funnel/episode and row/degree blocks, measure fitting-only per-action utility, and
freeze the next configuration before temporal confirmation. The 14 unresolved families
remain open; the original six covered scopes and two data-based exclusions are unchanged.

To reconstruct the completed evidence without training:

```bash
uv run --frozen --extra cloud --extra ml python scripts/collect_domain_study.py --output artifacts/domain_feature_audit
```

## Promotion requirements

Continue feature research with explicit hypotheses and controlled ablations. Reuse
verified artifacts and preserve negative findings. Feature-only comparisons retain
candidate identity; retrieval comparisons report candidate ceilings and the final
ranking metric separately. Screening uses fitting data, and new configurations are
chosen on development data only.

Before final training, resolve or document a concrete data-based exclusion for each
high-value open family and preregister temporal confirmation of the selected feature
set. Budget-deferred or unmeasured work remains open. Do not repurpose the already
inspected evaluation as a fresh test. No guaranteed leaderboard score or arbitrary
percent-complete claim follows from a test count or a completed notebook.

The invalidated 0.93583 private score used full test sessions with future events and
is retained only as incident evidence. The corrected accepted file used the official
truncated prefixes. [Source audit and delivery](INFERENCE.md).

[Original temporal results](ROBUSTNESS.md) · [Model card](MODEL_CARD.md) ·
[Research coverage](FEATURE_RESEARCH.md) · [Retrieval experiments](retrieval-research.md)
