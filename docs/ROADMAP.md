# Project status and research direction

**Baseline delivered; the performance objective remains open.** The accepted private score is
**0.56842**, compared with the historical winner's **0.60503**: a gap of **0.03661**, or
**3.661 percentage points**. A completed engineering release and nine successful robustness
runs do not establish competitive performance or exhaust feature engineering.

## Active experiment: learned ranking features

The submitted ranker uses 100,000 fitting sessions, 20,000 selection sessions and at most
400 candidates per query. Its 102 retained features contain no learned embedding similarities.
The 1,482-formula catalog is heavily concentrated in co-visitation transformations; feature
count alone is not evidence that the important representation families have been explored.

The [first-place write-up](https://www.kaggle.com/competitions/otto-recommender-system/writeups/mrkmakr-1st-place-solution)
describes roughly 1,200 candidates, multiple neural retrievers, candidate/session neural
similarities, multi-hop co-visitation, objective-specific feature selection and negative
retention. Its single LightGBM ranker reached about 0.604 on the leaderboard; nine-model
averaging provided the final increment to about 0.605. This motivates improving signals and
training coverage before investing in a large ensemble. These are author-reported results,
not a reproduced comparison on this repository's cohort.

The [third-place implementation](https://github.com/TheoViel/kaggle_otto_rs) also uses
Word2Vec and matrix-factorization similarities among its 744 features. Its reported private
score is 0.60382. The first pilot tests this missing representation family without changing
our candidate pools or ranker hyperparameters.

| Controlled component | Preregistered pilot |
|---|---|
| Representation fitting | Historical events strictly before August 20, 2022, 22:00 UTC |
| Learned families | Skip-gram Item2Vec on all actions; separately on carts and orders |
| Embedding settings | 128 dimensions, window 10, 10 negatives, 3 epochs, seed 20260910 |
| Similarity features | 148 proposed columns: known-item flags, frequency, action/window coverage, last-item, mean, maximum, centroid and recency-weighted similarities |
| Quality screening | 100,000 fitting rows only; remove nonfinite, constant and redundant new columns |
| Matched arms | Existing 102 features; add all-action embeddings; add intent embeddings; add both |
| Ranking protocol | Same 100,000 fit / 20,000 selection sessions, negative samples, complete selection candidate pools and official Recall@20 |
| Reporting | Per-task recall, weighted recall, candidate ceiling, gain importance, paired selection differences and runtime |
| Compute boundary | One `ml.c7i.16xlarge` Processing job, at most 7,200 seconds; no standing endpoint |
| Recovery | Byte-verified native embedding epochs, feature partitions and ranker iterations in a separate S3 namespace |

Configuration: [representations.toml](../configs/representations.toml). The pilot is
**development**, not a new unbiased test result. It cannot read evaluation queries or replace
the accepted submission. Selection intervals are descriptive and do not correct selection
optimism or multiple comparisons. Multithreaded Word2Vec is not bit deterministic; the exact
realized vector artifacts are pinned by SHA-256 for reproducible downstream comparisons.

## Where the current model loses recall

The existing reference evaluation provides this diagnostic. It has already been inspected
and must not be relabeled as an untouched holdout for a newly chosen model.

| Official objective | Candidate ceiling at 400 | Selected ranker | Fixed candidate fusion |
|---|---:|---:|---:|
| Clicks | 0.64815 | 0.50565 | 0.52678 |
| Carts | 0.56902 | 0.42792 | 0.43021 |
| Orders | 0.75238 | 0.67575 | 0.58917 |
| Weighted | **0.68695** | **0.58439** | **0.53524** |

The 0.10256 gap between the candidate ceiling and ranker is theoretical headroom, not an
achievable gain forecast. Click ranking currently underperforms the fixed fusion by 2.114
percentage points, whereas order ranking improves strongly. Any fallback or blending policy
must therefore be selected on development data, not retrofitted to these reported outcomes.
Values come from [the audited evaluation](../reports/research/evaluation.json).

## Subsequent research gates

1. Evaluate the four matched feature arms. Retain useful learned signals based on incremental
   per-task utility, coverage, feature redundancy and paired development results. Preserve
   negative results; do not declare the family exhausted after a small pilot.
2. Test complementary candidates: task-conditioned neural retrieval and multi-hop co-visitation,
   with 400/800/1,200-candidate recall/latency frontiers. Refit every learned component at the
   correct cutoff; the earlier exploratory neural checkpoint is not automatically eligible.
3. Expand fitting sessions and compare objective-specific negative policies and feature sets
   only after identifying useful signals. Selection uses complete queries and counts targets
   missed during retrieval. Do not inject true items into candidate pools.
4. Preregister temporal confirmation before expanding model selection, explicitly documenting
   prior data exposure. Promote only after verified incremental gains; then run official-prefix
   inference and submit a genuinely different artifact. Update the executed notebook and Plotly
   comparison with measured results, preserving the current accepted baseline.

The first pilot has a two-hour compute ceiling plus provisioning/input setup. Its progress and
checkpoint timing determine the next estimate. There is no evidence-based deadline or guarantee
for reaching 0.60503; calendar age alone does not remove data, model or compute constraints.

SageMaker accepted **`otto-representations-fe780ed36e9e`** at **00:18 UTC on September 10**.
The [launch receipt](../reports/research/representation_run.json) pins the source commit,
source archive, input contract, checkpoint destination and bounded resources. The saved status
is an observation, not a live progress indicator. Monitor the running experiment with:

```bash
cd "$HOME/otto-recommender-system" &&
uv run --frozen --extra cloud python scripts/representation_status.py --watch
```

The monitor prints the managed job status and recent CloudWatch heartbeats. It does not launch
compute. This experiment is separate from the completed nine-cell robustness batch below.

## Verified baseline

**All nine frozen temporal validation cells are complete and independently audited.** Full inference on the official competition prefixes is also complete, and the downloaded gzip passed the exact session/action ledger, every top-20 list and full-file checksum validation.

| Milestone | Current verified result |
|---|---|
| Event processing and temporal protocol | 216.7M training events; 217 partitions; chronological role contracts |
| Retrieval and neural/ANN benchmarks | Executed notebooks 02–06, with their original scope preserved |
| Feature research | 1,482 explicit formulas; fitting-only screening; eight ablations per cell |
| Robustness | **9/9 audited** across three windows and three model seeds |
| Matched feature gain | **+1.791 to +2.180 percentage points** over compact controls |
| Original reference model | **0.584392** weighted Recall@20; remains frozen for submission |
| Highest observed offline score | **0.590759**, middle seed 20260908; a different evaluation cohort |
| Official-prefix full prediction | **1 verified file**, 1,671,803 sessions and 5,015,409 rows |
| Kaggle delivery | **Complete (after deadline)**; official-prefix artifact accepted and scored |
| Corrected Kaggle score | **0.56842 private / 0.56862 public**; observed in the submission details |
| Employer-facing publication | Canonical notebooks, model card, complete comparison and Plotly report |
| Earlier 50-file collection | **Not executed**; one full file is verified, not 50 |

## Completed closeout

The verified file was submitted to Kaggle after explicit final confirmation. Its accepted status and **0.56842 private / 0.56862 public** scores are recorded in the canonical receipt and Notebook 10. No further training, inference or user action is required for this release. The first 0.93583 private score is invalidated because its input contained future events. It is retained as incident evidence, not as a performance claim.

The broader previously requested 50-file collection remains unfulfilled and must not be described as complete. It would require distinct model or ensemble recipes chosen on fitting/selection evidence, cache reuse, and content deduplication. The present delivery prioritizes the single corrected submission requested for closeout; it does not start additional training or manufacture renamed copies.

## Runtime and monitoring

The managed remaining queue ran from **20:36 to 22:26 UTC on September 9**. Its final two training runs executed concurrently, and both audits succeeded. Corrected full inference completed at **21:43 UTC**. No validation or inference jobs remain in progress for this batch.

```bash
cd "$HOME/otto-recommender-system" &&
uv run --frozen --extra cloud python scripts/robustness_status.py
```

This read-only command reports actual SageMaker job names. An offline saved view is available with `--snapshot reports/robustness/batch/execution.json`. The queue used up to three simultaneous steps under the authorized scheduling amendment while preserving the frozen data, feature and evaluation protocol.

[Complete results](ROBUSTNESS.md) · [Submission and source audit](INFERENCE.md) · [Model card](MODEL_CARD.md)

## Further research

Certified neural retrieval within the same cutoffs, new sequence models, additional rankers, blending studies and online experiments are extensions. Current evidence supports repeatable offline feature gains on this historical dataset; it does not claim state-of-the-art performance or online lift.
