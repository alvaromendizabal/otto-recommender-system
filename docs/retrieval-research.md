# Retrieval development study

The accepted competition baseline is 0.56842 private / 0.56862 public. The historical winning private score is 0.60503. Development selection scores below use a different cohort and are not leaderboard estimates.

The learned-feature pilot did not support promotion: the baseline achieved 0.59952343 and intent embeddings achieved 0.59963590 on the same 20,000 selection sessions. The paired descriptive 95% interval for the difference spans zero. Full results and provenance are retained in `reports/research/representation_results.json` and `representation_run.json`.

## Domain hypothesis

Shopping sequences are directional: viewing an accessory after a product does not imply the reverse transition is equally useful. Long comparisons and delayed purchases can also place relevant products beyond three intervening actions. Our original graph is symmetric, uses a three-position window, and retains 40 neighbors per scoring channel.

The [winning implementation](https://github.com/mrkmakr/OTTO-Multi-Objective-Recommender-System/blob/main/codes/otto/scripts_covisitation/covisitation.py) constructs previous-item to future-item transitions. Its variants include a 20-position lookback, action-dependent weights, and 100–300 neighbors. This motivates testing direction and horizon; it does not establish that transplanting these settings will improve this system.

## Matched experiment

| Arm | Direction | Position window | Neighbors per channel |
|---|---|---:|---:|
| baseline | symmetric | 3 | 40 |
| wide_symmetric | symmetric | 20 | 100 |
| wide_forward | forward | 20 | 100 |

All arms retain 400 candidates, the same 102 baseline features, chronological 100,000 fitting / 20,000 selection sessions, 60 sampled negatives, task rankers, and training seed. The wide-arm contrast isolates direction. The baseline comparison jointly tests width and horizon; it cannot attribute the gain to either individually. Per-session deduplication, six-hour decay, action weights, 30-event historical tail, and 24-hour maximum pair interval remain fixed. This is a controlled adaptation, not a reproduction of the winner's full system.

The baseline must reproduce 0.5995234305889497 before wider graphs are fitted. Every arm reports final weighted Recall@20, task metrics, fixed-fusion scores, candidate ceilings, and paired selection differences. Models, graph partitions, feature partitions, complete-session hit statistics, and SHA-256 contracts persist independently. Candidate discovery and feature generation receive observed prefixes; labels are joined afterward. Graph fitting accepts only history preceding the frozen cutoff. Evaluation queries are never requested by this development job.

Selection intervals are exploratory: repeated development on this cohort creates selection optimism. An improved arm requires a separately preregistered temporal confirmation before full-data training and an official-prefix Kaggle submission. No new submission or leaderboard improvement is implied by completed code or higher candidate coverage.

## Execution

`configs/retrieval_study.json` declares the experiment. `otto_recsys.cloud.retrieval_job` runs it through the verified `scripts/processing_research.py` bootstrap with task `retrieval`, a fresh checkpoint prefix, the original corpus, and original cached model inputs. No embedding retraining or evaluation archive is needed. One capped instance runs arms sequentially to bound memory; feature generation uses four processes and ranker fitting uses 32 threads. The configured managed runtime cap is two hours, not a promise that the experiment finishes within it. Completed parts can resume under the identical source and input contracts.
