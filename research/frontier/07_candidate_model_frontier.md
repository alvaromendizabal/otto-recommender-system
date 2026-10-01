# Candidate, representation and model frontier · October 1, 2026

This is an aggregate, employer-facing snapshot of the work completed after the September 29 transition/model update. AWS remains the canonical private execution workspace. Raw events, row-level labels and predictions, cohort identifiers, full embeddings, fitted model binaries, optimizer state, private runners, credentials and account logs are intentionally excluded.

## Verified competition state

The strongest verified post-competition Kaggle result remains **0.57586 private / 0.57601 public**, submission **56542128**. The recorded historical private winner is **0.60503**, leaving a **0.02917** private-score gap. No study below produced a newer leaderboard submission.

The research target of **0.65000** is intentionally aspirational and is not presented as an achieved or expected score.

## Transition/XGBoost reserved evaluation: closed

The 496-feature transition/source representation previously passed fitting and selection with a fixed blend family. The selected arm, `zmean_all3`, completed the full **432,492-session** reserved temporal evaluation:

- incumbent weighted Recall@20: **0.591589**
- challenger weighted Recall@20: **0.594255**
- gain: **+0.002665**
- additional order hits: **+308**
- paired 95% interval: **[+0.002021, +0.003271]**
- chronological-half gains: **+0.002910 / +0.002430**

The effect was positive and stable, but the frozen evaluation gate required at least **+0.003 weighted gain and +347 order hits**. The branch therefore closed without deployment or submission.

The recovery itself was accelerated from a serial evaluator to a bounded block-parallel evaluator on the same 32-vCPU instance. This is an engineering improvement, not a model-quality claim.

## Candidate-ceiling escape: MF + neural retrieval

A new retrieval study added exact full-catalog general and intent factor retrieval to the existing candidate system and v42 neural retrieval. On **20,000 fitting sessions**, the strongest `base_all_800` pool moved weighted candidate ceiling from **0.683999 to 0.716214**, a **+0.032215** increase.

Recovered targets versus the established 400-candidate pool:

- clicks: **+1,446**
- carts: **+261**
- orders: **+69**

This is a candidate oracle ceiling, not achieved recommendation recall. The study established that relevant products were missing from the previous pool and justified candidate-aware integration experiments.

## Candidate-aware integration: two negative controls

A three-objective, 527-feature LambdaRank study retrained on the expanded candidate pool. Compared with the same fitted models restricted back to the original candidates, unrestricted use of the 800-candidate pool **regressed weighted OOF Recall@20 by 0.004683**, with **-205 clicks / -35 carts / -11 orders**.

A second study protected the incumbent head and allowed only a small tail of novel candidates to compete. The best policy, protecting 18 positions and requiring two-source consensus, still lost **0.001277 weighted Recall@20** with **-61 clicks / -9 carts / -3 orders**.

Decision: the tested tree-based MF/v42 insertion recipes are closed. The retrieval representations remain useful diagnostic and feature sources.

## Action-specific Word2Vec retrieval: closed

Three point-in-time Word2Vec channels were trained for click/time, cart and order transition evidence. Replacing stronger sources regressed, while appending Word2Vec candidates out to roughly 1,100 candidates recovered **zero additional targets** beyond the MF+v42 pool.

Decision: action-specific Word2Vec retrieval is closed unchanged. This does not invalidate earlier embedding-similarity feature ideas.

## Lag-conditioned Seq2Seq retrieval: closed as a retrieval lane

A lag-conditioned sequence representation adapted the public third-place mechanism with 10-item histories, temporal context and objective-conditioned retrieval. The strongest append policy increased the MF+v42 candidate ceiling by only **+0.000819**, recovering **+27 clicks / +14 carts / +0 orders**.

The frozen candidate gate required **+0.0035**, at least **+10 order targets**, and positive recovery in at least two objectives. The retrieval lane is therefore closed, while the learned representation remains eligible as feature evidence.

## Broad third-place feature recreation

A 579-feature candidate/session representation added MF, Word2Vec and Seq2Seq similarity aggregates with position/time/action weighting. On fitting-only OOF it improved weighted Recall@20 by **+0.000813**, with **+7 clicks / -15 carts / +9 orders**. The branch failed its frozen gate.

The final CPU-side reconstruction added within-session rank transforms, week/day/hour popularity and ratios, candidate/session action counts, thresholded embedding similarities, an XGBoost binary classifier and fixed blends. The resulting representation contained **785 features**.

Best arm: fixed standardized-score mean (`zmean`):

- weighted OOF gain: **+0.003234**
- hit gains: **+54 clicks / +3 carts / +17 orders**
- nonnegative folds: **4/5**
- worst fold: **-0.000364**

The preregistered gain gate was **+0.0035**. The arm missed it by **0.000266**, so selection remained closed. RRF reached **+0.002985** and the complete XGBoost classifier alone regressed.

Decision: the bounded CPU boosted-tree / broad-feature recreation is closed under its current gates. The result is preserved as near-threshold evidence, not promoted after the fact.

## Leading-solution reproduction status

First-place source: `mrkmakr/OTTO-Multi-Objective-Recommender-System`, pinned commit `befc46bdfb747a1bcd840bd04af7ef2e84052b9c`.

- v42-derived sequence evidence: independently adapted, trained, evaluated and promoted through similarity features.
- v31-style complementary attention representation: independently adapted and fitting-only evaluated; tested candidate union regressed.
- v27/v29: mechanism audit complete; GPU execution is the next frontier. No result is claimed yet.
- v15/v18/v21/v23: still not claimed as reproduced.

Third-place source: `TheoViel/kaggle_otto_rs`, pinned commit `b1507a8728cfed4edb7c36a48417c8603a697fcc`.

- matrix-factorization candidate retrieval: adapted and shown to increase candidate ceiling materially.
- action-specific Word2Vec retrieval: adapted and closed for zero incremental target recovery beyond MF+v42.
- lag-conditioned sequence retrieval: adapted and closed as a weak incremental retrieval lane.
- broad similarity/popularity/rank-transform features plus XGBoost and fixed blends: substantially adapted and evaluated; the best blend was near-threshold but did not qualify.

These are independent adaptations of mechanisms, not copied checkpoints, predictions or weights.

## Current next frontier

The next bounded study moves the major remaining first-place mechanisms to GPU rather than extending the CPU tree family. It targets v29/v27-style long-context, multi-positive, task-conditioned and hard-negative sequence learning, with v29 adding attention.

No GPU outcome, new leaderboard score, or new submission is claimed in this snapshot.

## Publication boundary

Public files contain aggregate metrics, source attribution, frozen gate definitions, decisions and reproducibility contracts. Private data, predictions, embeddings, model binaries, optimizer state, exact orchestration, cloud-account details and secret-bearing artifacts remain outside GitHub.

A scientifically valid rejection counts as a successful execution. Engineering failures are tracked separately. Candidate coverage/ceiling is never presented as an achieved model score.

