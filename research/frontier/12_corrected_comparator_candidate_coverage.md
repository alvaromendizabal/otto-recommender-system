# Corrected comparator and candidate-coverage frontier · October 5, 2026

This historical update records the OTTO research diagnosis after the deployment-parity correction, through the public snapshot of October 6, 2026, 02:15 UTC. It preserves the results and decisions available then; it is not a claim about the latest private experiment or a direction to launch another run. Public material and the bounded historical native-model replay are described in the [publication scope](../../docs/PUBLICATION_SCOPE.md).

## Verified external state

The strongest verified post-competition release remains **0.57586 private / 0.57601 public** on submission **56542128**.

No later offline result is presented as a leaderboard improvement.

## Comparator correction completed

A deployment-parity audit found that one newer click/cart validation path had not reconstructed the exact deployed objective-specific routing policy. The corrected audit then:

- reproduced the incumbent click/cart top-20 membership on **4,096 / 4,096** preserved official prefixes;
- verified **423 / 423** archived statistic parts covering **432,492 sessions**;
- recomputed the frozen challenger against the true incumbent.

The corrected challenger showed:

| Quantity | Corrected result |
| --- | ---: |
| Weighted Recall@20 gain | **+0.00395690** |
| Click-hit gain | **−3,529** |
| Cart-hit gain | **+1,922** |

The frozen qualification contract required a non-regressing click result. The challenger therefore **failed qualification and was rejected**. The externally verified release remained unchanged.

This is a deliberate validation outcome: a positive aggregate point estimate does not override a failed objective-level gate.

## Controlled objective research after the correction

The project then tested a top-20-aligned training objective under matched controls.

Across **3,502 five-fold development sessions**, the strongest development arm recovered **2,806 cart targets** versus **2,696** for the incumbent reference, an aggregate gain of **110 cart targets**.

That result is retained as **development evidence only**. The candidate replayed exactly on its development population, but no unused independent confirmation cohort was certified. The model was therefore not promoted.

A stricter forward-time history-only study later compared ordinary pairwise ranking with the top-20 objective on **16,000 chronological evaluation queries**. The top-20 arm recovered **2,180** cart targets versus **2,161** for pairwise, but its descriptive interval crossed zero. That bounded configuration was closed rather than tuned until it passed.

## Archived diagnosis: candidate availability and ranking headroom

The time-controlled diagnosis separates ranking error from retrieval error on this cohort.

| Diagnostic | Cart targets |
| --- | ---: |
| Capped evaluation denominator | **4,778** |
| Archived source-intent reference ranking | **2,181** |
| Existing candidate-pool ceiling | **2,824** |
| Misses still inside the candidate pool | **643** |
| Misses outside the candidate pool | **1,954** |

Roughly three quarters of this reference's remaining misses are outside its candidate pool. The **643** within-pool misses also leave ranking headroom; the counts do not determine which intervention will deliver a gain.

The bounded retrieval experiments recorded in this snapshot were designed to attack that gap:

- nearest-session memory added only single-digit equal-budget coverage;
- direct item-to-cart association added **4** retrievable targets;
- a rank-96 spectral approximation regressed by **22** targets at equal budget;
- even the full sampled cart-output catalog raised the diagnostic ceiling only from **2,824 to 3,005**.

The archived research question was: **which point-in-time item evidence could expand the candidate vocabulary?** This hypothesis did not exclude improving ranking within existing pools. Candidate coverage remains a diagnostic ceiling, not an achieved recommendation score or a prerequisite for every ranking study.

## What this demonstrates

The research record intentionally includes negative results because they drive architecture decisions:

1. comparator identity is treated as part of the metric contract;
2. deployment-parity failures block release;
3. development-only gains stay development-only without independent confirmation;
4. retrieval ceiling and achieved ranking are reported separately;
5. failed mechanisms are closed instead of repeatedly retuned;
6. completed AWS artifacts remain resumable and content-addressed.

## Public reproduction boundary

Public GitHub contains selected implementation kernels, aggregate evidence, executed notebooks, tests, CI contracts, architecture documentation, and source attribution.

The repository excludes full competition datasets and prediction populations, current private models and embeddings, private runners, credentials, and exact private orchestration. It already includes a bounded historical replay with three reference models and eight official-prefix examples; that exception is explicit in the [publication scope](../../docs/PUBLICATION_SCOPE.md).

## Release state at this snapshot

| Component | State |
| --- | --- |
| Verified competition release | **0.57586 private / 0.57601 public** |
| Corrected heterogeneous click/cart challenger | **Rejected** |
| Top-20 / uniform candidate | **Development evidence only; not promoted** |
| Official-feature engineering replay | Passed on preserved sample |
| Archived research hypothesis | Candidate-vocabulary / coverage expansion |
| New Kaggle submission | None |

A later public result needs its own evidence and qualification; this snapshot does not authorize reopening a closed experiment or promoting a development-only challenger.
