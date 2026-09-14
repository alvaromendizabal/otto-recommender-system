# Round 11 — Historical item–session latent affinities

## Hypothesis and novelty

A candidate may share a broader pattern of historical shopping sessions with several observed products even when its direct local transition counts are weak. Learning a compact representation from the training-anchor-selected historical item/session incidence structure could expose these indirect relationships to the fixed ranker. The saved 134-column control has graph and domain features but no latent-embedding columns in its supplied feature-order contract. This is a new representation class in this controlled comparison, not simply another weighting of a per-query neighbor vote.

The primary tests rarity weighting DURING representation learning, not the Round08 rarity-weighted neighbor support calculation. The ablation learns the same-size representation without item rarity weighting. That distinction must be retained when reporting positive or negative results.

## Construction

Select historical sessions using the earliest fold training anchors ONLY, then build binary historical item-by-session matrix B. Count one item membership per historical session irrespective of repeated events. Also build B_click, B_cart and B_order in the identical item/session coordinate system. Let N be the number of retained sessions and df_i the binary row sum. Both arms apply a session-column weight 1/sqrt(max(1, number of retained vocabulary items in that session)). Primary additionally applies item-row weight 1+log((1+N)/(1+df_i)); ablation applies item weight 1.

For each arm, learn a randomized truncated SVD of the weighted all-action matrix. Use up to 32 components, fixed seed 20260912, eight oversamples, two power iterations and QR normalization. Rank is min(32, smaller matrix dimension minus one). This gives TWO self-supervised factorizations total. There is no basis rotation tuned to target outcomes.

Project the all-action rows and each of the three action-specific rows onto the SAME session basis V. Normalize each resulting item vector to unit length; preserve genuine zero vectors as zero. All-action vectors encode query anchors, while action-specific vectors encode candidates. Separate bases are NOT naively compared by cosine. Compute the eight summaries per action described below, yielding 24 features per arm.

The all-action basis may underrepresent sparse order behavior; the action-specific projections and support diagnostics make that limitation observable. Session-length weighting is identical in both arms, so the ablation isolates item rarity weighting rather than two simultaneous changes. Vocabulary truncation can change df and session-length context; no claim of unbiased global IDF is made.

## Research rationale — external sources, not this experiment's results

A first-person third-place OTTO solution describes item embeddings (including Word2Vec and matrix factorization) and aggregates of item-to-session similarity as important feature families: https://github.com/TheoViel/kaggle_otto_rs . That motivates testing learned item relationships. Our incidence-SVD construction is not a reproduction of that solution, its training scale, or its score.

Sparse truncated SVD documentation supports the bounded representation-learning implementation: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html and https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html . Documentation establishes API behavior, not predictive utility for these data.


## Frozen comparison and interpretation

This is one predeclared 24-column family and one 24-column information-removal ablation. Each is independently appended to the saved 134-column control, yielding 158 columns. There is no combined 48-column arm and no carry-forward of losing prior-round features. Candidate budget stays 400 on the exact 4,096 fitting sessions. The two chronological validation folds total 2,048 queries. Three objectives use the official 0.1 / 0.3 / 0.6 click/cart/order weights and complete capped target denominators. There is a six-hour query embargo. Training targets are censored to the fold training boundary BEFORE negative sampling. The saved seed, 60-negative policy, 150 boosting rounds, 15 leaves, and other LambdaRank settings remain unchanged in protocol.json.

All six native controls must replay the same per-session validation hits and feature order before a new challenger fit. There are at most twelve new supervised ranker fits per round, not a hyperparameter search. Training-only diagnostic correlations, support, and available-vector rates do not trigger automatic feature removal or tuning. Learned representations are made before any target-dependent candidate subsampling. Complete validation candidate pools are scored. Native challenger models are reloaded for prediction parity and stored with immutable contracts.

A primary pooled delta >= +0.003, nonnegative deltas on both folds, and no pooled order-recall decline earns a separate confirmation proposal, never automatic retention. Primary-minus-ablation superiority does not rescue a losing primary. This cohort has informed repeated exploratory hypotheses. Paired bootstrap intervals are descriptive and do not correct for all prior selection or time dependence. Independent different-cohort / different-history confirmation remains required. No holdout, selection-role data, competition test set, or submission is accessed.

## Historical source and leakage boundary

The storage source is the existing certified Round08 retrieved-history cache, NOT all original OTTO history. BEFORE representation learning, restrict its historical sessions to postings for the earliest chronological fold TRAINING anchors only. Validation-query anchors cannot expand vocabulary or the learned basis. The same conservatively frozen basis is used in both later validation folds. This addresses the fact that the original cache union was assembled using the larger cohort’s observed anchors. It still has training-query-retrieval selection and tail-truncation biases. Learning across this training-selected history is broader than pooling top neighbors for a single query, but must not be described as a full-catalogue/full-history embedding system. All 4,096 current and 1,024 earlier study sessions remain excluded; every retained event is strictly before 1660687200000 ms (2022-08-16 22:00 UTC). Session IDs, product IDs, valid actions, monotonic timestamps, and original event indices are checked. Original event gaps are never compressed into adjacency.

Vocabulary selection uses only DISTINCT historical (session, item) frequency in the training-anchor-selected history. The top 60,000 items are retained with deterministic item-ID tie breaking. No validation target controls vocabulary membership. This is a declared vocabulary restriction, not an undisclosed sample. Fewer than 16 retained items stops preparation. Unknown or zero-vector items receive zero affinities and explicit availability measures. A large vocabulary/target coverage deficit would be a reason to inspect the source before another larger run, not proof that latent representations do not work.

The query uses its last four DISTINCT observed products, most recent first, from the certified cache. Candidate-self anchors are excluded so a product cannot get a trivial cosine of one merely by being revisited. Other revisit features remain in the 134-column control. Negative cosines remain negative. There are eight summaries for each candidate destination action: latest, maximum, mean, population standard deviation, reciprocal-recency weighted mean, cosine to the weighted centroid, latest-minus-older mean, and available-anchor fraction. The catalogue documents exact zero conventions. Candidate order is not a predictor; reordering candidates must preserve their per-item features within floating-point tolerance.

## Bounded stages and resource accounting

Run tests -> prepare representation inputs -> learn representations -> build features -> replay controls / screen -> report -> save notebooks -> bundle. Never launch the entire sequence as an unattended shell loop. Each stage is an explicit user action. A test pass is a functionality gate, not predictive evidence. Existing environment dependencies include NumPy, SciPy, scikit-learn, LightGBM, Polars, Plotly and DuckDB; versions are recorded. Nothing auto-installs. A missing dependency stops with a specific name to report.

Useful-work limits are 220s preparation, 280s representations, 240s feature generation, 240s screen and 60s report. Outer limits are 120s tests, 240s preparation, 300s representations, 260s features, 260s screen, 90s report. They are caps, not estimates. Four numerical worker threads, 26 GiB worker RSS stop and 10 GiB free disk are required. Dataset caps: 400,000 retained historical sessions, 10 million retained events, 60,000 vocabulary items and eight million combined typed pair records. Exceeding a cap stops rather than silently dropping pairs or raising budgets.

UTC heartbeat messages expose stage progress. Sparse input units, representation units, 64-query feature chunks and native ranker models have checksums and immutable receipts. Saved units are reused only when code, schema, protocol, dependencies and historical-source identity match. A planned pause or failure is not permission to retry automatically. Download its return ZIP for diagnosis; do not delete orphan evidence. Expensive single factorizations cannot be checkpointed internally; a hard interruption may require revisiting that one unit after review. Completed separate units remain intact.

No new raw JSON scan, historical Parquet scan, graph rebuild, candidate expansion, AWS operation, Git write, or submission occurs. These two rounds jointly allow 24 supervised challenger fits; self-supervised representation factorizations are separately counted, not hidden as zero compute. Inference would require the frozen historical basis and the same vocabulary/anchor/missing conventions; applicability to full competition inference remains to be established.

## Evidence and next decision

Nine result charts report matched scores, descriptive uncertainty, fold deltas, objective-specific recall, weighted hit contributions, primary/ablation representation differences, redundancy, candidate ceilings and training-positive support. Representation notebooks additionally show matrix density, learned rank/singular values and input statistics. Save executed notebooks to preserve inline Plotly payloads. External HTML is supplementary.

A gain must be attributable to the specified representation rather than changed candidates, rows, denominator or ranker. A loss is recorded, not hidden. A sparse or zero PPMI matrix, high out-of-vocabulary rate, inadequate action evidence, instability across folds, or strong redundancy prompts diagnosis. Repeatedly retuning dimensions, context windows or smoothing on these same labels is not the next default. Different historical windows, larger justified history, richer embeddings and candidate availability remain distinct hypotheses, not claims that this one failed configuration exhausts the family.

## Preparation and execution status

Prepared for manual execution. No project tests, notebooks, representation fits or ranker fits were executed by the assistant for this delivery. Static syntax and artifact consistency inspection do not establish runtime correctness. Tests are supplied for the user, including pure-function edge cases, synthetic sparse representations, native ranker/reload fixtures, complete denominators, checkpoint behavior and report/notebook integrity. The mandatory installed DuckDB/Parquet smoke runs in the user's environment; no database fallback is used on real source data. The original experiment checkout stays pinned and untouched. Publish from a separate clone, not from that checkout.
