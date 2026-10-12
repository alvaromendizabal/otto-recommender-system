# Research status · October 12, 2026

**The portfolio release is complete. Active experimentation is concluded; the verified delivered policy is preserved.** This is a closeout, not an assertion that every research hypothesis has been exhausted or that private full-system restoration has been proven.

## Delivered result

The recorded post-competition release scores **0.57586 private / 0.57601 public**. It combines established click/cart policies with the promoted neural-similarity orders ranker. The [original receipt](../reports/submissions/similarity_stack_20260925.json) records 1,671,803 sessions and 5,015,409 output rows. There is no online deployment or measured business-lift claim.

The closeout audit independently matched the existing submitted CSV's complete SHA-256 against that release receipt. It performed no fitting, inference or new submission. [Sanitized current closeout](../reports/closeout/summary.json).

## Final completed modeling comparison

On the previously exposed 20,000-query orders selection population, a learned retrieval expansion increased candidate-oracle hits from 2,556 to 2,600. That passed its declared retrieval screen. Its first candidate-aware ranking comparison did not qualify:

| Policy | Correct top-20 order hits |
|---|---:|
| Existing incumbent | 2,326 |
| Matched retrained ranker | 2,312 |
| Primary retrained ranker | 2,302 |
| Frozen incumbent applied to expanded candidates | 2,315 |

The candidate denominator was 3,370 capped order targets. **Reachable targets and ranked hits are different measurements.** The primary was rejected; neither it nor its matched control replaced the release. A subsequent objective experiment was blocked before fitting because its required cached inputs were unavailable. It is untested, not a negative scientific result.

## Reproduction status

A new public entry point connects raw JSONL validation, conversion, chronological corpus construction, retrieval, feature computation, native fitting, evaluation and prediction validation. The small native profile uses invented data and short fits; a reader-supplied official training file enables the published reference protocol. [Run it](QUICKSTART.md).

The later neural-enhanced release is not reproducible merely by cloning and running the fixture. The private full-data pipeline has unresolved exact-replay and artifact-availability requirements. The closeout did not verify a full durable backup. No unfavorable result or missing artifact is hidden behind a new filename, changed threshold or synthetic substitute.

## Historical evidence remains available

[The v27 owner summary](../reports/latest_execution/summary.json) and [archived frontier record](../reports/research/current_frontier_20261005.json) describe earlier stages, not the present status. Their original decisions and measurements remain available for audit. The corrected click/cart policy was rejected against the actual deployed comparator; it is not a pending release.

Further research would require a new explicit decision, durable eligible inputs, a declared comparator and budget, and a complete validated release path. No further experimental run is scheduled by this closeout.
