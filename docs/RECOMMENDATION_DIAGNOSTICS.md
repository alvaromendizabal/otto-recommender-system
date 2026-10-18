# Recommendation diagnostics

The public review answers a concrete modeling question: **are relevant items missing
from the candidate pool, or are available items failing to reach the top 20?**

Run `python3 -B -S review/run.py --output ../otto-review` with Python 3.11+, then
open `../otto-review/index.html`. The linked diagnostic report uses all 48 invented
evaluation sessions. The interactive walkthrough remains available beside it.

## Independent, complete evaluation

The demo seals `predictions.json` before writing `evaluation.json`, which contains
the complete invented target population and observed-prefix lengths. Its completed
manifest binds both exports. The auditor reads those files directly; it does not
import the generator, retriever, ranker, or original evaluator. Exact session coverage,
all three objectives, unique recommendation items, candidate membership, and strictly
future target timestamps are checked before calculating any result.

For each objective, distinct future target items are deduplicated within a session.
The denominator is the sum of `min(20, distinct targets)` across all sessions.
Sessions without targets contribute zero to that objective's denominator and remain
part of the audited query population.

| Count | Definition |
|---|---|
| Hits | Distinct target items present in the final top-20 list |
| Candidate hits | Distinct target items reachable from the saved candidate pool, capped at 20 per session |
| Ranking shortfall | Candidate hits minus achieved hits |
| Retrieval shortfall | Capped target denominator minus candidate hits |

The identity `hits + ranking shortfall + retrieval shortfall = denominator` must
hold. Counts are pooled before rates are calculated; averaging per-session recall
would be a different metric. Clicks, carts, and orders receive weights 0.1, 0.3,
and 0.6 respectively. Objectives with zero targets contribute zero evaluable weight.
Two prefix-length slices cover every query and expose their own exact counts.

## Interpretation and scope

Candidate coverage is an upper bound for the saved pool, not a forecast that a better
ranker will recover every reachable target. The report locates theoretical headroom;
it does not claim a model improvement. These invented sessions demonstrate software
and evaluation behavior, not historical model quality or online business lift.

The same command verifies a completed review on repeated invocation. A changed
source, prediction, target export, or report invalidates reuse. Use a fresh output
directory after an intentional source update; prior results are preserved.

To audit a separately generated demo:

```bash
python3 -S scripts/run_public_demo.py --output artifacts/standalone-demo
python3 -S tools/audit_recommendations.py --demo-output artifacts/standalone-demo --output artifacts/recommendation-audit.json --html artifacts/recommendation-diagnostics.html
```

[Implementation](../tools/audit_recommendations.py) ·
[Metric and boundary tests](../tools/test_recommendation_audit.py) ·
[Public reproduction scope](PUBLICATION_SCOPE.md)
