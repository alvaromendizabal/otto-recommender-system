# Retrieve broadly. Rank carefully. Deliver reliably.

**Alvaro Mendizabal · Recommendation / learning to rank / ML systems**

A large-scale session recommender with multi-source retrieval, objective-specific ranking, controlled feature studies, and verified batch delivery. Public reference training and bounded evidence review serve different reproducibility needs.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Applied AI | **Keep retrieval coverage separate from ranking quality.** Measure what candidate generation can reach and what the final ranked list actually recovers. | [Code](../src/otto_recsys/ranking/candidates.py) · [Tests](../tests/test_candidates.py) · [Evidence](../docs/EVIDENCE_MAP.md) |
| Data science | **Make feature gains a matched comparison.** Compare controlled representations on the same temporal population and preserve paired uncertainty estimates. | [Code](../src/otto_recsys/ranking/features.py) · [Tests](../tests/test_ranking_features.py) · [Evidence](../docs/PORTFOLIO.md) |
| ML engineering | **Recover training without losing model identity.** Use native model reloads, full-query ranking metrics, and atomic iteration checkpoints. | [Code](../src/otto_recsys/ranking/lambdarank.py) · [Tests](../tests/test_lambdarank.py) · [Evidence](../docs/INFERENCE.md) |
| ML engineering | **Offer a reference workflow without the private recipe.** Provide explicit review, native training, verification, and resource-bounded full-data research modes. | [Code](../reproduce.py) · [Tests](../tests/test_public_demo.py) · [Evidence](../docs/QUICKSTART.md) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `reference-review` | python | [`reproduce.py`](../reproduce.py) |
| `recommendation-interface` | node | [`tests/public_demo_ui.test.mjs`](../tests/public_demo_ui.test.mjs) |

## What is actually reproduced

The selected Python command runs the existing public reference review and synthetic recommendation report. Optional Node checks exercise the report interface. The existing native training and full-data reference workflows remain available, separate from the later private neural release.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The fingerprint covers this curated review surface, **not every transitive dependency or every repository file**. Checksums establish consistency, not independent authentication of historical scores. HTML links follow the current default branch; the receipt is the record of the local bytes inspected. The static page is explicitly an inspection brief, not an execution receipt.

## Public/private boundary

Raw event populations, private embeddings, current neural checkpoints, exact selected blend rules, complete prediction rows, and private AWS orchestration remain excluded.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
