# Retrieve broadly. Rank carefully. Deliver reliably.

**Alvaro Mendizabal · Recommendation / learning to rank / ML systems**

A large-scale session recommender with multi-source retrieval, objective-specific ranking, controlled feature studies, and verified batch delivery. Public reference training and bounded evidence review serve different reproducibility needs.

[Project overview](../README.md) · [Printable inspection brief](index.html) · [Review configuration](project.json)

## Review the engineering, not just the headline

| Focus | Decision to inspect | Follow the proof |
|---|---|---|
| Applied AI | **Keep retrieval coverage separate from ranking quality.** Measure what candidate generation can reach and what the final ranked list actually recovers. | [Code](../src/otto_recsys/research/retrievers.py) · [Tests](../tests/test_research_retrieval.py) · [Evidence](../docs/ARCHITECTURE.md) |
| Data science | **Make feature gains a matched comparison.** Compare controlled representations on the same temporal population and preserve paired uncertainty estimates. | [Code](../src/otto_recsys/research/features.py) · [Tests](../tests/test_research_features.py) · [Evidence](../docs/MODEL_CARD.md) |
| ML engineering | **Recover training without losing model identity.** Use native model reloads, full-query ranking metrics, and atomic iteration checkpoints. | [Code](../src/otto_recsys/research/training.py) · [Tests](../tests/test_research_training.py) · [Evidence](../docs/QUICKSTART.md) |
| ML engineering | **Offer a reference workflow without the private recipe.** Provide explicit evidence review, native reference training, verified reuse and output inspection. | [Code](../reproduce.py) · [Tests](../tests/test_public_reproduction.py) · [Evidence](../docs/QUICKSTART.md) |

The linked tests are review entry points, not a claim that this compact command executes the full research suite. The project’s canonical reproducibility guide describes the broader environment and tests.

## One command, an inspectable result

From a checkout with Python 3.11 or newer:

```bash
python3 -S review/run.py --output /tmp/ml-review
```

Open `/tmp/ml-review/index.html`. The directory contains an accessible HTML brief, `receipt.json`, `manifest.json`, and logs for each executed command. No Python package installation, model download, cloud credentials, or external dataset is required. On Windows, substitute a writable directory outside the checkout for `/tmp/ml-review`.

The generated brief links **Session walkthrough** and **Recommendation diagnostics**.
`recommendation-diagnostics.html` reconciles hits, ranking shortfall, and retrieval
shortfall for every objective and observed-prefix slice. The corresponding
`recommendation-audit.json` records the complete counts and source identities.
The audit reads all 48 invented sessions from the sealed JSON exports; it does not
import the recommender, target generator, or original metric implementation.

To execute the browser engine checks as well, install Node.js and use a separate output directory:

```bash
python3 -S review/run.py --output /tmp/ml-review-node --with-node
python3 -S review/run.py --output /tmp/ml-review-node --with-node --check
```

The default path does not execute Node checks and never labels them as passed. CI runs both the Python and browser checks on Python 3.11 and 3.13. It retains the exact review artifacts with the workflow run.

| Check | Runtime | Inspected entry point |
|---|---|---|
| `reference-review` | python | [`reproduce.py`](../reproduce.py) |
| `independent-recommendation-audit` | python | [`tools/audit_recommendations.py`](../tools/audit_recommendations.py) |
| `recommendation-interface` | node | [`tests/public_demo_ui.test.mjs`](../tests/public_demo_ui.test.mjs) |

## What is actually reproduced

The selected Python command runs the existing public reference review and synthetic recommendation report. Optional Node checks exercise the report interface. The maintained native reference training is separate from the later delivered release, which additionally uses private similarity-feature and routing artifacts.

A repeat invocation verifies the saved bundle rather than rerunning completed checks. `--check` performs verification only. A changed linked source file, missing evidence, changed artifact, failed command, invalid path, or mismatched check mode causes a nonzero exit. Use a fresh output directory after an intentional source change; stale evidence is never silently relabeled as current.

Each command has a default 90-second deadline, configurable with `--timeout`. The runner records UTC time, interpreter version, elapsed command times, command identities, and SHA-256 hashes for its configuration, template, invoked entry points, linked implementations, tests, and evidence. The source inventory also covers every matching file in the declared public source trees, including imported Python modules, browser engines, fixtures and lockfiles. Adding, removing or editing a covered file invalidates reuse. Failed runs keep diagnostic logs in a separately named failed-run directory and never produce a success receipt.

The exact `source_scopes` and `source_files` are inspectable in [project.json](project.json). This is an explicit public source inventory, not automatic discovery of every dependency: external package binaries, private training inputs and historical cloud state remain outside its scope. Checksums establish consistency, not independent authentication of historical scores. A receipt from a clean Git checkout links code to that immutable commit; a source archive or modified checkout uses the default branch and records that limitation. Python and optional Node runtime versions are recorded. The static page is explicitly an inspection brief, not an execution receipt.

Verification also binds each command to its declared entry point, arguments and saved log, then regenerates the HTML from the receipt to detect an inconsistent presentation. It rejects source symlinks, incomplete inventories and altered outputs. These are consistency checks; an attacker able to replace both the source and every receipt is outside this unsigned local verification model.

## Public/private boundary

Raw event populations, private embeddings, private fitted checkpoints, exact selected blend rules, complete prediction rows, and private AWS orchestration remain excluded.

This review adds no training result and changes no historical score, selection decision, notebook output, source attribution, or license. A synthetic demonstration, an aggregate evidence check, public reference training, and exact private-model replay are different reproducibility levels.

## Verify the review kit itself

```bash
python3 -B -S -m unittest discover -s review -p test_run.py -v
python3 -B -S -O -m unittest discover -s review -p test_run.py
python3 -S review/run.py --check-page
```

After an intentional configuration or template change, regenerate the static brief with `python3 -S review/run.py --render`, inspect it, then run the checks above. The kit needs no JavaScript or network access to render its HTML. It uses semantic headings, keyboard-visible links, a skip link, responsive layouts, and print styling.
