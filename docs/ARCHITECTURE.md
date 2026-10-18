# Retained system architecture

The public implementation follows the feature-and-ranking approach that produced the controlled reference. It is not a catalog of every model explored during development.

## Data → features → ranking → validated output

| Stage | Retained implementation | Contract |
|---|---|---|
| Input validation and conversion | [data/](../src/otto_recsys/data/) | Typed JSONL, chronological events and complete conversion receipts |
| Temporal corpus | [protocol.py](../src/otto_recsys/research/protocol.py) | Disjoint history, fitting, selection and evaluation roles |
| Retrieval | [retrievers.py](../src/otto_recsys/research/retrievers.py) | Co-visitation computed from eligible historical events |
| Representation | [features.py](../src/otto_recsys/research/features.py) | Observed-prefix and historical features; no query targets |
| Fit and selection | [training.py](../src/otto_recsys/research/training.py), [study.py](../src/otto_recsys/research/study.py) | Native LightGBM rankers and sealed model selection |
| Evaluation | [evaluation.py](../src/otto_recsys/research/evaluation.py) | Capped pooled Recall@20 and matched populations |
| Batch inference | [inference.py](../src/otto_recsys/research/inference.py) | Three complete lists per session, unique IDs and verified partitions |
| Recovery | [checkpoints.py](../src/otto_recsys/checkpoints.py), [materialize.py](../src/otto_recsys/research/materialize.py) | Exclusive writer, compatible identities and atomic completion |

## Delivered release versus public reference

The retained scored release preserves established click/cart routing and adds promoted similarity features to orders ranking. Its private similarity artifacts and routing recipe are deliberately not re-created by the public fixture. The public reference runs historical retrieval, feature construction and native task rankers; its boundaries are explicit in the [model card](MODEL_CARD.md).

## Review and reproducibility

[reproduce.py](../reproduce.py) exposes review, train, research and verify modes. Review uses authored records without external packages. Train exercises the real converter and native LightGBM components on a small invented dataset. Research accepts user-supplied official JSONL and remains plan-only until `--execute` is supplied. Verify rejects changed or unlisted artifacts.

[Recommendation diagnostics](RECOMMENDATION_DIAGNOSTICS.md) independently reconcile exported target counts, rather than trusting the engine's aggregate score. [Quickstart](QUICKSTART.md) describes resource limits and the reproducible paths.
