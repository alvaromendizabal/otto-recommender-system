# Contextual routing and sequence frontier

This snapshot records the two nonlinear follow-up families created after the source-aware joint click/cart recipe failed independent selection.

## Motivation

The selection result showed asymmetric transfer: click ranking improved materially while cart ranking regressed. The next experiments therefore focused on **conditional information use** rather than adding more flat source features.

## CRAFT-style contextual transport

Four cart rankers were evaluated across five OOF folds:

- `flat_mlp`
- `craft_gate`
- `craft_cross`
- `craft_moe`

All **20/20 GPU fits** completed on an NVIDIA L4.

Best arm: `craft_gate`

Result:

- combined deployment-aligned gain: **−0.002352**
- cart hits versus deployment incumbent: **−71**
- cart hits versus source400: **−28**
- combined nonnegative folds: **0/5**

Decision: **STOP_CRAFT_V1**

The result was a clean scientific rejection, not an execution failure.

## Candidate-conditioned temporal sequence interaction

A second family modeled session chronology directly.

Architectures:

- residual MLP control
- DIN-style candidate-conditioned cross-attention
- BST-style sequence self-attention
- behavior-specific cross-attention

Each architecture was trained across five OOF folds: **20/20 fits**.

Policies were evaluated without extra model fits:

- all sessions
- fixed short-only
- fixed long-only

Best arm:

**`din_xattn__long_only`**

Result:

- deployment-aligned combined gain: **+0.001678**
- cart hit gain versus deployment incumbent: **+12**
- cart improvement versus source400: **+55**
- combined folds nonnegative: **5/5**
- cart folds nonnegative: **4/5**
- worst combined fold: **+0.000597**
- worst cart fold: **−0.000239**

The arm was stable but below the frozen +0.003 combined promotion threshold.

Decision: close the standalone sequence recipe while preserving its OOF predictions.

## Engineering evidence

These experiments also hardened the execution stack:

- CPU and CUDA interpreters are capability-checked separately;
- inference uses `torch.inference_mode()`;
- raw OTTO actions 1/3/6 are explicitly mapped to embedding indices 1/2/3;
- candidate/vector indices are validated before CUDA;
- per-fold models checkpoint independently;
- large reusable caches live on NVMe;
- child-process failures return their real tracebacks;
- one owner execution produces one return ZIP.

## Why the negative results still mattered

CRAFT and sequence models made **different errors** from source400 and from one another. That is exactly the setting where OOF ensembling can add value even when individual challengers miss a standalone gate.

The next experiment therefore moved to heterogeneous OOF stacking rather than training another near-duplicate sequence model.

See [11_heterogeneous_stack_and_fresh_selection.md](11_heterogeneous_stack_and_fresh_selection.md).
