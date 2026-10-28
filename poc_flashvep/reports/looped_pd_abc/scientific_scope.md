# Scientific scope and competing explanations

This document defines review requirements. It does not approve a result.
The active campaign still measures repeated serving configurations.
Each track requires an independent decision for each model.

## Implementation versus research contribution

| Track | Implemented mechanism | Strong competing explanation | Evidence still required for a loop-specific claim |
|---|---|---|---|
| A | Exact prefill continuation after a completed recurrent loop | Ordinary migration at a contiguous logical-depth boundary | A recurrence-dependent deployment or queue benefit beyond tuned conventional PD and colocated replicas |
| B | Immutable KV depth-plane transfer before full-prefill completion | Generic immutable layerwise streaming and descriptor coalescing | Lower uncovered transfer time and better serving performance than existing overlapped chunk transfer |
| C | Initial placement with GPU-completed loop frontiers | Ordinary service-progress-aware queue estimation | Incremental benefit over calibrated service-time placement under identical telemetry and GPU budget |

The source dependency graph establishes which executions are legal.
It does not establish novelty or achieved speedup.
All decode tokens retain full fixed-depth execution on their selected decode worker.
No track implements decode Loop PP, tensor parallelism, or expert parallelism.

## Generic optimizations remain in the controls

A and B use contiguous descriptor coalescing when both source and destination ranges are adjacent.
The coalescer does not copy gaps or invalid planes.
This generic optimization also belongs in their matched controls.
The extra Ouro `PD_A_coalesced` control supplies it explicitly.
The original uncoalesced A control remains separate evidence.

Huginn and Dory omit unused boundary-layer depth planes in both candidate and control transfers.
This preserves cache validity and avoids transferring uninitialized state.
It is not a new loop-specific optimization by itself.
Dory's packed GEMMs and native rotary operator also apply to every adapter-v2 arm.

## Correctness has three separate scopes

1. Transfer identity checks compare the transferred valid KV and registered hidden buffer.
2. Internal output comparisons check each candidate against its matched RLT execution.
3. Official-reference comparisons test a separate implementation and generation protocol.

Passing the first two checks does not resolve a failed third check.
Sequence agreement is not task accuracy.
Bounded generation can terminate at its cap before it produces a complete answer.
Full GSM8K test and HumanEval accuracy remain `NOT_MEASURED`.

The initial official-reference agreements are Ouro 88/100, Huginn 89/100, and Dory 80/100.
The queued Ouro gate audit tests one specific protocol hypothesis.
It cannot explain Huginn or Dory differences automatically.
Any explicit-R4 reference must retain a separate name and configuration.

## What a negative result can establish

A measured negative result can reject the tested implementation and workload combination.
It cannot reject every possible recurrence-aware distributed runtime.
A single-start scaling result cannot establish a confidence interval across server starts.
A low-pressure cohort cannot establish sustainable open-loop capacity.
A tail-only transfer counterfactual cannot bound all queue and contention effects.

The final review must identify missing ratios, unimplemented hybrid roles, and incomplete reference fidelity.
It must not replace these limits with an assumed best-case implementation.

## Evidence classes

| Class | Permitted interpretation |
|---|---|
| `MEASURED` | Recorded GPU or host execution under the stated configuration |
| `STRUCTURAL_ACCOUNTING` | Derived layer, depth, storage, or ownership counts |
| `ORACLE_FROZEN_TIMELINE_NOT_MEASURED_SPEEDUP` | A declared interval-removal counterfactual without queue replay |
| `MODEL_PREDICTION` | A calibrated or assumed cost, not an observed completion time |
| `NOT_MEASURED` | No qualifying measurement exists |

The primary latency denominator is the same-GPU-budget matched baseline.
The strongest measured colocated, Mixed, and static-ratio controls remain visible.
Only matching output workloads can support a lossless timing comparison.
