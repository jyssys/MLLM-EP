# Native prefill graph audit

Status: diagnostic work in progress. This document does not establish a serving improvement.

## Source inspection

The qualified runtime contains `RecurrentGraphs` in `vllm_rlt/worker/cuda_graph.py`.
Its device-cache proxy supports packed prefill attention.
The graph implementation retains KV validity checks, metadata copies, and stream events.

The ordinary recurrent runner selects graphs only when `batch.emit_gate` is true.
The grouped prelude path supplies recurrent metadata without a graph owner.
Thus native graph support does not imply that current prefill batches use it.

These findings are source inspection, not measured evidence of a CPU bottleneck.

## Falsifiable control

The diagnostic hook reuses native recurrent graphs for packed P-side calls.
It leaves D execution unchanged.
The native device-cache proxy bypasses the hook during capture.
Unsupported rows use the original eager operation.
The hook retains the qualified projection arithmetic and fixed recurrent depth.

The first comparison uses eight requests per model.
Both arms use the same reduced P pool and two P slots.
This memory setting reserves space for capture.
It is not compared directly with the previous larger-pool serving baseline.

The gate checks output parity, transfer checks, and full recurrent work.
The candidate must also record nonzero native graph replays.
No serving expansion is permitted after an unexplained correctness failure.

Configurations and results use the `prefill_graph_*_gate8` namespace.
The state ledger is `looped_pd_autonomous_prefill_compute_decode_headroom/state/prefill_graph_control_results.json`.

## Limits

Lazy graph capture synchronizes the device and adds startup work.
Any timing comparison must separate capture from steady execution.
Native capture performs warmup calls before replay.
Those calls are instrumentation/setup work, not removed model work.

This is a generic runtime optimization candidate.
It is not a new loop-specific algorithm.
Output parity with the qualified runtime does not establish original-HF equivalence.
