# Fixed-shape MoE tail root-cause investigation specification

## Objective

Determine whether the fixed-shape online MoE tail is a real execution
phenomenon or a measurement/runtime artifact.  The investigation must end in
`ROOT_CAUSE_FOUND` or `STRONG_NO_GO`; an unresolved HOLD is not a valid final
state.  No production optimization is implemented.

## Environment

- `CUDA_VISIBLE_DEVICES=1,2,3,4` (physical GPUs 1–4 only)
- Qwen3-VL-30B-A3B-Instruct, BF16
- vLLM 0.20 V1, TP2/DP2/EP4/PP1
- DeepEP high-throughput, TritonExperts, linear placement
- DBO off, prefix cache off, eager/stable timing path

## H1–H10 hypotheses

| ID | Hypothesis | Falsifying prediction |
|---|---|---|
| H1 | The large tail is a measurement/join artifact. | Route-id and same-device event instrumentation removes the tail. |
| H2 | A pre-MoE scheduler/backlog or upstream attention delay is first. | Tail is already present at layer-entry/attention and boundary synchronization removes it. |
| H3 | DeepEP prepare/dispatch stream/event dependency is first. | Dispatch span or event wait diverges first; serialization/stream isolation removes it. |
| H4 | Expert execution/workspace/kernel state is first. | Expert CUDA span diverges first and allocator/workspace preconditioning reduces it. |
| H5 | Combine/finalize synchronization is first. | Combine completion is the first divergence and controlled combine serialization changes tail rate. |
| H6 | A previous MoE invocation leaves a rank-local backlog. | Lagged prior stage duration/idle gap predicts tail; drain/reset intervention removes it. |
| H7 | Tail is a global GPU/runtime interference event. | Non-MoE attention and unrelated CUDA activity co-tail; isolated replay does not. |
| H8 | DP/EP cross-rank synchronization amplifies a local spike. | Multiple ranks diverge together at one collective/event and critical span tracks it. |
| H9 | The anomaly is intrinsic to the fixed routing shape. | Exact route replay remains slow across a persistent worker and independent repetitions. |
| H10 | A specific layer/M regime has a kernel or allocator threshold. | Early/middle/late fixed-M groups show a repeatable discontinuity at the same stage. |

## Instrumentation hierarchy

1. Record `scheduler_iteration_id`, normalized `route_id`, `layer_id`, EP rank,
   DP rank, local invocation ID, phase, M, active requests and prior-step IDs.
2. Use same-device CUDA events for: layer entry, attention start/end,
   router/MoE-ready, prepare/layout, dispatch start/end, expert start/end,
   combine start/end, post-MoE and layer completion.
3. Never subtract CUDA timestamps across GPUs.  Cross-rank analysis uses
   duration and event-order categories only.
4. Preserve exact top-k route, M_e histogram, rank loads and request context.
5. Attach lagged state (previous 1–4 invocations): M, phase, iteration duration,
   each stage duration, idle gap, active requests and DP/rank imbalance.

## Anchor workloads and sampling

- Primary prefill: exact M≈284, early/middle/late layers, ≥200 observations
  (target ≥500) per representative layer.
- Secondary decode: M=1, ≥500 observations.
- A tail is the within `(phase,layer,M,route-shape)` p95 group; extreme tail is
  p99 or >3× the group median.  Report p50/p90/p95/p99/max and CV.
- Use a persistent worker, global warmup, per-shape warmup and interleaved
  normal/tail case ordering.

## Fixed-route replay

Select representative normal and tail snapshots.  In a persistent worker replay
each exact route at least 100 times, preserving M, top-k, M_e histogram and
rank loads.  If a tail snapshot is fast under replay, classify the excess as
online state; if it remains slow, classify it as intrinsic execution shape.

## Causal interventions

At least one intervention is required after localization:

- controlled pre-MoE synchronization or backlog drain;
- diagnostic dispatch/expert/combine serialization;
- allocator/workspace preconditioning;
- isolated DeepEP route replay;
- expert-only replay.

Report tail frequency and p99/p50, not only a single latency.  A root cause
requires ≥30% tail reduction (preferably ≥50%) under an intervention.

## Non-MoE and Nsight escalation

Record attention/non-MoE timings in the same scheduler iteration.  If corrected
events still leave an unexplained tail, capture normal and tail cases with
Nsight Systems/NVTX, annotating run, iteration, route, layer, rank and stage.
Locate the first kernel/stream/event divergence; if `nsys` is unavailable,
record that fact and use CUDA-event/kernel-name evidence.

## Root-cause matrix and decisions

Maintain `ROOT_CAUSE_MATRIX.md` with SUPPORT++, WEAKEN or REJECT after each
experiment.  `ROOT_CAUSE_FOUND` requires attribution, repeated reproduction,
and an intervention reducing tail magnitude/frequency by ≥30%.

`STRONG_NO_GO` is allowed only if corrected instrumentation removes the old
tail, the excess is entirely pre-MoE/global and non-MoE, or fixed-route replay
and corrected serving show no MoE/EP-specific signal.  “Could not determine”
alone is not sufficient.

## Artifact and time policy

Target 4–8 hours.  Keep raw traces, stage CSV/JSON, replay manifests and
intervention logs in a timestamped result directory.  Do not implement a
scheduler, routing change, kernel, placement change or production method.

Required outputs: `ROOT_CAUSE_MATRIX.md`, `INSTRUMENTATION_VALIDATION.md`,
`TAIL_CASES.md`, `NSIGHT_FINDINGS.md`, `EXPERIMENT_LOG.md`, report and
`gate_summary.json`.
