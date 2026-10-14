# Timing contract

## Fixed baseline

Use only physical GPUs 4, 5, 6, and 7. Logical ranks 0 through 3 map in that order.
The baseline is native TP4/EP1/DP1/PP1/CP1 on the pinned non-preview LLaDA2.0-mini.
Native precision, mask, sampler, threshold 0.9, block32, output cap64, and EOS remain unchanged.
The model already caches prefix KV. Existing padding semantics remain unchanged.
No EP dispatch/combine All-to-All exists in this configuration.

The workload reuses eight frozen calibration IDs and 24 untouched held-out IDs.
This is continuity data, not a new held-out validation claim.
Prefixes are 512, 4,096, 8,192, and 16,384 where physical memory permits execution.
Graph policy and any eager-prefill exception must appear in every result row.

## Three separate clocks and accounts

1. Clean CPU-monotonic wall starts immediately before native generation.
   Completion includes one final GPU synchronization. Loading, graph capture,
   warmup, input tokenization, output serialization, and server queueing are excluded.
   Clean runs have no Atlas observer hooks. Returned output and NFE are checked afterward.
2. Coarse CUDA events record every actual occurrence outside captured graph internals.
   Events bracket prefill, blocks, iterations, model calls, input staging, replay,
   sampling, cache-manager updates, and finalization. The event queue is preallocated.
   Event resolution and acceptance transfers occur only after request completion.
   A GPU interval can include host launch gaps. It is not kernel compute time.
3. Nsight Systems records actual CPU APIs, NVTX scopes, graph nodes, kernels,
   and physical memory transfers. This is a separate diagnostic lane.
   Its wall time is not the headline serving latency.

No graph operator's last replay is multiplied by an iteration count.
No rank-local service sum is presented as an EP or E2E stage fraction.
No component maxima from unrelated ranks are added.

## Parity and observer gate

Three clean repeats and three coarse repeats use the same request and seeds.
A separate parity-only run retains native token-update outputs without changing them.
Final token IDs and NFE must match across ranks and lanes.
Coarse and parity-only token updates must match exactly.
EOS/stop interpretation follows native token updates and the bounded output cap.
These snippets do not establish benchmark accuracy.

Prefer observer overhead at most 5%. Reject quantitative fine attribution above 10%.
Bootstrap uses requests, not repeats, iterations, or layers. Use 2,000 resamples.
Nsight node-trace overhead must remain visible even when output parity passes.

## Timeline and causality

Host timestamps use the same system monotonic clock across worker processes.
Each rank's CUDA origin receives a conservative host-completion bracket.
The bracket width is recorded as timestamp uncertainty.
Do not subtract unaligned CUDA timestamps across ranks.
Nsight timestamps can provide a separate common capture clock.

Intervals form a parent-child tree. Exclusive wall accounting subtracts children
from parents. Kernel service accounting preserves overlap.
For physical critical-path attribution, classify only intervals with supported
stream and dependency evidence. Preserve UNKNOWN or PARTIAL intervals.
GPU idle does not prove avoidable host overhead. Collective duration does not
separate payload transfer from rank waiting without additional evidence.

## Promotion

An architecture model must include memory, actual communication, serial refinement
dependencies, and pipeline bubbles. DP throughput is not B1 latency.
No optional intervention starts before new Stage A and architecture screening.
Require a credible removable E2E target of 15%, or realistic 2× target gain of 10%.
Unknown attribution can support only CONDITIONAL_MEASUREMENT, not GO.
Select at most one bounded intervention. Do not create a new serving engine.
