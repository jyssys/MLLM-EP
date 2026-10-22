# Physical timelines and observer validity

## Scope

All 18 planned heavy captures completed and exported to SQLite.
There are four Static short-allocation traces, four Dynamic short-allocation traces, two native controls, and eight serving-layout traces.
Each trace has a separate unprofiled or clean-replay comparison.
All use permitted physical GPUs only.

The serving-layout stratum generates at most 32 tokens but retains O4096 KV allocation.
All eight controls match their primary task's bounded output and depth prefixes on four requests.
That validation does not turn a bounded trace into whole-generation quality or E2E profiling.

The parser maps CUPTI CUDA ordinals through the launch mask and verified owner UUIDs.
It does not interpret the system GPU inventory index as the logical rank.
See the mapping evidence in each timeline summary.

## Actual cross-request overlap

The following intervals contain real GPU kernels from different requests on different ranks.
They are not inferred from Python queue overlap.
All four PP traces reach four simultaneously tagged kernel ranks.

| Trace | Allocation cap | NVTX window ms | Different-request kernel overlap ms |
|---|---|---|---|
| Static Layer PP | 32 | 1838.425 | 656.780 |
| Static Loop PP | 32 | 2436.468 | 568.930 |
| Dynamic Layer PP | 4096 | 2560.751 | 880.567 |
| Dynamic Loop PP | 4096 | 3596.253 | 848.706 |

These are instrumented activity windows, not clean latency benchmarks.
Do not compare different allocation strata as a speedup.
The clean replay and actual task clocks remain separate.

- [Static Layer PP timeline](timelines/profile_L4_v1/actual_kernel_timeline.png).
- [Static Loop PP timeline](timelines/profile_O4_v1/actual_kernel_timeline.png).
- [Dynamic Layer PP, serving allocation](timelines/profile_serving_shape_LD_v1/actual_kernel_timeline.png).
- [Dynamic Loop PP, serving allocation](timelines/profile_serving_shape_OD_v1/actual_kernel_timeline.png).
- [Native TP4 timeline](timelines/profile_native_TP4_v1/actual_kernel_timeline.png).

## GPU activity, host queues, and synchronization

gpu_activity.csv contains per-rank kernel count, kernel-service sum, and the union of recorded kernel/copy activity.
It also records time without those recorded activities inside the trace window.
That final quantity is not certified hardware idle time or an independently removable E2E cost.
Stream waiting, host gaps, unclassified work, and boundary effects remain possible.

stage_timing_summary.csv separately reports observed host ready-to-dispatch intervals.
The calculation excludes the initial ready_s=0 sentinel.
Those intervals include queue and process-control delay, not only GPU waiting.
CUDA compute/export intervals can include launch gaps and completion waits.
They are not additive critical-path components.

TP/native ranges lack the PP request-stage tags.
A zero tagged-request overlap count there does not imply absent four-GPU execution.
All-rank kernels and owner-local tensors establish physical participation.

## Direct Dynamic KV protocol observations

The short-allocation Dynamic traces contain 107 early-exit repair events in each PP arm.
The observed OD R3 acknowledgement median is 3.658 ms across 106 R3 events.
The corresponding export median is 1.557 ms.
The LD acknowledgement median is 0.908 ms across its 107 repair events.

These are directly timed NVTX host intervals.
They include packing, queue, completion, and scheduling effects where the code performs them.
They are not pure data-movement measurements.
Nested REPAIR, EXPORT, REMOTE, and ACK_WAIT intervals overlap.
No additive E2E percentage or independently removable gain is computed from their sums.
Events and layers are not independent request samples.

See timelines/profile_live_OD_v1/KV_protocol_intervals.csv and the corresponding LD file.
The serving-layout stratum preserves the same protocol at the task allocation shape.
No new wait was inserted to create these NVTX ranges.

## Real TP collectives

The custom Static TP4 trace contains 24,704 NCCL kernels on every rank.
Recorded kernel-service sums differ substantially between ranks.
The same kernel count therefore does not imply equal pure movement latency.
NCCL can spend time waiting for other ranks or host-driven readiness.

The native TP4 trace contains 6,369 custom reduction kernels and 33 NCCL kernels per rank.
The custom reduction names are cross_device_reduce_1stage and cross_device_reduce_2stage.
These labels come from backend kernel classification, not a new logical call counter.
No dense-TP All-to-All cost is fabricated.

## Observer validity

All 18 bounded comparisons match generated output IDs.
All 16 custom comparisons also match exit-depth and computed-argmax sequences.
Native controls do not expose per-token exit arrays in this Static lane.
Those fields remain unavailable.

Heavy host-service overhead ranges from 24.9% to 405.0%.
The custom heavy origin precedes cudaProfilerStart.
Its wall duration can include profiler-start overhead outside the later NVTX GPU window.
Therefore, heavy timing is **UNVALIDATED FOR HEADLINE LATENCY**.
Semantic parity does not qualify its timing.

Headline hardware ratios use five-repeat clean replay instead.
Actual long-task times retain their task-observer label.
No whole-request non-overlapped component breakdown is claimed.

## Raw data and reproducibility

- [All observer comparisons](observer_validity.csv).
- [All profile windows](profile_summary.csv).
- [Per-rank GPU activity](gpu_activity.csv).
- [Host queue and CUDA service intervals](stage_timing_summary.csv).
- [Lossless SQLite archive manifest](profile_archive_manifest.json).
- [Commands and environment](reproduce.md).

The archive process checks SQLite integrity and verifies the decompressed SHA256.
Original Nsight containers and tensor snapshots remain unchanged on the server.
The native TP1 failed attempt remains in the failure ledger.
Its successful retry uses a separate label, profile_native_TP1_v2.
