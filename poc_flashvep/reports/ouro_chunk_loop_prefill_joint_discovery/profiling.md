# Timing, graph, and memory evidence

## Evidence classes

| Evidence | Measurement boundary | Valid use |
| --- | --- | --- |
| Clean serving | In-process scheduled arrival to observed token/completion | TTFT, ITL, TPOT, E2E, drained-cohort rates |
| Light scheduler trace | Host submit begin/end and request descriptors | Issued C/G, rows, phases, order, prompt work |
| CUDA-event trace | Same-device stream events around submission or model call | Elapsed intervals, including possible waits and host launch gaps |
| Nsight capture | Actual CPU APIs, NVTX, GPU kernels, and copies in one diagnostic process | Physical timeline and launch evidence |
| Model/oracle | None used for headline results | No modeled speedup is presented |

Clean runs do not add CUDA events per quantum.
They retain light host-side request descriptors.
Heavy runs use separate labels and do not supply headline E2E latency.
No sum of service intervals is presented as an E2E critical-path percentage.

## Request clock and admission

The client uses `time.perf_counter()` around the in-process API.
Each token event records an actual new token. The harness rejects a multi-token update rather than inventing timestamps.
Background E2E includes initial prefill and graph activity.
The long request's arrival is scheduled 20 ms after all background requests emit a first token.
Admission occurs at a legal engine boundary.

The raw `live_decode_at_long_admission` field actually counts live requests at API submission.
`interference_validity.csv` additionally checks live background requests at the first long-request GPU submission.
There were zero empty-background interference cases.
The concurrency1 control is not an interference case.

`client_cohort_makespan_s` in aggregate tables uses the earliest actual request arrival through the final engine drain.
The older raw `measured_cohort_s` is a shorter post-priming span.
For concurrency1, the aggregate clock also excludes the artificial waiting period before the only arrival.

## CUDA-event quantum measurements

`prefill_quantum_measurements.csv` records 24 observed long-prefill intervals.
The clean and heavy clock lanes remain separate.
The following diagnostic table uses one warmed 2K/concurrency4 cohort per policy.

| Policy, C1024 | Long-prefill compute quanta | CUDA elapsed p50 ms | CUDA elapsed p95 ms | Host submission p50 ms |
| --- | ---: | ---: | ---: | ---: |
| Atomic G4 | 2 | 113.885 | 121.276 | 113.927 |
| Grouped G2 | 4 | 54.075 | 54.797 | 54.100 |
| Mixed G1 | 12 | 28.076 | 50.952 | 28.109 |
| Staged G4 control | 2 | 105.772 | 105.946 | 105.798 |

Mixed can execute more submissions because the token budget also includes active decode rows.
Its actual selected C can be smaller than configured C1024.
Actual prompt rows and total active rows appear in the measurement CSV.
The exact padded graph row count is **NOT_AVAILABLE**.

Submission events include staging, waits, and execution on the selected stream.
The model-call timeline excludes outer staging, but can still include host launch gaps or graph capture.
Neither interval is a certified measurement of pure compute.
The legacy raw field `gpu_compute_ms` names this elapsed interval. Its name does not establish pure compute time.
The clock anchor uses a same-GPU event and a host synchronization bracket.
Each heavy raw record contains the anchor midpoint and its measured uncertainty.
No cross-GPU timestamp alignment is required for this single-GPU study.

## Actual Nsight timeline

Nsight Systems 2024.6.2 captured the separate `nsys_grouped_v1` cold diagnostic.
The exported record contains:

- 741,806 kernels on logical CUDA device0, physical GPU4.
- 9,867 copy activities and 123,083 CUDA runtime API records.
- 1,033 NVTX ranges matched one-to-one with scheduler submissions.
- Zero NCCL kernels. This experiment does not use model-parallel collectives.

| Heavy diagnostic interval | Time ms |
| --- | ---: |
| First-to-last NVTX host window | 13268.923 |
| Union of GPU kernel intervals inside that window | 2632.920 |
| Union of kernels and recorded copies inside that window | 2647.744 |

These are heavy-run timeline facts, not a removable-cost estimate for clean serving.
They do not assign every uncovered interval to CPU scheduling, GPU idle, or a specific dependency.

The largest summed kernel categories include GEMMs, elementwise kernels, and FA4 attention.
`nsys_kernel_service.csv` labels those values as summed service time.
The complete sanitized kernel/copy/runtime event exports are losslessly archived.
`nsys_quantum_alignment.csv` links host submission ranges to request IDs and C/G descriptors.
The Nsight report and SQLite stay intact on the server. Their sizes and SHA256 hashes appear in `nsys_summary.json`.
The Git artifacts exclude the binary reports and unrelated process metadata.

See [the physical timeline](figures/nsys_physical_timeline.png).
It distinguishes host submission ranges from actual GPU kernels.
An NVTX host range ending does not mean the associated GPU work completed.

## Graph and observer controls

All main arms enable the same native RLT graph setting and use the same selected attention backend.
The graph selector remains unmodified.
Prefill follows native eligibility. Decode-only recurrent shapes use native replay where supported.
Stock Mixed retains native fallback behavior for incompatible prefill/decode shapes.
We did not disable Mixed optimizations to favor G2.

Graph counters in a resident engine are cumulative.
`graph_counter_measurements.csv` records both cumulative values and per-cohort increments.
For the first corrected 2K/concurrency4 cohort, Atomic replayed 513 recurrent graph calls with zero new captures.
G2 replayed 517 with zero new captures.
Mixed C1024 replayed 512 and recorded 12 fallbacks.
These are runner counters, not total GPU kernel launch counts.

The graph-off control completed the four-request cohort in 11.927 s for Atomic and 11.540 s for G2.
Matched graph-on reference values were about 3.621 s and 3.593 s.
This single-cohort control demonstrates that disabling native graphs is not a fair strong baseline.
It does not provide a precise E2E attribution to each graph operation.

Relative to the light-observer reference, heavy-trace full-cohort ratios were 1.047 for Atomic and 1.029 for G2.
Those heavy runs preserved the matched four-request outputs and depth sequences.
The no-quantum-recording control also preserved matched outputs.
Its timing differed substantially for some arms, including a 1.422 Atomic long-TTFT ratio.
One cohort cannot separate observer cost from process/JIT variation.
Therefore, observer overhead is not proven negligible.
All headline intervals use the repeated, common light-observer lane instead.

## Prefill-only and decode-only controls

At 2K/concurrency1, long TTFT was 210.646 ms for Atomic, 221.418 ms for Split, and 227.494 ms for G2.
This single warmed control shows no prefill-only G2 gain.

The decode-only concurrency4 span was 3.183 s for Atomic and 3.268 s for G2.
This is a one-cohort diagnostic, not a statistically supported decode regression estimate.
The concurrency1 Atomic run incurred first-use compilation and took 5.614 s, versus 2.635 s for G2 afterward.
That asymmetric cold result is invalid as evidence of a G2 decode speedup.
The source keeps the decode scheduler and recurrent math unchanged.

## Memory

All primary arms reserve 4,096 global physical KV blocks with a 16-token page size.
This is a 12 GiB KV pool across all depth planes, not 4,096 blocks per depth.
The initial stock calibration used a smaller pool and remains a separately labeled lane.

| Shape | Atomic staged buffer MiB | Split/G2 C1024 staged buffer MiB | Mixed C128 staged buffer MiB | Atomic/G2 mean peak allocated GiB |
| --- | ---: | ---: | ---: | --- |
| 2K / concurrency4 | 0 | 16 | 2 | 14.898 / 15.293 |
| 8K / concurrency8 | 0 | 32 | 4 | 15.368 / 15.653 |

Each raw record includes allocated, reserved, peak allocated, KV-pool, and staged-activation bytes.
Graph-private and scratch bytes are not independently resolved.
Global JIT caches and fixed configuration order can affect total peak measurements.
The staged-buffer size is a direct tensor measurement. The total peak difference is not a pure scheduler allocation.

## Remaining profiling limits

This study did not measure an HTTP frontend, continuous open-loop saturation, multi-GPU scaling, or every GEMM's useful FLOPs.
Exact padded rows, per-operation pure kernel time, graph-private HBM, and a full non-overlapped E2E attribution remain PARTIAL.
These limits do not invalidate the recorded client latency trade-off.
They prevent a stronger claim that a new recurrent scheduler removes a proven architecture bottleneck.
