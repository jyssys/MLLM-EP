# Timing and memory contract

## Evidence classes

| Class | Meaning |
| --- | --- |
| LIVE_MEASURED_TRUE_EP | Actual multi-rank model execution with sharded expert weights and NCCL communication |
| LIVE_GENERIC_CONTROL | Actual current-router source selection under the same replicated-dense execution contract |
| LIVE_NATIVE_ONE_GPU | Actual original model execution on one physical GPU |
| LIVE_CUDA_EVENT_API_SPANS | Event intervals around API sequences, including possible stream waits and host submission gaps |
| LIVE_CUPTI_OBSERVER | A separate GPU kernel trace with profiler overhead |
| HELDOUT_TRACE_STATISTIC | A statistic from actual router outputs on prompts excluded from calibration |
| TRACE_MODEL | A placement or latency calculation, not an implemented E2E result |
| STRUCTURAL_ACCOUNTING | Exact tensor sizes, ownership, message payloads, or dependency facts |
| NOT_AVAILABLE | Evidence that the experiment did not obtain |

## Workload

The primary matrix contains three models, two EP degrees, five batch sizes, and three prompt lengths.
It contains 90 required cells.
The batch sizes are 1, 2, 4, 8, and 16 requests per complete model batch.
The prompt lengths are 128, 512, and 2,048 tokens.
They are global workload sizes, not per-rank independent requests.

Public prompts cover general text, knowledge, math, and code.
The shape workload truncates or repeats text to reach the stated token length.
It is not a natural benchmark-accuracy workload.
Each model uses its own tokenizer.
Identical token counts do not imply identical token IDs across models.

Each timed generation emits 32 actual greedy tokens.
The measurement disables EOS to keep the hardware work comparable.
It does not substitute frozen output tokens.
The natural benchmark evaluations use separate official task protocols.

## Execution boundary

Timing starts after model loading, native reference collection, expert sharding, and warm-up.
It excludes those preparation stages.
The native and EP paths use eager execution, not a new serving scheduler or CUDA Graph runtime.
No token chunking or loop partition pipeline enters the experiment.

Each forward cell uses ten warm-up forwards and five clean timed forwards.
An EP barrier and device synchronization precede each timed forward.
Device synchronization completes the timed interval.
A maximum reduction then collects the largest rank duration outside that interval.
The clean forward number includes all native model operations and the exact EP bridge.

Generation uses one full warm-up rollout before three clean rollouts.
The warm-up visits the growing context shapes.
The output endpoint is rank 0.
The cohort makespan is the maximum recorded rank duration for each rollout.
Throughput uses that cohort makespan and the actual emitted token count.

These are synchronized hardware timings.
They exclude client networking and request-queue delay.
They are not open-loop serving-capacity measurements.

## Decode semantics

Both native adapters recompute the full prefix at each generated step.
Their calls retain no KV cache between steps.
TPOT therefore includes full-prefix recomputation.
It is not cached single-token decode latency.
The first step supplies TTFT.
The remaining 31 intervals supply TPOT.

The separate decode observer traces the final measured forward.
Its input has the original prompt plus 31 generated tokens.
Its component spans do not represent the sum of all 31 decode steps.
The observer cannot replace the measured rollout E2E duration.

Native functional observers also record new-token routing over successive generated steps.
Their calibration and held-out prompts remain separate.
Those observations never enter clean latency intervals.

## Observer isolation

CUDA-event attribution and router extraction run after clean forward timing.
The observer records its own wall duration.
The summaries report its duration divided by the clean forward mean.
Diagnostic expert captures add copies at one recurrent site.
The API spans include that observer cost.

The event decomposition selects one cumulative-maximum rank.
It does not sum rank times into an E2E fraction.
It does not sum component-wise maxima from different ranks into a claimed critical path.
Expert-event dispersion is not an exact rank-wait measurement.

The CUPTI run is a separate B1, prefix-128 observer.
Kernel intervals use their union when concurrency exists.
NCCL service time can include synchronization waits.
A no-kernel interval is not necessarily hardware idle, because copies can remain active.
The CUPTI window is not the clean serving interval.

## Payload and memory

Dispatch payload counts include BF16 activation rows sent to remote ranks.
Combine payloads return expert rows with the same vector width.
These values exclude transport headers and internal NCCL protocol traffic.
The restoration field records the All-Reduce tensor's logical size per rank.
It is not measured network wire traffic.

Weight storage uses the actual owner-local tensor sizes.
Dense weights and positional buffers remain replicated.
Clean forward peak allocation comes from the CUDA allocator counter.
It is not a full-generation peak or an NVML measurement of total device HBM.
The reserved-memory snapshot follows the separate observer forwards.
It can retain allocator blocks created by those observers.

Persistent KV cache capacity is N/A on this cache-free path.
Attention still creates and accesses live K/V tensors during each forward.
The report does not call those accesses free.

## Repetition and uncertainty

The full 90-cell matrix uses serial warmed repetitions from one process start per cell.
Its bootstrap intervals describe those repetitions only.
They do not measure restart or GPU-group uncertainty.
Additional B1 checks use two further process starts.
One EP4 repeat changes from GPUs 0–3 to GPUs 4–7.
The report keeps that check separate from the complete matrix.

One experiment group runs at a time during headline timing.
Owned burners remain on unused authorized GPUs, as requested.
The quiet lane therefore excludes other experiment groups, not all board power effects.
CPU analysis uses separate affinity ranges when it overlaps GPU work.
The environment ledger records these conditions.

## Native allocator recovery

The original IFM Ouro native attempt completed fourteen cells before allocator fragmentation stopped generation.
The failed cell used B16 and prefix 2,048.
A real-checkpoint regression passes with expandable allocator segments.
The resume preserves byte-identical evidence for the fourteen completed cells.
Only the largest IFM native cells require the recovery allocator.
The native-versus-EP comparison marks those rows allocator-mismatched.
The EP policy comparisons retain the default allocator across all policies.
See `native_oom_diagnosis.md` and each native attempt's inheritance manifest.
