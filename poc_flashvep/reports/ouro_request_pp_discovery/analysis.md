# Analysis and final verdict

## Decision

**CONDITIONAL.** Both exact request-level pipelines work on four physical GPUs.
The experiment shows a repeatable implementation-level advantage for Loop-PP in most fixed-R4 cases.
It does not establish an architecture advantage over an optimized multiprocess engine, TP4, or DP4.
The evidence does not justify a new adaptive scheduler or a general serving-performance claim.

Gate A, executable correctness: PASS on the tested fixed-depth inputs.
Gate B, actual request-level pipeline overlap: PASS in both placements.
Gate C, paired discovery measurements: PASS, with the timing and backend limitations below.
Gate D, true dynamic depth: NOT_RUN.

## Primary result

There are 162 physical cohort measurements: 27 paired cases, two arms, and three repeats.
Loop-PP has the lower median makespan in 25 cases.
The median speedup range is 0.971× to 2.044× across all cases.
The two median Layer-PP wins occur at R2, C8/C16, P128, O32.
Both intervals include 1.000×. They do not establish a reliable crossover.
The complete values and intervals appear in [benchmark_summary.md](benchmark_summary.md).

Representative fixed-depth results:

| R | C | P | O | Layer-PP makespan s | Loop-PP makespan s | Loop/Layer speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 1 | 128 | 32 | 0.8716 | 0.6448 | 1.352× |
| 4 | 4 | 128 | 32 | 1.8765 | 1.0621 | 1.767× |
| 4 | 16 | 128 | 32 | 5.5157 | 3.1088 | 1.774× |
| 4 | 16 | 512 | 128 | 20.7039 | 12.3288 | 1.679× |
| 2 | 16 | 128 | 32 | 2.7022 | 2.7841 | 0.971× |
| 3 | 16 | 128 | 32 | 3.6103 | 2.7609 | 1.308× |

The input and output lengths describe hardware shapes, not public-benchmark quality settings.
TTFT includes whole-prompt prefill and queueing.
The supplied CSV also records mean, P50, and P90 request metrics.
No stage combines several requests into a larger GEMM batch.
Longer Loop-PP stages therefore do not increase the tensor batch size of each decoder GEMM.

## Actual overlap and physical transport

These values come from the separate heavy profiler lane.
The headline comparison does not use these wall times.
Kernel overlap requires different request IDs on different GPUs at the same profiler time.
Stage brackets alone do not provide this proof.

| Profile | Heavy window ms | Different-request kernel overlap ms | Max active kernel ranks | Actual P2P copies | P2P MiB | Sum P2P activity ms |
| --- | --- | --- | --- | --- | --- | --- |
| layer_pp4_R2_C8_v5 | 837.688 | 130.832 | 4 | 504 | 29.532 | 2.205 |
| layer_pp4_R4_C4_v5 | 906.187 | 106.274 | 4 | 508 | 31.641 | 1.608 |
| loop_pp4_R2_C8_v5 | 579.038 | 95.330 | 2 | 120 | 4.219 | 0.448 |
| loop_pp4_R4_C4_v5 | 509.185 | 136.611 | 4 | 124 | 6.328 | 0.613 |

Per-GPU activity in the same heavy lane:

The activity union includes recorded kernels and copies. Its complement includes pipeline bubbles, host gaps, and waiting.
It does not isolate architectural bubbles from observer or host-scheduler costs.

| Profile | GPU | Kernel service ms | Kernel/copy activity union ms | No recorded activity % |
| --- | --- | --- | --- | --- |
| layer_pp4_R2_C8_v5.sqlite | 4 | 128.247 | 129.145 | 84.58 |
| layer_pp4_R2_C8_v5.sqlite | 5 | 128.247 | 129.059 | 84.59 |
| layer_pp4_R2_C8_v5.sqlite | 6 | 127.530 | 128.537 | 84.66 |
| layer_pp4_R2_C8_v5.sqlite | 7 | 137.893 | 139.108 | 83.39 |
| layer_pp4_R4_C4_v5.sqlite | 4 | 128.115 | 129.072 | 85.76 |
| layer_pp4_R4_C4_v5.sqlite | 5 | 128.272 | 129.169 | 85.75 |
| layer_pp4_R4_C4_v5.sqlite | 6 | 127.582 | 128.439 | 85.83 |
| layer_pp4_R4_C4_v5.sqlite | 7 | 135.445 | 136.564 | 84.93 |
| loop_pp4_R2_C8_v5.sqlite | 4 | 252.771 | 253.534 | 56.21 |
| loop_pp4_R2_C8_v5.sqlite | 5 | 257.673 | 258.400 | 55.37 |
| loop_pp4_R2_C8_v5.sqlite | 6 | 0.000 | 0.000 | 100.00 |
| loop_pp4_R2_C8_v5.sqlite | 7 | 0.000 | 0.000 | 100.00 |
| loop_pp4_R4_C4_v5.sqlite | 4 | 126.750 | 127.261 | 75.01 |
| loop_pp4_R4_C4_v5.sqlite | 5 | 126.461 | 126.881 | 75.08 |
| loop_pp4_R4_C4_v5.sqlite | 6 | 125.677 | 126.113 | 75.23 |
| loop_pp4_R4_C4_v5.sqlite | 7 | 131.169 | 131.560 | 74.16 |

The R4/C4/O8 Layer-PP trace contains 480 hidden copies and 28 token returns.
The corresponding Loop-PP trace contains 96 hidden copies and 28 token returns.
Nsight reports these operations as Peer-to-Peer copies.
Its peer-copy bytes exactly match the executed runtime payload counter.
H2D traffic contains only four 128-token inputs. D2H traffic contains 32 scalar token IDs.
There is no hidden-state or KV staging through host RAM.

R4 hidden payloads fall by 80% because the handoff count changes from 15 to 3 per forward.
Total hidden bytes include a full-prompt tensor on prefill and one hidden row on later forwards.
For uniform shapes, they equal `C * (P + O - 1) * 2048 * 2 * handoffs`.
Token return and scalar control are additional measured traffic.
Each KV slot stays on its owner GPU for the whole request.

Do not interpret P2P service sums as an E2E critical-path share.
Different links and compute can overlap.
The physical transfer service difference is much smaller than the observed cohort wall-time difference.
This rules out a payload-bandwidth-only explanation in the traced condition.

## Stage granularity and runtime cost

The principal measured distinction is the number of stage boundaries.
R4 Layer-PP executes 16 stage items per forward. Loop-PP executes four.
Each item includes a queue operation, input publication, graph replay, local KV staging, and completion wait.
Layer-PP also re-enters GPU4 three times per forward.
The heavy trace records 512 versus 128 graph launches for C4/O8.
It records 520 versus 136 event-synchronization calls, including workload-clock checks.
The per-rank kernel service times are broadly similar. The gaps between active GPU intervals are substantial.

The native model computes position embeddings once for a forward pass.
The initial common pipeline path recomputes those embeddings at each stage.
That adds more redundant RoPE preparation to Layer-PP.
The backend controls below remove that cost with immutable position constants.
They preserve every input-dependent decoder operation and exact RoPE values.

Python threads, GIL contention, event publication, allocator behavior, and local graph staging can affect the gap.
The experiment does not isolate each of those costs as a non-overlapped E2E component.
Nsight node tracing substantially changes timing.
Its activity fractions describe that heavy execution only.
They are not clean-run GPU utilization measurements.

## Backend controls

The first control preallocates destination activation buffers before admission.
It changes no placement, model math, queue discipline, or payload byte count.
The second control also caches position-only RoPE constants outside graph replay.
Each control passes a separate single-GPU reference comparison before performance measurement.
The controls use three physical repeats per row.
Control execution order differs from the main alternating matrix. Treat them as sensitivity checks.

| Control | R | C | P | O | Layer-PP s | Loop-PP s | Loop/Layer speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| receive_pool | 2 | 16 | 128 | 32 | 2.5889 | 2.6803 | 0.966× |
| receive_pool | 4 | 1 | 128 | 32 | 0.7935 | 0.6317 | 1.256× |
| receive_pool | 4 | 4 | 128 | 32 | 1.7179 | 0.9938 | 1.729× |
| receive_pool | 4 | 16 | 128 | 32 | 4.9973 | 3.1503 | 1.586× |
| receive_pool_rope | 4 | 4 | 128 | 32 | 1.7664 | 0.9429 | 1.873× |
| receive_pool_rope | 4 | 16 | 128 | 32 | 4.5272 | 2.7545 | 1.644× |

These controls do not remove the single-process host scheduler or publication waits.
They cannot establish performance for an optimized four-process transport implementation.
Do not attribute the full remaining gap to link bandwidth or to a fundamental loop-specific bottleneck.

## Low depth and pipeline bubbles

At R2, Loop-PP schedules no decoder work on GPUs 6 and 7.
At R3, it schedules no decoder work on GPU7.
Layer-PP uses all four GPUs for every executed loop at either depth.
The depth traces record actual omitted work, not a post-hoc exit label.
Idle late GPUs alone do not prove a throughput crossover.
In this prototype, shorter routes offset some loss of distributed compute capacity.
The R2 primary medians approach parity. The R4 gap is larger.

The finite cohort includes initial fill and final drain.
The same FIFO policy does not imply the same order of eligible work across different stage granularities.
The trace retains every ready-queue wait with request, layer range, loop, and GPU identifiers.
No per-request global synchronization prevents a different request from progressing.

The following short trace records sum stage brackets on each GPU.
Queue time sums waits across distinct stage items. It can exceed cohort wall time.
Compute brackets include launch gaps. They are not pure kernel service.

| Arm, R4 C4 P128 O8 | GPU | Stage calls | Prefill compute brackets ms | Decode compute brackets ms | Sum ready-queue waits ms | P90 stage queue wait ms |
| --- | --- | --- | --- | --- | --- | --- |
| layer_pp4 | 4 | 128 | 53.29 | 185.81 | 271.35 | 5.94 |
| layer_pp4 | 5 | 128 | 25.79 | 126.80 | 125.15 | 2.52 |
| layer_pp4 | 6 | 128 | 48.18 | 182.61 | 196.85 | 6.06 |
| layer_pp4 | 7 | 128 | 66.15 | 148.88 | 237.50 | 4.74 |
| loop_pp4 | 4 | 32 | 21.48 | 114.00 | 199.73 | 22.02 |
| loop_pp4 | 5 | 32 | 24.29 | 114.46 | 64.16 | 7.94 |
| loop_pp4 | 6 | 32 | 21.25 | 114.09 | 29.47 | 3.96 |
| loop_pp4 | 7 | 32 | 20.96 | 115.06 | 63.81 | 6.66 |

Prefill path spans include internal stage waits but exclude the initial admission wait.
These trace-lane values are separate from the clean TTFT table.

| Arm | Median prefill path wall ms | Median initial entry wait ms | Median TTFT ms |
| --- | --- | --- | --- |
| layer_pp4 | 114.20 | 2.62 | 116.87 |
| loop_pp4 | 31.14 | 8.14 | 39.39 |


## Memory trade-off

Layer-PP uses 0.9493 GiB of weights per GPU, including auxiliary replicas.
Loop-PP uses 2.6722 GiB per GPU.
The extra weight cost is approximately 1.7229 GiB per GPU.
The same full LM head and embedding are present in both arms.
For R4, owner-local KV storage is equal across placements at the same request shape.
There is no intrinsic logical-KV compression.
At R2/R3, Loop-PP concentrates all live KV on its active depth ranks.

The prototype retains separate graph storage for exact attention lengths.
That storage is much larger than necessary in a production engine.
Peak PyTorch allocation is not a model-weight-plus-KV capacity estimate.
The memory plot explicitly separates weights, owner-local KV, and other allocations.
No maximum serving-concurrency claim follows from this matrix.

## Observer validity and correctness scope

| Arm | Clean median s | Event-trace median s | Trace/clean ratio |
| --- | --- | --- | --- |
| layer_pp4 | 0.4487 | 0.4843 | 1.079 |
| loop_pp4 | 0.2286 | 0.2749 | 1.202 |

Three short control repeats give a noisy overhead estimate, not a universal correction factor.
The analysis applies no correction factor to clean latency.
The pipeline's safe-publication event also exists in clean mode.
Heavy profiling and per-stage trace data remain outside the headline lane.

The final runtime matches the single-GPU reference bitwise on the tested numerical probes.
Both placements also match every primary generated token ID.
This is not an accuracy evaluation on GSM8K or MATH.
EOS-enabled capped probes remain separate from the forced-length matrix.
See [correctness.md](correctness.md) for actual stop outcomes and preserved failures.

## Dynamic depth and future architecture

True online token-level dynamic exit: NOT_IMPLEMENTED.
Fixed R2/R3 do not establish correctness for a variable-depth request's missing historical KV.
An exit-aware runtime needs separate gate, active-token, and ragged-KV correctness checks.
The earlier adaptive `last_exited` CDB implementation is not silently substituted here.

Full weight replicas make alternative compute ownership possible in principle.
They do not make depth-indexed KV available on an arbitrary GPU.
Moving a later decode stage can require KV transfer, duplication, or a new cache policy.
None of those costs can be omitted from a future proposal.
Neither fixed Loop-PP nor generic PP is a novelty claim in this report.

The smallest useful next experiment is a common multiprocess/event-driven transport control.
Keep the checked placements, exact caches, frozen inputs, and no-chunking rule.
Remove avoidable Python scheduling overhead before designing any new adaptive placement.
Then compare against a same-four-GPU DP or optimized serving baseline.
Proceed to true dynamic depth only after that control preserves the advantage and numerical correctness.

## Artifacts

- [Environment and ownership](environment.md)
- [Correctness and failures](correctness.md)
- [Complete performance tables](benchmark_summary.md)
- [Timing definitions](timing_contract.md)
- [Reproduction commands](reproduce.md)
- [Raw aggregate CSV](results.csv)
- [Per-stage data](stage_metrics.csv)
- [Transport edges](transport_edges.csv)
- `raw/`: losslessly compressed inputs, generated IDs, token times, memory, and counters.
- `profiles/`: Nsight reports, exported activity, four-GPU kernel timelines, and per-rank summaries.
- `commands/` and `logs/`: exact invocations, source hashes, status, and failure logs.
- `source_v5.tar.gz`: source snapshot before optional transport/RoPE controls.
