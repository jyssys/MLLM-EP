# Cost and headroom checkpoint

Status: active research checkpoint, not the final campaign result.
No candidate has a measured architecture E2E speedup.
The ten-hour duration requirement is not met at this checkpoint.

## Evidence boundaries

| Evidence | Meaning | Does not establish |
|---|---|---|
| `analysis/checkpoint_03/native_summary.csv` | Fresh optimized fixed-R4 controlled-work E2E | Natural task latency or candidate speedup |
| `outputs/components_*_graph_v1/components.json` | Physical P2/P4 operator timing | Legal future-depth readiness or full-model speedup |
| `outputs/finite_correction_*/finite_correction.json` | Actual pretrained current-token state/KV diagnostic | Free-running quality or physical solver latency |
| `artifacts/cost_screens_checkpoint2.json` | Explicitly charged model estimates | Measured E2E or an optimal schedule |
| `outputs/residency_*/residency.json` | Alias/intervening-access controls | Full-stack cache hit rate or actual HBM traffic |

## Optimized fixed-R4 native reference

The checkpoint pins both Base models and BF16 execution.
Native vLLM keeps CUDA Graphs, FlashAttention, and custom All-Reduce enabled.
EP is disabled because these models are dense.
The native path does not use the adaptive exit gate.

The workload contains two frozen calibration request shapes.
Each shape runs five times at each context length.
The model generates 128 controlled teacher tokens.
The reported latency includes prompt and output execution.
The repeated prompt tokens create hardware shapes, not natural long-context reasoning tasks.

| Model | Context tokens | TP1 E2E, ms | TP2 E2E, ms | TP4 E2E, ms | TP1/TP4 |
|---|---:|---:|---:|---:|---:|
| Base1.4B | 1024 | 1345.194 | 1299.269 | 1118.287 | 1.203× |
| Base1.4B | 4096 | 1608.300 | 1379.223 | 1167.487 | 1.378× |
| Base1.4B | 8192 | 1688.025 | 1527.183 | 1287.809 | 1.311× |
| Base2.6B | 1024 | 2445.190 | 2110.269 | 1929.455 | 1.267× |
| Base2.6B | 4096 | 2659.659 | 2261.996 | 2016.080 | 1.319× |
| Base2.6B | 8192 | 2992.262 | 2526.339 | 2193.266 | 1.364× |

These are medians of ten trials per cell, from only two distinct request shapes.
Method order was sequential, not randomized across TP degrees.
The table does not support a broad request-level confidence claim.
A separate generic All-Reduce/RMSNorm fusion control is in progress.
That control may strengthen the same-P baseline. It is not a new execution axis.

## C02: attention owner and remote MLP

The tested layout preserves local attention and historical KV.
It sends a normalized row to a full-weight MLP worker.
The worker returns the completed down-projection output.
The source waits before its next normalization and layer.

| Model | P | Rows | Local MLP, ms | TP MLP, ms | Remote round trip, ms |
|---|---:|---:|---:|---:|---:|
| Base1.4B | 2 | 1 | 0.03707 | 0.02994 | 0.06178 |
| Base1.4B | 4 | 1 | 0.03725 | 0.02760 | 0.06677 |
| Base2.6B | 2 | 1 | 0.03705 | 0.02902 | 0.06481 |
| Base2.6B | 4 | 1 | 0.03713 | 0.02747 | 0.06265 |

Each cell uses graph-captured operators and 31 trials after seven warm-up calls.
The remote path includes both real communication directions and the complete MLP.
The source output matches the tested local operator bitwise.
The timing rule selects the maximum rank-local duration for the complete operator trial.
It does not add independent component maxima.

The tested remote path loses at every row count from 1 through 512 for both checkpoints.
This result rejects that isolated-source layout before an engine rewrite.
It does not reject every pooled attention/FFN architecture or every workload.
The nearest generic ownership controls remain AFD and KernelFlume.

## C05: finite-trajectory correction

The numerical probe evaluates the actual pretrained stack with matching historical depth KV.
Only current-token embedding or one exact pass initializes the guesses.
Future reference states enter diagnostics only.
Provisional current-token KV never becomes a new generated sequence.

The executed full-Jacobi scheme reaches exact boundary states after four zero-warm-up rounds in the tested cases.
One exact warm-up plus three correction rounds also reaches exact states.
The matching current-token KV recovers at those points.
Earlier greedy agreement can coexist with incorrect internal KV.

The optimistic model uses native TPOT divided by four as a coarse core-round approximation.
This is not a measured recurrent-pass timer or a hardware lower bound.
The model charges replicated-core rounds separately from the strongest same-P native TPOT.
It excludes transport, correction arithmetic, validation, contention, allocation, and KV repair.
These exclusions favor the candidate.

The second cost screen also skips the proven exact prefix for free.
For P2, a full exact recovery can then require six ideal rounds without warm-up, or five with one warm-up.
For P4, it still requires four ideal rounds.
This alternative is an analytic refinement, not the executed implementation.
It does not establish useful headroom against the measured TP2/TP4 reference.

Finite-round approximations remain `QUALITY_UNRESOLVED`.
This checkpoint does not rescue them with teacher-forced top-1 agreement.
The prior stale/delta quality failure remains part of the evidence.

## C08: lossless transport screen

The CPU screen reversibly encodes actual BF16 activation differences.
Its byte ratios are encoded file sizes, not physical transfer ratios.
The GPU probe measures XOR/byte-plane reconstruction without an entropy codec.

For a 4096-byte payload, P2 broadcast takes about 0.01035 ms in the tested primitive lane.
The unfused transform round trip takes about 0.00908 ms before any entropy coding or transport.
Protocol startup matters at these sizes.
Smaller encoded size therefore does not establish lower latency.
A fused implementation remains untested, but no recurrence-specific architecture gain exists yet.

## Residency and fairness

A repeated operator can reuse a small TP weight shard in cache.
The complete core has a much larger reuse distance.
The new control inserts explicit 128/512 MiB read-modify-write work between operator calls.
It compares aliased pointers with four physically distinct, value-identical shard copies.
Both arms allocate the copies, so allocation alone does not distinguish the arms.

The test records operator intervals and the entire sequence separately.
It does not prove L2 eviction or measure HBM transactions.
The intervening access is a controlled disturbance, not a native Ouro layer trace.
Its result cannot become a full-model speedup or a cache-hit-rate claim.

## Unresolved critical-path attribution

The first full native diagnostic trace changed clean timing after profiler use.
Its clean-before trials take about 106 ms.
Its clean-after trials take about 131 ms.
The profiled request takes about 137 ms.
One pooled observer percentage would hide that drift.
Kernel identity and counts remain useful diagnostics.
Headline component shares and removable E2E cost remain unresolved for this trace.

No P8 measurement exists.
No hardware HBM traffic counter exists in the current evidence.
No candidate receives a GO from these component or modeled screens.
