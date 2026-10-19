# C14: does a physical transpose resolve the decision?

## Decision and boundary

Status: `MODELED_REFERENCE_FROM_NATIVE_MEASUREMENTS`.
A transpose replay alone cannot establish a recurrence-specific candidate.
No faster candidate prefill measurement exists in this screen.
The proposed transport remains a generic phase-change cost.

The requested interpolation gives seven negative transition budgets at the 10% wall-reduction target.
The only positive budget is 2.138 ms for Base1.4 at N32768.
All eight budgets are negative at the 15% target.
These statements apply only to the specified interpolation and 128 controlled outputs.

This interpolation is not a lower bound.
It is not a measured PP stage or a measured candidate.
It does not reject all pipelines.
A different measured prefill time can change every budget.

The [transpose design](c14_kv_transpose_replay_design_v1.md) and its byte ledger remain unchanged.
No transport implementation or GPU execution occurred.
This supplement adds only arithmetic on completed native measurements.

## Measured inputs and denominators

The inputs come from the [completed native context controls](native_long_context_observation_v1.md).
The source summary is [long_context_complete_v2/summary.json](../analysis/long_context_complete_v2/summary.json).
This screen uses TP1 and TP4 only.
TP2 measurements remain outside this interpolation.

The selected data contain 96 clean trials across 16 native model/TP/context cells.
Each cell contains two frozen prompt patterns with three repeats each.
The patterns are `gsm8k/test/1074` and `math500/test/311`.
Every trial uses C1, fixed R4, and 128 constant-token controlled outputs.
These controls do not measure natural reasoning quality.

The following table lists measured cell medians in milliseconds.
W denotes native API wall.
F denotes native frontend first-token latency.
P denotes native cached TPOT.

| Base | N | W1 | F1 | P1 | W4 | F4 | P4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1.4B | 1024 | 1045.545 | 38.456 | 7.929 | 847.197 | 92.964 | 5.934 |
| 1.4B | 8192 | 1457.683 | 201.268 | 9.892 | 951.656 | 116.135 | 6.579 |
| 1.4B | 16384 | 2011.687 | 483.846 | 12.033 | 1118.899 | 207.321 | 7.176 |
| 1.4B | 32768 | 3373.688 | 1318.026 | 16.190 | 1541.552 | 485.800 | 8.317 |
| 2.6B | 1024 | 2076.056 | 72.891 | 15.772 | 1578.830 | 83.451 | 11.770 |
| 2.6B | 8192 | 2908.533 | 405.367 | 19.710 | 1846.560 | 202.847 | 12.942 |
| 2.6B | 16384 | 4026.822 | 982.145 | 23.974 | 2198.798 | 403.429 | 14.136 |
| 2.6B | 32768 | 6739.666 | 2640.642 | 32.272 | 3050.169 | 963.156 | 16.431 |

The cached TPOT denominator is 127 inter-token intervals.
It is not 128 output tokens.
No interval or layer becomes an independent request sample.
This supplement computes no confidence interval.

## Clock boundary and interpolation

The [runner](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/scripts/native_long_context_control.py:65) measures W with `perf_counter_ns` around one native generate call.
W includes prompt processing, sampling, all outputs, and completion.

Native [IterationStats](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/metrics/stats.py:353) uses `time.time()` for its frontend iteration timestamp.
The [first-token update](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/metrics/stats.py:393) subtracts request arrival time to obtain F.
F is not isolated GPU prefill compute.
It includes frontend and first-output timing effects.

The [engine timestamp update](/home/esjung/.venvs/cp-composition-vllm026/lib/python3.14/site-packages/vllm/v1/metrics/stats.py:420) records first-token and last-token timestamps.
The [existing analyzer](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/scripts/analyze_native_long_context.py:14) obtains P from their difference divided by 127.
Do not subtract an arrival timestamp from these engine timestamps.

Define an algebraic residual E4:

```text
E4 = W4 - F4 - 127*P4
F_reference = F1/4
M0 = E4 + F_reference + 127*P4
   = W4 - F4 + F1/4
```

M0 is the modeled wall at zero additional transition cost.
It preserves the measured native TP4 wall outside the replaced first-token term.
It therefore retains the clock and aggregation residual instead of silently deleting it.

E4 ranges from −0.452400890 ms to 0.638148169 ms.
The negative value occurs for Base1.4 at N32768.
E4 combines different timing boundaries and separately computed medians.
It is not a measured frontend stage or negative physical cost.

The model divides the entire TP1 first-token metric by four.
No source proves that this complete frontend interval scales that way.
The model also assumes unchanged subsequent TP4 decode.
It supplies no placement, chunk schedule, or numerical proof for such a prefill stage.

## Modeled gain and remaining transition budget

Let X denote all additional exposed phase costs.
X includes KV movement, weight handling, metadata publication, staging, and other new overhead.
A complete transition must charge these costs where they affect API wall.
A standalone transfer duration does not automatically equal exposed X.

For a wall-reduction target q:

```text
M(X) = M0 + X
G0 = W4 - M0 = F4 - F1/4
Bq = (1-q)*W4 - M0 = G0 - q*W4
target passes only when 0 <= X <= Bq
```

The 10% target means wall at most 0.90W4.
The 15% target means wall at most 0.85W4.
Neither target means a 10% or 15% throughput increase.

The following table contains MODELED values only.
Time values use milliseconds.
The gain percentage is `100*G0/W4`.

| Base | N | M0 | Zero-cost gain G0 | Gain % | Budget B10 | Budget B15 |
|---|---:|---:|---:|---:|---:|---:|
| 1.4B | 1024 | 763.847 | 83.350 | 9.838 | -1.370 | -43.730 |
| 1.4B | 8192 | 885.838 | 65.818 | 6.916 | -29.348 | -76.930 |
| 1.4B | 16384 | 1032.539 | 86.360 | 7.718 | -25.530 | -81.475 |
| 1.4B | 32768 | 1385.258 | 156.294 | 10.139 | 2.138 | -74.939 |
| 2.6B | 1024 | 1513.602 | 65.228 | 4.131 | -92.655 | -171.597 |
| 2.6B | 8192 | 1745.055 | 101.505 | 5.497 | -83.151 | -175.479 |
| 2.6B | 16384 | 2040.905 | 157.893 | 7.181 | -61.987 | -171.927 |
| 2.6B | 32768 | 2747.173 | 302.996 | 9.934 | -2.021 | -154.529 |

Base1.4 at N32768 has a 2.138373745 ms budget for the 10% target.
This budget covers every exposed added phase cost, not KV transport alone.
Its previously derived remote logical payload is 18 GiB across the cluster.
No bandwidth assumption converts that payload into a measured transition delay.

Base2.6 at N32768 misses the 10% target by 2.020998521 ms even at X=0.
It therefore has no nonnegative transition budget under this interpolation.
The other six cells also miss that target at X=0.
All eight cells miss the 15% target at X=0.

These modeled deficits do not establish that all physical pipelines fail.
They reject only this chosen first-token substitution at these targets with nonnegative X.
A faster actual prefill, another output length, or another supported execution mechanism requires its own evidence.

## Optional output-length parameter

The following equations retain the measured P4 as a constant reference:

```text
W4_model(m) = E4 + F4 + (m-1)*P4
M_model(m,X) = E4 + F1/4 + (m-1)*P4 + X
Bq(m) = F4 - F1/4 - q*(E4 + F4 + (m-1)*P4)
```

At m=128, these equations recover the reported wall reference and budgets.
For other m, the equations provide untested constant-TPOT extrapolations.
Attention context and per-token service can change with output length.
This supplement invents no new measured output-length series.

## Is the transport replay decision-critical?

Not as a standalone architecture experiment at the current evidence state.
The transpose can measure one generic component cost.
It cannot supply the missing faster-prefill result.
It cannot establish that depth ownership creates a distinct transport mechanism.

A tied-aware generic phase-specific PP-to-TP control can use the same ownership mapping.
It can retain the same recurrent state keys and native destination layout.
The matched control must charge packing, communication, scatter, publication, and any weight changes.
The earlier design specifies those controls.

Same-DAG expressibility does not settle realized transport cost.
Actual source strides, page layouts, buffer lifetimes, message grouping, and kernels still matter.
However, a matched physical transpose can reproduce the same work under generic PP labels.
No recurrence-specific advantage follows from a transport measurement alone.

A standalone replay also does not establish transition overlap with prefill.
Streaming completed KV chunks could change the exposed X.
That protocol needs actual producer dependencies and a demonstrated completion path.
Do not assume overlap or use a zero-cost ownership change.

A transport replay becomes more useful after a measured prefill result leaves a meaningful exposed-cost budget.
It can also serve an explicitly requested generic hardware-cost study.
Neither use establishes candidate speedup or recurrence-specific novelty by itself.
The current screen therefore supports deferral, not a universal pipeline rejection.

## Source checks and time

The [supplementary artifact](../artifacts/c14_transpose_decision_screen_v1.json) retains full precision and every selected raw-file hash.
All 96 raw timing hashes match the completed summary.
Independent CPU calculations reproduce all 16 selected cell medians.
All four selected guards and runner statuses report success.

The artifact also records the source hashes for metric semantics and analysis.
It records the frozen transpose design hashes.
Those frozen files remain unchanged.

Actual segment start: 2026-10-19 16:53:12 UTC.
Source/CPU verification checkpoint: 2026-10-19 16:57:32 UTC.
Elapsed time through that checkpoint: 260 seconds.
All eight modeled interpolation and budget identities passed CPU checks.
All source hashes and report links passed checks.
The report lint found zero violations before this time-record update.
No GPU time occurred.

