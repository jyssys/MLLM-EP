# Ouro and Huginn: matched Loop and Token-width comparison

## Result and scope

This report answers the requested short serial comparison. It does not report a new online Serving improvement.
Each arm uses the same fixed BF16 model, recurrent depth, tuned projection policy, and Native connector.
Every new split passes an N8 full-state qualification before its serial measurement.
The state checks compare output tokens, valid KV, Hidden State, full recurrent work, handoff, ACK lifetime, and normal worker exits.

Completed split qualifications: 8, with eight requests per qualification.
Each serial cell uses eight requests within one fresh process start.
Eight requests are not eight independent starts. This short comparison does not provide independent-start confidence intervals.
The arms reuse the same eight model-specific prompt IDs. Request counts do not represent independent texts.

## Matched work

| Model | Loop continuation | Token continuation | Recurrent rows per request |
| --- | --- | --- | --- |
| ouro | 4096 tokens × 1 Loops | 1024 tokens × 4 Loops | 4096 |
| huginn | 2048 tokens × 2 Loops | 128 tokens × 32 Loops | 4096 |
| huginn | 2048 tokens × 4 Loops | 256 tokens × 32 Loops | 8192 |
| huginn | 2048 tokens × 8 Loops | 512 tokens × 32 Loops | 16384 |

The paired arms match recurrent token rows and recurrent dense FLOPs.
They do not match total Attention, Prelude, prompt Coda, state bytes, or P critical-path saving.
Both arms can group all remaining Loops in one scheduler submission when the Native frontier permits it.
The Token arm is not restricted to one-loop host submissions.
Actual submission counts appear below. A model submission is not one non-preemptive GPU kernel.

## Fresh executed recurrent-core and D Prefill costs

CUDA Events measure the following enclosures in milliseconds.
The core columns cover the actual packed recurrent graph instance, including output clones and possible host submission gaps.
The D Prefill columns also include the submitted Prelude and prompt Coda work.
They exclude separate first-output generation and state transfer. They are not Service TTFT.
These measurements are not exclusive kernel unions, physical idle, or concurrent Decode interference.

| Model / Token width | Loop core p50 | Token core p50 | Token / Loop core | Loop D Prefill p50 | Token D Prefill p50 | Token / Loop D Prefill |
| --- | --- | --- | --- | --- | --- | --- |
| ouro / 1024 | 38.967 | 47.449 | 1.218 | 40.034 | 48.064 | 1.201 |
| huginn / 128 | 31.953 | 76.405 | 2.391 | 41.944 | 81.367 | 1.940 |
| huginn / 256 | 63.649 | 91.000 | 1.430 | 74.439 | 96.770 | 1.300 |
| huginn / 512 | 127.479 | 143.948 | 1.129 | 137.780 | 151.044 | 1.096 |

## Submission, source saving, and transfer accounting

P saving is a paired difference against the same-policy serial Native-path Zero control.
It is an event-enclosure difference, not exclusive P kernel saving or measured online queue improvement.
Native also transfers KV. The entire split payload is not incremental communication cost.
GB means 1,000,000,000 bytes.

| Model / width | Arm | P saved p50 ms | D Prefill p95 ms | Largest D submission ms | D submissions/request | Payload GB/request |
| --- | --- | --- | --- | --- | --- | --- |
| ouro / 1024 | loop | 38.999 | 40.169 | 40.183 | 1.000 | 2.433 |
| ouro / 1024 | token | 41.625 | 48.193 | 48.208 | 1.000 | 2.416 |
| huginn / 128 | loop | 42.217 | 42.672 | 43.020 | 1.000 | 5.320 |
| huginn / 128 | token | -31.009 | 82.091 | 82.454 | 1.000 | 5.354 |
| huginn / 256 | loop | 74.704 | 76.460 | 77.306 | 1.000 | 4.974 |
| huginn / 256 | token | 52.787 | 97.748 | 98.057 | 1.000 | 4.999 |
| huginn / 512 | loop | 139.199 | 138.641 | 138.772 | 1.000 | 4.282 |
| huginn / 512 | token | 133.381 | 152.588 | 152.740 | 1.000 | 4.288 |

Huginn Token128 uses a new 1920-row P prefix. The frozen policy has no tuned entry for that P geometry.
That prefix uses the qualified original operator fallback. Its negative P saving is not proof of an inherent Token limitation.
The D-side 128/256/512-row geometries all use frozen operator choices.
A stronger end-to-end Token128 baseline requires separate P1920 tuning and qualification.
Inner core-event counts differ with Loop count. The following control checks the smallest measured advantage.

## Outer-event-only control at Token512

Fresh serial controls remove every inner core event. Both arms retain two outer events on their single D submission.
Output parity, full work, Native lifetime, and normal worker exit pass in both controls.

| Arm | D Prefill p50 ms | Earlier inner-marked / outer-only ratio |
| --- | --- | --- |
| loop | 137.944 | 0.9988 |
| token | 150.807 | 1.0016 |

Token / Loop D execution ratio: 1.093.
The difference remains after removal of unequal inner-event counts.
These controls also use different process starts. Their ratios combine observer effects and ordinary run variation.
They do not prove an online profiler-overhead bound or provide independent-start confidence intervals.

## Published tuned-GEMM confirmation at the same widths

This table reanalyzes the published isolated projection timers. It does not contain fresh recurrent-core timers.
The fixed start-1 policy remains fixed in starts 2 and 3.
Structural multiplicities match the paired rows above. A same-row GEMM does not distinguish a longer suffix from compatible batched suffixes.
Those Attention layouts and readiness conditions remain different.
The search is finite. It does not prove a globally optimal GEMM lower bound.

| Model / Token width | Token / Loop dense sum, three starts |
| --- | --- |
| ouro / 1024 | 1.125, 1.124, 1.129 |
| huginn / 128 | 2.204, 2.164, 2.253 |
| huginn / 256 | 1.198, 1.174, 1.227 |
| huginn / 512 | 0.984, 1.012, 1.054 |

## Attention and boundary differences

The following values are structural counts, not measured kernel time or hardware traffic.

| Model / width | Loop causal pairs per core layer | Token causal pairs per core layer |
| --- | --- | --- |
| ouro / 1024 | 8390656 | 14682112 |
| huginn / 128 | 4196352 | 8128512 |
| huginn / 256 | 8392704 | 15732736 |
| huginn / 512 | 16785408 | 29368320 |

A suffix attends to the already-computed prefix at every remaining recurrent depth.
A whole-prompt Loop contains the triangular causal work across the prompt.
Huginn Loop continuation also performs full-prompt final Coda work on D.
Token continuation performs Prelude and final Coda work only for its suffix.
The measured component fields retain these differences. Equal recurrent rows alone are not full-cost fairness.

## Interpretation limits

Wider Token suffixes can reduce or remove the isolated GEMM disadvantage.
A faster GEMM does not remove dependencies between nonlinear recurrent passes.
Different batching, fusion, persistent execution, or future kernels can change the measured frontier.
This report does not establish an advantage under every Decode-safe execution budget.

It does not establish online TTFT, goodput, or Decode non-inferiority for these new widths.
It does not establish exclusive Loop-specific novelty over an equivalent logical-layer frontier.
Hardware HBM traffic and exclusive kernel attribution remain NOT_MEASURED.

## Evidence

Evidence namespace: `poc_flashvep/reports/ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111`.
The priority plan, immutable configurations, source snapshots, exact-state receipts, and per-request timers remain in that namespace.
[Measured results and structural accounting](ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/analysis/priority_width_comparison_results.json)
[Frozen priority plan](ouro_huginn_prefill_heavy_decode_safe_loop_offloading_20261111/state/priority_width_comparison_plan.json)

The broader workload-discovery campaign resumes after this bounded comparison.
Its original deadline and GPU restoration rules remain unchanged.
