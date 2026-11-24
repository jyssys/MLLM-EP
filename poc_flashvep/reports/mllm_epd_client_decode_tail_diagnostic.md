# Native MLLM100 Decode Delivery Diagnosis

Evidence: TRACE_ANALYSIS of DIAGNOSTIC online controls.
These runs use Main100 at QPS2. They replay the same 5269 output tokens per run.
Equal output length does not guarantee equal answer text.
Every completed profile aligns EngineCore, API and client token IDs exactly for 100/100 requests.
This alignment does not prove equality to the colocated model.
The table retains clean, observed and clean-after results separately. They are not independent server starts.
Task quality remains pending. No row is qualified Primary performance.

## Completed control results

| Run suffix | Stage | TTFT p95 ms | ITL p99 ms | Max ITL ms | TPOT p95 ms/token | All-ITL PASS % |
|---|---|---:|---:|---:|---:|---:|
| v1_Esync | clean_before | 8178.891 | 23.630 | 58.128 | 8.242 | 74.00 |
| v1_Esync | profile | 8615.375 | 23.851 | 575.304 | 8.592 | 71.00 |
| v1_Esync | clean_after | 8313.920 | 23.720 | 57.430 | 7.782 | 75.00 |
| v2_client_host | clean_before | 7609.945 | 27.396 | 238.284 | 8.165 | 71.00 |
| v2_client_host | profile | 8230.576 | 28.058 | 72.491 | 8.929 | 64.00 |
| v2_client_host | clean_after | 8163.562 | 26.636 | 61.941 | 8.814 | 70.00 |
| v3_API1ms | clean_before | 8808.438 | 27.149 | 1260.143 | 7.895 | 74.00 |
| v3_API1ms | profile | 8124.159 | 25.458 | 658.081 | 8.278 | 74.00 |
| v3_API1ms | clean_after | 8076.140 | 27.567 | 212.885 | 8.759 | 69.00 |
| v4_client_heap | clean_before | 8134.048 | 25.618 | 64.366 | 7.665 | 71.00 |
| v4_client_heap | profile | 8160.910 | 27.019 | 59.675 | 7.817 | 71.00 |
| v4_client_heap | clean_after | 8094.350 | 25.118 | 62.440 | 7.930 | 72.00 |

## Aligned examples

| Run suffix | Request | Token index | Client ITL ms | Engine ITL ms | API ITL ms | API-to-client next-token delay ms |
|---|---|---:|---:|---:|---:|---:|
| v1_Esync | TIGER-Lab__Mantis-Eval__xh_35 | 132 | 575.304 | 6.675 | 6.676 | 569.322 |
| v1_Esync | yubo2333__MMLongBench-Doc__154 | 24 | 568.431 | 6.675 | 6.682 | 562.185 |
| v1_Esync | yubo2333__MMLongBench-Doc__841 | 63 | 489.386 | 7.258 | 7.057 | 483.298 |
| v1_Esync | yubo2333__MMLongBench-Doc__658 | 21 | 488.104 | 7.258 | 7.042 | 481.671 |
| v2_client_host | lmms-lab__MP-DocVQA__57028 | 16 | 72.491 | 6.669 | 43.319 | 29.563 |
| v2_client_host | TIGER-Lab__Mantis-Eval__xh_38 | 16 | 64.890 | 7.287 | 41.218 | 24.565 |
| v2_client_host | yubo2333__MMLongBench-Doc__1001 | 5 | 64.685 | 7.517 | 39.687 | 25.989 |
| v2_client_host | TIGER-Lab__Mantis-Eval__xh_33 | 16 | 64.667 | 7.287 | 41.194 | 24.204 |
| v3_API1ms | yubo2333__MMLongBench-Doc__311 | 41 | 658.081 | 6.602 | 622.461 | 36.136 |
| v3_API1ms | yubo2333__MMLongBench-Doc__1001 | 48 | 233.932 | 14.636 | 15.587 | 218.873 |
| v3_API1ms | TIGER-Lab__Mantis-Eval__xh_33 | 228 | 90.688 | 6.125 | 89.917 | 1.171 |
| v3_API1ms | TIGER-Lab__Mantis-Eval__xh_33 | 10 | 53.666 | 7.308 | 34.103 | 20.186 |
| v4_client_heap | yubo2333__MMLongBench-Doc__148 | 17 | 59.675 | 7.369 | 37.291 | 22.877 |
| v4_client_heap | TIGER-Lab__Mantis-Eval__xh_33 | 10 | 56.049 | 7.306 | 34.509 | 22.147 |
| v4_client_heap | TIGER-Lab__Mantis-Eval__m_17 | 18 | 55.712 | 7.306 | 34.415 | 22.269 |
| v4_client_heap | TIGER-Lab__Mantis-Eval__xh_38 | 10 | 55.424 | 7.306 | 34.497 | 21.690 |

## Controlled and refuted claims

The D API switch interval changed from 5ms to 1ms. E, P and inference configuration stayed unchanged.
The API process recorded its actual 1ms interval. A same-geometry N1 control matched all 16 output token IDs.
This control did not restore 95% all-ITL attainment. It is not a successful decode-protection fix.
A 218.786ms client GC occurred in v2. No active decode stream overlapped it.
That event cannot explain the v2 ITL violations. The planned client-heap control was initially not run.
A later v3 GC lasted 233.657ms during an active stream. The matching client ITL was 233.932ms.
The corresponding Engine ITL was 14.636ms. The API ITL was 15.587ms.
This new event supports a separate client-GC intervention. It does not explain all delivery tails.
Another v3 client ITL was 658.081ms. Its Engine ITL was 6.602ms and API ITL was 622.461ms.
That event overlaps API renderer processing and a 611.808ms API heartbeat delay.
The client heartbeat delay was 4.018ms. No measured client append or GC overlapped that event.
The API renderer enclosure includes waits and worker-thread work. It is not a CPU-exclusive cost.
These examples refute GPU decode slowdown as the sole cause of the client tails.
They do not identify the exact API operation that blocks delivery.
API GC, preprocessing and IPC serialization need separate spans before another causal claim.
The client heap intervention changes no server or inference configuration. Its result remains separate.
The v4 client-heap control completed all three 100-request stages.
Its all-ITL attainment was 71%, 71% and 72%. It did not restore the 95% target.
The observed maximum client GC was 29.994ms. The observed profile maximum client ITL was 59.675ms.
These maxima come from one sequence. They do not establish an independent-start confidence interval.
The absence of a rare API stall in that sequence is not a proven effect of client GC policy.
## Observer qualification

- `native_main100_timeline_CPC_q2_v1_Esync`: aggregate observer gate `False`.
- `native_main100_timeline_CPC_q2_v2_client_host`: aggregate observer gate `False`.
- `native_main100_timeline_CPC_q2_v3_API1ms`: aggregate observer gate `False`.
- `native_main100_timeline_CPC_q2_v4_client_heap`: aggregate observer gate `False`.

The gate checks profile-versus-clean difference and clean-to-clean spread. Each limit is 5%.
A failed gate can indicate an unstable clean baseline. It does not automatically prove profiler overhead.
These diagnostic observations cannot provide qualified GPU-exclusive costs or an online speedup claim.
All client-visible violations remain in the metrics. No timestamp correction removes delivery delay.
CSV and JSON: `mllm_epd_prefill_heavy_token_split_20261124/analysis/native_main100_host_delivery_controls.*`.
The source archive preserves current Python files. Boot-hash comparisons identify unavailable older source bytes.
