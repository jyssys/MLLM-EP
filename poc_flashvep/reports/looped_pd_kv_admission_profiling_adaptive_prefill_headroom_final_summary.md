# KV admission recovery and Prefill-heavy PD headroom

## Decision and scope

This study does not establish the strict Prefill-heavy and SLO-safe Decode-spare hypothesis on the tested grid.
Admission recovery and profiler recovery are separate results.
Neither result alone proves safe Decode-side borrowing.
The report preserves negative controls and measurement failures.

The strict final decision is `NOT_CONFIRMED_ON_TESTED_GRID` for both models.
The next Dynamic A campaign does not pass this study's entry gate.
This decision does not establish that Prefill-heavy Looped serving is impossible.

## Executive result

The clean PD/DP2 comparison completed 128 requests per arm in three independent starts for each model.
All 1,536 requests completed. The internal output comparisons and fixed-depth work checks passed.
The complete matrix contains 223 completed configurations, nine failed configurations, and 99 unrun or superseded configurations.

| Model and selected workload | Native PD TTFT p95 | Native Mixed DP2 TTFT p95 | Native PD TPOT p95 | Native Mixed DP2 TPOT p95 | Native PD completed req/s | Native Mixed DP2 completed req/s |
| --- | --- | --- | --- | --- | --- | --- |
| Ouro 8192/32, nominal 2.5 req/s | 35.632 s | 4.270 s | 35.292 ms | 160.942 ms | 1.531 | 2.470 |
| Huginn 2560/32, nominal 1.5 req/s | 51.208 s | 37.267 s | 124.440 ms | 209.691 ms | 0.933 | 1.046 |

Each cell is the mean of three independent-start metrics, not a pooled-request quantile.
The arm-specific and paired 95% intervals remain below and in the CSV files.
These finite overloaded cohorts do not establish stable serving capacity.
The large TTFT/Decode-tail trade-off resembles the proposed service-level motivation.
It does not prove that P GPU compute causes the queue.

The qualified Ouro N128 profile shows 66.582% central P kernel union, below the fixed 80% screen.
Its P completion rate reaches 78.800% of the strongest finite matched P reference, below the fixed 85% screen.
The same interval has acceptable Decode tails and no sampled D-credit refusal.
This is a P service queue without the required P-compute saturation proof.

The qualified Huginn N24 profile shows 61.855% central P kernel union.
It also records D KV-credit refusals at 28 of 43 central observation timestamps.
The N128 Huginn traces complete, but their +16.758% and +7.162% wall perturbations fail the fixed 5% limit.
They remain diagnostic. They do not replace the qualified N24 evidence.
No actual D-side borrowing probe runs because the joint prerequisite does not pass.

The model depth remains Ouro R4 and Huginn R32.
The model weights, BF16 precision, and native attention semantics do not change.
The existing numerical patches remain enabled in all relevant comparisons.
Original-HF equivalence and downstream accuracy are not established here.

## Source and environment

The execution baseline commit is `f4ecea59f36f620c69c4fef09c590bf790802cd9`.
The private runtime HEAD is `6b164d3844d35ab8e753df1864f0c87dc180401a`.
Its preserved working diff SHA256 is `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.
Each run records its actual source archive hash.
The working branch is `flashvep/looped-pd-kv-profile-headroom-recovery`.
The publication branch is `flashvep/looped-pd-kv-profile-headroom-recovery-results`.
The final Git SHA appears in the verified publication receipt and final user response.
A commit cannot contain its own hash without changing that hash.

The environment uses Python 3.14.6, torch 2.11.0+cu129, CUDA 12.9, transformers 5.16.1, NIXL 1.4.1, and FA4 4.0.0b30.
Ouro revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Huginn revision is `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
NVML reports 85,520,809,984 physical bytes, or 79.6474609375 GiB, per H100.
CUDA reports 85,028,372,480 managed bytes, or 79.1888427734375 GiB.
Earlier percent_HBM columns use CUDA-visible capacity. The physical-memory audit supplies both explicit denominators.
Clean measurements use one isolated GPU pair while all other managed burners remain stopped.
The experiment does not assume independent pairs are free of shared-host interference.

## Data and frozen decisions

Prompts come from the pinned ShareGPT and LongBench public-text pool.
Length-matched prefixes are controlled workloads, not natural Azure requests.
Generation uses greedy ignore-EOS output caps.
This study does not report these outputs as natural response-length distributions.
The 24 pilot prompts and 128 final-repeat prompts are disjoint.
Final repeats use three arrival seeds and physical pairs 0/1, 4/5, and 2/3.
All paired PD and DP2 starts use identical prompts, generation seeds, and arrivals.
Calibration prompts are separate from both sets.
Huginn input plus output remains within 4096 tokens.

SLO limits freeze before held-out pilots.
The primary joint request SLO uses TTFT and request-average TPOT.
Decode-spare assessment also requires the separately frozen token-ITL tail limit.
The tighter per-metric Atomic/C2048 serial calibration determines each limit.
No candidate result changes these limits.

## Admission recovery

The original Ouro 8192/64 condition has a 66-GiB KV pool and about 6.047 GiB per reserved long request.
Ten reservations fit in 22,528 native blocks.
The next request can fail D reservation before P compute starts.
This is an admission limit, not evidence of P GPU compute saturation.

Supported controls reduce D workspace, enlarge the legal D pool, bound draining reservations, and tune wire credit.
The D pool increases to 25,088 blocks, or 73.5 GiB.
Native ACK and buffer-lifetime checks remain active.
No state releases before a verified native acknowledgment.
The P pool remains unchanged in these controls.
The exact before/after request metrics remain in `serving_runs.csv`.
Memory categories and percentages remain in `gpu_memory_inventory.csv` and `workload_capacity_envelope.csv`.
Raw refusal frequency is not exclusive waiting time.
Binding refusal and co-occurring refusal remain separate fields.

Huginn reserves 256 uniform planes while only 132 planes carry valid state.
The 48.4375% difference is structural accounting.
This study does not implement compact physical allocation or claim that difference as reclaimed HBM.
The native wire path already excludes unused boundary planes.
The 3500/32 control can fill all four D reservations exactly.
The lighter 2560/32 and 3072/32 cases test the capacity/compute trade-off separately.

## Profiling recovery

Historical broad Nsight starts stopped after 80/128 and 14/128 responses.
The exact first blocked operation remains unknown.
Selected-stage process startup and bounded cohorts restore measured progress without replacing model execution.
Several Nsight controls still exceed the predeclared 5% perturbation limit.
Those traces remain diagnostic only.

The native CUDA-only profiler passes a matched 24-request Huginn 3500/32 comparison.
It records 472,272 measured-window kernel invocations with exact 24/24 output parity.
The matched wall difference is +1.351%.
Its whole-window P kernel union is 63.349%.
Kernel union is not SM occupancy or an exclusive E2E component.
The attempted three-second activity toggle returns no kernels despite completed inference.
That empty collector result is not zero GPU activity.
Subsequent matched profiles use the checked full-cohort collector.

## Generic runtime controls

The original evidence writer synchronously fsyncs individual records on a scheduling thread.
The replacement reuses the project's durable background writer.
It preserves record contents and checks enqueued/written counts before accepting a terminal result.
This change does not disable evidence or change model arithmetic.

Native CUDA graphs capture new row geometries lazily.
The warm-up control exercises supported row shapes with real fixed-depth startup requests.
It preserves the original full-length warm-up first.
It applies the same startup principle to PD and Native Mixed DP2.
Additional warm-up requests do not enter measured latency or throughput denominators.
`graph_warmup_controls.csv` reports actual measured-phase capture counts and request tails.
Capture host spans include synchronization and garbage collection.
They are not pure GPU execution time.

Exact CPU descriptor packing is a generic transport control.
Its CPU replay reduces input descriptors without changing the final coalesced descriptors.
CPU replay timing is not an E2E speedup.
The warmed packed/unpacked GPU controls report actual request effects separately.
None of these generic controls is a new Looped-specific method.

## P/D capacity and causal gates

`independent_capacity.csv` contains measured finite P-only and prepared-D service results.
P-only service excludes D reservation and handoff.
Prepared D uses actual native valid KV imports before starting its Decode timer.
Neither finite rate proves sustainable online capacity.
Three repeats within one process are not three independent starts.

The P gate requires growing D-clear P-ready backlog, at least 80% kernel union, and at least 85% of the strongest matched finite P reference.
The ready-queue count excludes every submitted unretired request.
D-reservation waiting is not P compute-ready waiting.
The D gate requires legal free blocks and request TPOT p95 below 90% of its frozen limit.
The separate token-ITL p99 limit also applies.
All conditions must overlap within one server start.
Borrowing remains unproven without actual added D work with checked state and a no-regression comparison.

## Execution counts

| Terminal status | Configurations |
| --- | --- |
| COMPLETE | 223 |
| FAILED | 9 |
| NOT_RUN | 99 |

## Warmed serving results

All listed latency values are milliseconds.
These are finite arrival-to-drain results, not proven sustainable capacity.

| Run | Offered | Completed | Target req/s | Completed req/s | TTFT p95 | TPOT p95 | ITL p99 | Joint SLO goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_huginn_primary_DP_start1_rate_matched_v2_n128 | 128 | 128 | 1.500 | 1.035 | 27844.238 | 210.295 | 1129.223 | 0.065 |
| confirm_huginn_primary_DP_start2_rate_matched_v2_n128 | 128 | 128 | 1.500 | 1.032 | 38054.517 | 208.606 | 1130.283 | 0.016 |
| confirm_huginn_primary_DP_start3_rate_matched_v2_n128 | 128 | 128 | 1.500 | 1.072 | 45901.519 | 210.173 | 2018.971 | 0.042 |
| confirm_huginn_primary_PD_start1_rate_matched_v2_n128 | 128 | 128 | 1.500 | 0.947 | 39006.554 | 126.029 | 129.569 | 0.148 |
| confirm_huginn_primary_PD_start2_rate_matched_v2_n128 | 128 | 128 | 1.500 | 0.923 | 52418.960 | 123.747 | 131.586 | 0.065 |
| confirm_huginn_primary_PD_start3_rate_matched_v2_n128 | 128 | 128 | 1.500 | 0.928 | 62196.999 | 123.544 | 131.166 | 0.022 |
| confirm_ouro_primary_DP_start1_rate_matched_v2_n128 | 128 | 128 | 2.500 | 2.259 | 1469.951 | 159.024 | 459.518 | 0.582 |
| confirm_ouro_primary_DP_start2_rate_matched_v2_n128 | 128 | 128 | 2.500 | 2.553 | 3030.540 | 156.962 | 556.308 | 0.359 |
| confirm_ouro_primary_DP_start3_rate_matched_v2_n128 | 128 | 128 | 2.500 | 2.598 | 8309.730 | 166.840 | 551.184 | 0.122 |
| confirm_ouro_primary_PD_start1_rate_matched_v2_n128 | 128 | 128 | 2.500 | 1.851 | 13657.743 | 35.671 | 41.573 | 0.405 |
| confirm_ouro_primary_PD_start2_rate_matched_v2_n128 | 128 | 128 | 2.500 | 1.323 | 45814.410 | 35.075 | 43.429 | 0.062 |
| confirm_ouro_primary_PD_start3_rate_matched_v2_n128 | 128 | 128 | 2.500 | 1.419 | 47425.011 | 35.130 | 41.116 | 0.033 |
| warm_aggregated_huginn_c2048_q0p6_pilot24 | 24 | 24 | 0.600 | 0.696 | 1067.746 | 193.474 | 1016.268 | 0.638 |
| warm_aggregated_huginn_c2048_q1p0_pilot24 | 24 | 24 | 1.000 | 0.972 | 3665.045 | 205.307 | 1118.093 | 0.041 |
| warm_aggregated_huginn_c2048_q1p2_pilot24 | 24 | 24 | 1.200 | 1.002 | 5645.823 | 205.464 | 1120.896 | 0.000 |
| warm_aggregated_huginn_c512_q0p6_pilot24 | 24 | 24 | 0.600 | 0.695 | 1128.766 | 199.710 | 961.587 | 0.608 |
| warm_aggregated_huginn_c512_q1p0_pilot24 | 24 | 24 | 1.000 | 0.965 | 3647.811 | 205.348 | 1058.302 | 0.040 |
| warm_aggregated_huginn_c512_q1p2_pilot24 | 24 | 24 | 1.200 | 0.996 | 5806.128 | 206.778 | 1060.805 | 0.000 |
| warm_aggregated_ouro_c2048_q1p5_pilot24 | 24 | 24 | 1.500 | 1.630 | 488.237 | 68.803 | 337.774 | 1.087 |
| warm_aggregated_ouro_c2048_q2p5_pilot24 | 24 | 24 | 2.500 | 2.168 | 1438.038 | 86.288 | 444.416 | 0.452 |
| warm_aggregated_ouro_c2048_q3p0_pilot24 | 24 | 24 | 3.000 | 2.293 | 2249.447 | 85.032 | 452.709 | 0.000 |
| warm_aggregated_ouro_c2048_q4p0_pilot24 | 24 | 24 | 4.000 | 2.438 | 2890.548 | 78.184 | 451.511 | 0.000 |
| warm_aggregated_ouro_c2048_q5p0_pilot24 | 24 | 24 | 5.000 | 2.498 | 3483.937 | 75.709 | 451.025 | 0.000 |
| warm_aggregated_ouro_c512_q1p5_pilot24 | 24 | 24 | 1.500 | 1.600 | 981.655 | 81.988 | 233.843 | 0.467 |
| warm_aggregated_ouro_c512_q2p5_pilot24 | 24 | 24 | 2.500 | 1.976 | 2535.842 | 96.168 | 303.504 | 0.000 |
| warm_aggregated_ouro_c512_q3p0_pilot24 | 24 | 24 | 3.000 | 2.094 | 3404.629 | 94.015 | 324.655 | 0.000 |
| warm_aggregated_ouro_c512_q4p0_pilot24 | 24 | 24 | 4.000 | 2.140 | 4582.201 | 88.632 | 333.379 | 0.000 |
| warm_aggregated_ouro_c512_q5p0_pilot24 | 24 | 24 | 5.000 | 2.265 | 4971.844 | 82.439 | 381.957 | 0.000 |
| warm_epoch_huginn_c2048_q0p6_pilot24 | 24 | 24 | 0.600 | 0.679 | 2767.930 | 123.376 | 130.303 | 0.622 |
| warm_epoch_huginn_c2048_q1p0_pilot24 | 24 | 24 | 1.000 | 0.692 | 12936.078 | 116.026 | 123.794 | 0.086 |
| warm_epoch_huginn_c2048_q1p2_pilot24 | 24 | 24 | 1.200 | 0.728 | 14233.890 | 120.993 | 132.248 | 0.091 |
| warm_epoch_huginn_c2560_q0p6_pilot24 | 24 | 24 | 0.600 | 0.678 | 2717.491 | 145.315 | 206.876 | 0.622 |
| warm_epoch_huginn_c2560_q1p0_pilot24 | 24 | 24 | 1.000 | 0.801 | 7784.640 | 147.237 | 240.893 | 0.167 |
| warm_epoch_huginn_c2560_q1p2_pilot24 | 24 | 24 | 1.200 | 0.737 | 13950.297 | 119.062 | 128.213 | 0.092 |
| warm_epoch_ouro_c2048_q1p5_pilot24 | 24 | 24 | 1.500 | 1.164 | 8312.615 | 32.147 | 36.031 | 0.000 |
| warm_epoch_ouro_c2048_q2p5_pilot24 | 24 | 24 | 2.500 | 1.733 | 5589.761 | 40.079 | 47.426 | 0.000 |
| warm_epoch_ouro_c2048_q3p0_pilot24 | 24 | 24 | 3.000 | 1.261 | 11816.612 | 31.687 | 34.274 | 0.000 |
| warm_epoch_ouro_c2048_q4p0_pilot24 | 24 | 24 | 4.000 | 1.766 | 7640.727 | 33.778 | 45.460 | 0.000 |
| warm_epoch_ouro_c2048_q5p0_pilot24 | 24 | 24 | 5.000 | 1.789 | 8232.386 | 32.804 | 44.135 | 0.000 |
| warm_epoch_ouro_c6144_q1p5_pilot24 | 24 | 24 | 1.500 | 1.458 | 4303.368 | 31.654 | 34.120 | 0.182 |
| warm_epoch_ouro_c6144_q2p5_pilot24 | 24 | 24 | 2.500 | 1.965 | 3913.781 | 35.505 | 44.316 | 0.246 |
| warm_epoch_ouro_c6144_q3p0_pilot24 | 24 | 24 | 3.000 | 1.506 | 8692.612 | 32.148 | 35.861 | 0.000 |
| warm_epoch_ouro_c6144_q4p0_pilot24 | 24 | 24 | 4.000 | 2.044 | 5764.868 | 33.860 | 44.000 | 0.170 |
| warm_epoch_ouro_c6144_q5p0_pilot24 | 24 | 24 | 5.000 | 2.053 | 6493.642 | 34.107 | 44.323 | 0.171 |
| warm_huginn_Dslots4_c2048_q0p6_pilot24 | 24 | 24 | 0.600 | 0.672 | 2973.457 | 135.797 | 143.283 | 0.616 |
| warm_huginn_Dslots4_c2048_q1p0_pilot24 | 24 | 24 | 1.000 | 0.829 | 7299.491 | 125.067 | 132.143 | 0.138 |
| warm_huginn_Dslots4_c2048_q1p2_pilot24 | 24 | 24 | 1.200 | 0.792 | 11562.424 | 121.954 | 127.201 | 0.099 |
| warm_huginn_Dslots4_c2560_q0p6_pilot24 | 24 | 24 | 0.600 | 0.669 | 2904.369 | 138.864 | 146.227 | 0.558 |
| warm_huginn_Dslots4_c2560_q1p0_pilot24 | 24 | 24 | 1.000 | 0.843 | 6853.668 | 123.324 | 128.560 | 0.176 |
| warm_huginn_Dslots4_c2560_q1p2_pilot24 | 24 | 24 | 1.200 | 0.856 | 9334.592 | 124.644 | 131.497 | 0.143 |
| warm_huginn_Pslots2_c2048_q0p8_pilot24 | 24 | 24 | 0.800 | 0.765 | 6619.614 | 138.996 | 212.076 | 0.287 |
| warm_huginn_Pslots2_c2048_q1p0_pilot24 | 24 | 24 | 1.000 | 0.745 | 11035.046 | 130.336 | 202.189 | 0.031 |
| warm_huginn_Pslots2_c2048_q1p2_pilot24 | 24 | 24 | 1.200 | 0.747 | 12920.913 | 142.770 | 213.755 | 0.062 |
| warm_huginn_Pslots2_c2560_q0p8_pilot24 | 24 | 24 | 0.800 | 0.761 | 6171.671 | 138.399 | 214.304 | 0.317 |
| warm_huginn_Pslots2_c2560_q1p0_pilot24 | 24 | 24 | 1.000 | 0.747 | 10704.782 | 131.499 | 187.678 | 0.093 |
| warm_huginn_Pslots2_c2560_q1p2_pilot24 | 24 | 24 | 1.200 | 0.677 | 16387.461 | 163.783 | 272.430 | 0.028 |
| warm_secondary_aggregated_huginn_l3072_c2048_q1p2_pilot24_pool25088_retry_v2 | 24 | 24 | 1.200 | 0.921 | 8062.526 | 231.593 | 1343.824 | 0.000 |
| warm_secondary_aggregated_huginn_l3072_c512_q1p2_pilot24 | 24 | 24 | 1.200 | 0.915 | 7984.608 | 229.479 | 1266.921 | 0.000 |
| warm_secondary_aggregated_ouro_l8192_c2048_q2p0_pilot24 | 24 | 24 | 2.000 | 2.193 | 1003.200 | 127.700 | 451.938 | 0.366 |
| warm_secondary_aggregated_ouro_l8192_c512_q2p0_pilot24 | 24 | 24 | 2.000 | 2.037 | 1625.671 | 188.134 | 325.628 | 0.255 |
| warm_secondary_huginn_l3072_c2048_q0p8_pilot24 | 24 | 24 | 0.800 | 0.660 | 10163.512 | 121.325 | 128.950 | 0.083 |
| warm_secondary_huginn_l3072_c2048_q1p0_pilot24 | 24 | 24 | 1.000 | 0.722 | 11387.897 | 121.009 | 127.706 | 0.090 |
| warm_secondary_huginn_l3072_c2048_q1p2_pilot24 | 24 | 24 | 1.200 | 0.710 | 15029.687 | 122.630 | 130.373 | 0.089 |
| warm_secondary_huginn_l3072_c3072_q0p8_pilot24 | 24 | 24 | 0.800 | 0.723 | 7336.482 | 121.891 | 129.491 | 0.151 |
| warm_secondary_huginn_l3072_c3072_q1p0_pilot24 | 24 | 24 | 1.000 | 0.704 | 12273.681 | 120.920 | 126.867 | 0.088 |
| warm_secondary_huginn_l3072_c3072_q1p2_pilot24 | 24 | 24 | 1.200 | 0.668 | 17172.056 | 116.885 | 124.738 | 0.056 |
| warm_secondary_ouro_l8192_c2048_q2p0_pilot24 | 24 | 24 | 2.000 | 1.505 | 6620.706 | 33.970 | 38.570 | 0.251 |
| warm_secondary_ouro_l8192_c2048_q2p5_pilot24 | 24 | 24 | 2.500 | 1.022 | 15754.203 | 34.464 | 37.482 | 0.000 |
| warm_secondary_ouro_l8192_c2048_q3p0_pilot24 | 24 | 24 | 3.000 | 1.445 | 10302.204 | 35.138 | 39.681 | 0.120 |
| warm_secondary_ouro_l8192_c8192_q2p0_pilot24 | 24 | 24 | 2.000 | 1.791 | 4646.836 | 35.249 | 40.143 | 0.746 |
| warm_secondary_ouro_l8192_c8192_q2p5_pilot24 | 24 | 24 | 2.500 | 1.785 | 6159.939 | 35.579 | 41.600 | 0.595 |
| warm_secondary_ouro_l8192_c8192_q3p0_pilot24 | 24 | 24 | 3.000 | 1.272 | 12555.275 | 35.138 | 41.397 | 0.106 |

## Same-start kernel and SLO assessment

| Model | Actual profiled run | P kernel union % | P completed/s | P/reference | Ready half-growth | D TPOT p95 | Central D ITL p99 | D free blocks min | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | long_profile_ouro_PD_start1_P_only_n128 | 66.582 | 1.797 | 0.788 | 0.214 | 38.476 | 43.747 | 8640 | P_SERVICE_QUEUE_BUT_NOT_GPU_COMPUTE_BOUND |
| huginn | profile_warm_huginn_selected_both_24 | 61.855 | 0.838 | 0.702 | -1.547 | 151.756 | 182.416 | 2368 | KV_ADMISSION_BOUND |

## Confidence intervals and full metrics

`paired_repetitions.csv` records each independent PD/DP2 pair.
`paired_start_uncertainty.csv` reports two-sided Student-t 95% intervals across three starts.
A missing third start produces no invented interval.
The raw table retains mean, P50/P95/P99 request metrics, output throughput, token tails, and peak HBM.
`e2e_component_shares_summary.csv` reports duration-weighted exclusive host phase percentages.
Those phases sum to the host E2E interval.
They do not identify exclusive dispatch, GEMM, or DMA percentages.
Cumulative overlapping transfer durations cannot substitute for those percentages.

## Limits and next step

The tested grid does not prove that recurrence prevents Prefill-heavy serving.
Qualified fixed-row numerical geometry, host submission, KV layout, transport, and model depth can all affect these results.
Structural row padding applies only to selected projections, not the entire model.
The study retains those qualified fixes rather than exchanging correctness for throughput.
A full Dynamic A campaign is not justified without same-start compute saturation and safe borrowing evidence.
Remaining unrun workloads and failed jobs remain listed in `failures_and_unrun.md`.
The figure manifest names every exact plotted source.
The reference-code audit states why LoopServing is not a drop-in runtime replacement.
Reproduction commands and immutable configuration/source archives remain available.

## Final independent-start comparison

All latency columns use milliseconds. Each shown p95 or p99 is the mean of independent-start quantiles.
The table does not pool requests across starts. The paired uncertainty table follows.

| Model | Arm | Starts | Offered | Completed | Target req/s | Completed req/s | Mean TTFT p95 | Mean TPOT p95 | Mean ITL p99 | Mean joint goodput |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | PD | 3 | 384 | 384 | 2.500 | 1.531 | 35632.388 | 35.292 | 42.040 | 0.167 |
| ouro | DP | 3 | 384 | 384 | 2.500 | 2.470 | 4270.074 | 160.942 | 522.337 | 0.354 |
| huginn | PD | 3 | 384 | 384 | 1.500 | 0.933 | 51207.504 | 124.440 | 130.774 | 0.078 |
| huginn | DP | 3 | 384 | 384 | 1.500 | 1.046 | 37266.758 | 209.691 | 1426.159 | 0.041 |

DP means two native mixed colocated replicas. PD means native 1P1D.
Both arms use two GPUs, identical prompts, identical generation seeds, and paired arrival schedules.
These finite N128 cohorts do not establish infinite-horizon sustainable capacity.

| Model | Paired metric | Independent starts | Mean | 95% CI lower | 95% CI upper |
| --- | --- | --- | --- | --- | --- |
| ouro | paired_goodput_difference_PD_minus_DP | 3 | -0.1877074887001436 | -0.4475081966559077 | 0.07209321925562051 |
| ouro | paired_completed_throughput_ratio_PD_over_DP | 3 | 0.6278798248833632 | 0.214953535188938 | 1.0408061145777885 |
| ouro | paired_TTFT_p95_difference_ms | 3 | 31362.31415501485 | -10139.246341920669 | 72863.87465195038 |
| ouro | paired_TPOT_p95_difference_ms | 3 | -125.65007512607883 | -138.81229958954532 | -112.48785066261236 |
| ouro | paired_ITL_p99_difference_ms | 3 | -480.2970044221714 | -614.482316461531 | -346.11169238281184 |
| huginn | paired_goodput_difference_PD_minus_DP | 3 | 0.037329490483882205 | -0.09352357520108559 | 0.16818255616885 |
| huginn | paired_completed_throughput_ratio_PD_over_DP | 3 | 0.8918755230406153 | 0.8294194904369535 | 0.9543315556442772 |
| huginn | paired_TTFT_p95_difference_ms | 3 | 13940.746378712356 | 7500.1765012194355 | 20381.316256205275 |
| huginn | paired_TPOT_p95_difference_ms | 3 | -85.25098723619777 | -88.30424500929992 | -82.19772946309563 |
| huginn | paired_ITL_p99_difference_ms | 3 | -1295.3852437840148 | -2569.8737066247354 | -20.89678094329429 |

The interval uses a two-sided Student-t distribution with two degrees of freedom.
Three starts give wide uncertainty. A negative lower confidence bound for a nonnegative metric is a statistical interval, not negative physical latency.
`final_start_level_metrics.csv` provides arm-specific start-level intervals.

## Original Ouro 8192/64 admission controls

| Supported control | Offered | Completed | Completed req/s | TTFT p95 ms | TPOT p95 ms | ITL p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| original | 24 | 24 | 0.959 | 12949.461 | 60.819 | 328.998 |
| workspace128 | 24 | 24 | 0.914 | 13765.376 | 57.209 | 288.426 |
| workspace128_pool25088 | 24 | 24 | 1.044 | 11527.994 | 57.268 | 263.888 |
| workspace128_pool25088_drain0 | 24 | 24 | 1.350 | 6847.949 | 59.732 | 273.579 |

These are single N24 starts with matched original arrival traces. They do not have independent-start confidence intervals.
The controls change supported resource budgets. They do not reduce recurrent work or release unacknowledged state.
The improved original control still must satisfy the frozen Decode tail limit. Admission relief alone is not a Prefill-heavy proof.

## Actual selected-point device profiles

| Profiled start | N | Wall change % | Perturbation qualified | Central P kernel union % | P compute gate | D SLO gate | D credit gate | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_profile_huginn_PD_start1_P_only_n128 | 128 | 7.162 | False | 66.909 | False | True | True | KV_ADMISSION_BOUND |
| long_profile_huginn_PD_start1_both_n128 | 128 | 16.758 | False | 60.848 | False | True | True | KV_ADMISSION_BOUND |
| long_profile_ouro_PD_start1_P_only_n128 | 128 | 3.957 | True | 66.582 | False | True | True | P_SERVICE_QUEUE_BUT_NOT_GPU_COMPUTE_BOUND |
| profile_warm_huginn_pressure_P_only_24 | 24 | 13.539 | False | 58.949 | False | True | True | MIXED_OR_INCONCLUSIVE |
| profile_warm_huginn_pressure_both_24 | 24 | 10.238 | False | 60.384 | False | True | True | KV_ADMISSION_BOUND |
| profile_warm_huginn_selected_both_24 | 24 | -0.654 | True | 61.855 | False | True | True | KV_ADMISSION_BOUND |
| profile_warm_ouro_nsys_alternative24 | 24 | 6.635 | False | 66.716 | False | True | True | PROFILING_OR_MEASUREMENT_BLOCKED |
| profile_warm_ouro_selected_P_only_24 | 24 | 48.957 | False | 40.246 | False | True | True | PROFILING_OR_MEASUREMENT_BLOCKED |
| profile_warm_ouro_selected_both_24 | 24 | 7.710 | False | 66.704 | False | True | True | PROFILING_OR_MEASUREMENT_BLOCKED |

Each row uses that profiled start only. A profile must remain within the predeclared 5% wall-perturbation limit.
Kernel union is recorded device activity. It is not SM occupancy, achieved bandwidth, or exclusive compute ownership.
A failed perturbation gate leaves the trace diagnostic. It does not become a clean-latency measurement.
The central window covers 20% to 90% of the actual arrival span. Arrival and drain remain separately visible.

## Accuracy scope and full-work evidence

`correctness_and_parity.md` lists the byte and output gates. Their denominators remain separate.
`paired_repetitions.csv` reports PD/DP output parity within each independent pair.
`state/cross_start_output_parity.json` reports PD output parity across independent arrival seeds.
`full_depth_checks.csv` checks actual recurrent rows, not padded projection rows.
No GSM8K, MATH-500, or original-HF accuracy result is measured in this campaign.

## Workload and rate-selection limit

The N24 Poisson pilot realized an arrival-interval rate about 24.86% above its nominal rate.
A declared rule adjusted the N128 nominal rates before any N128 measurement.
Ouro confirmation uses 8192/32 at 2.5 nominal requests/s. Huginn confirmation uses 2560/32 at 1.5 nominal requests/s.
The old nominal 2.0 and 1.2 pilot rates are not described as identical-rate independent repeats.
Both paired policies receive the same revised schedule. No SLO threshold changes.
The primary 6144/64 Ouro and 2560/32 Huginn sweeps remain in the complete configuration matrix.
The secondary 3072/32 Huginn DP2 C2048 control first failed with a real CUDA OOM.
One smaller legal KV-pool retry completed its byte/output gate and N24 pilot.
The original failure and its source/configuration remain preserved.

## Memory denominator correction

NVML reports 85,520,809,984 bytes, or 79.6474609375 GiB, per physical GPU.
CUDA reports 85,028,372,480 managed bytes, or 79.1888427734375 GiB.
The 0.4586181640625-GiB difference is outside the CUDA-managed capacity.
Earlier `percent_HBM` fields divide by CUDA-visible capacity. Their raw records remain unchanged.
`physical_hbm_accounting.csv` provides both explicit denominators for every recorded owner endpoint.
Do not add active KV bytes to the pool that already contains them.
Do not add allocator totals to their contained tensor-storage categories.

## E2E percentages and unmeasured device components

`e2e_component_shares_summary.csv` reports additive host-observed phase percentages.
Their weighted sum covers request E2E. The CSV does not report exclusive dispatch, GEMM, or DMA percentages.
NIXL host-observed cumulative durations can overlap compute and other transfers.
Exclusive dispatch/combine/kernel percentages, SM occupancy, and measured HBM bandwidth remain NOT_MEASURED when no valid device attribution exists.

## Elapsed time and final artifacts

This campaign used 7.812 hours through analysis closeout.
The exact timestamp pair remains in `state/elapsed_time.json`.
`figure_manifest.json` identifies each plotted file, exact run namespace, and source hash.
`config_matrix.csv` retains failed, superseded, and unrun conditions.
`failures_and_unrun.md` explains their limits.
`state/resource_closeout.json` records the final managed-burner restoration.
`state/results_publication.json` records the publication branch and verified remote SHA.
The active staged index and the external qualified runtime remain unchanged.

## Phase-by-phase decisions

| Model | Phase | Decision | Evidence scope |
| --- | --- | --- | --- |
| ouro | 0: internal correctness | PASS | Recorded token/state gates and full-depth audit only. Original-HF fidelity remains unmeasured. |
| ouro | A: admission | CONDITIONAL | Supported controls reduce specific binding limits. Long-context KV and slot limits remain workload dependent. |
| ouro | B: profiling | PASS_SCOPED | 1 matched profiles meet the predeclared wall-perturbation gate. Other profiles remain diagnostic. |
| ouro | C: workload/QPS search | COMPLETED_TESTED_GRID | Primary and secondary length sweeps remain visible. The chosen finite point has three paired starts when recorded. |
| ouro | D: same-start target | NOT_CONFIRMED | P compute gate=False. D SLO gate=True. D credit gate=True. |
| ouro | E: actual D borrowing | NOT_RUN_PREREQUISITE | No complete same-start compute/SLO/credit prerequisite supports the real-work probe. No synthetic point substitutes for it. |
| ouro | F: artifacts | COMPLETE_AFTER_VERIFIED_PUSH | The final response supplies the remote SHA. The publication receipt records the independent verification. |
| huginn | 0: internal correctness | PASS | Recorded token/state gates and full-depth audit only. Original-HF fidelity remains unmeasured. |
| huginn | A: admission | CONDITIONAL | Supported controls reduce specific binding limits. Long-context KV and slot limits remain workload dependent. |
| huginn | B: profiling | PASS_SCOPED | 4 matched profiles meet the predeclared wall-perturbation gate. Other profiles remain diagnostic. |
| huginn | C: workload/QPS search | COMPLETED_TESTED_GRID | Primary and secondary length sweeps remain visible. The chosen finite point has three paired starts when recorded. |
| huginn | D: same-start target | NOT_CONFIRMED | P compute gate=False. D SLO gate=True. D credit gate=True. |
| huginn | E: actual D borrowing | NOT_RUN_PREREQUISITE | No complete same-start compute/SLO/credit prerequisite supports the real-work probe. No synthetic point substitutes for it. |
| huginn | F: artifacts | COMPLETE_AFTER_VERIFIED_PUSH | The final response supplies the remote SHA. The publication receipt records the independent verification. |

## Why the target remains unconfirmed

Ouro admission relief changes available reservations and transfer draining. It does not create additional HBM or prove GPU saturation.
Huginn lighter inputs reduce per-request KV capacity. They also reduce Prefill compute, so the trade-off remains visible.
Several starts have legal D memory and acceptable Decode tails. Their P device-activity or service/backlog gate still fails.
Other starts have stronger pressure but binding D admission, a failed Decode tail, or a materially perturbed profile.
These conditions cannot be combined across starts into a positive result.
The measured kernel gaps are not exclusive CPU delay. A profiler does not identify the first historical blocked operation.
The experiment therefore does not prove that recurrence itself prevents Prefill-heavy serving.
A full Dynamic A campaign remains unjustified on this evidence.

## Implemented changes and explicit non-changes

All new source resides in `poc_flashvep/looped_pd_headroom_recovery/`.
The implementation uses supported native memory/slot/wire-credit controls. Native release and ACK dependencies remain intact.
The harness adds exact descriptor preparation, durable asynchronous evidence, and real fixed-depth graph warm-up.
The native CUDA-only profiler and selected-stage launcher collect the device evidence.
These controls are generic runtime changes, not a new Looped-specific scheduling contribution.
The external qualified runtime and its working diff remain unchanged.
No compact Huginn KV allocator, delayed-D-reserve protocol, new Dynamic A policy, or actual D borrowing path is implemented.
No model weight, BF16 precision, fixed depth, or attention equation changes.
LoopServing remains a read-only reference. The applicability matrix explains the API and execution differences.

## Native handoff and source-retention observations

| Model | Independent run | Host interval | N | Mean ms | P95 ms |
| --- | --- | --- | --- | --- | --- |
| ouro | confirm_ouro_primary_PD_start1_rate_matched_v2_n128 | P_compute_observation | 128 | 1032.558 | 1477.541 |
| ouro | confirm_ouro_primary_PD_start1_rate_matched_v2_n128 | compute_end_to_commit | 128 | 460.712 | 1561.705 |
| ouro | confirm_ouro_primary_PD_start1_rate_matched_v2_n128 | commit_to_D_received | 128 | 16.900 | 20.459 |
| ouro | confirm_ouro_primary_PD_start1_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 479.324 | 1579.118 |
| ouro | confirm_ouro_primary_PD_start2_rate_matched_v2_n128 | P_compute_observation | 128 | 1803.024 | 2724.600 |
| ouro | confirm_ouro_primary_PD_start2_rate_matched_v2_n128 | compute_end_to_commit | 128 | 570.663 | 2077.420 |
| ouro | confirm_ouro_primary_PD_start2_rate_matched_v2_n128 | commit_to_D_received | 128 | 19.227 | 22.203 |
| ouro | confirm_ouro_primary_PD_start2_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 591.828 | 2096.293 |
| ouro | confirm_ouro_primary_PD_start3_rate_matched_v2_n128 | P_compute_observation | 128 | 1737.897 | 2700.514 |
| ouro | confirm_ouro_primary_PD_start3_rate_matched_v2_n128 | compute_end_to_commit | 128 | 515.808 | 2158.422 |
| ouro | confirm_ouro_primary_PD_start3_rate_matched_v2_n128 | commit_to_D_received | 128 | 19.044 | 22.538 |
| ouro | confirm_ouro_primary_PD_start3_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 536.750 | 2176.607 |
| huginn | confirm_huginn_primary_PD_start1_rate_matched_v2_n128 | P_compute_observation | 128 | 997.245 | 1081.911 |
| huginn | confirm_huginn_primary_PD_start1_rate_matched_v2_n128 | compute_end_to_commit | 128 | 12.893 | 16.466 |
| huginn | confirm_huginn_primary_PD_start1_rate_matched_v2_n128 | commit_to_D_received | 128 | 10.956 | 12.931 |
| huginn | confirm_huginn_primary_PD_start1_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 24.954 | 28.182 |
| huginn | confirm_huginn_primary_PD_start2_rate_matched_v2_n128 | P_compute_observation | 128 | 1009.461 | 1097.345 |
| huginn | confirm_huginn_primary_PD_start2_rate_matched_v2_n128 | compute_end_to_commit | 128 | 15.208 | 17.774 |
| huginn | confirm_huginn_primary_PD_start2_rate_matched_v2_n128 | commit_to_D_received | 128 | 9.690 | 11.754 |
| huginn | confirm_huginn_primary_PD_start2_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 26.034 | 29.343 |
| huginn | confirm_huginn_primary_PD_start3_rate_matched_v2_n128 | P_compute_observation | 128 | 1017.359 | 1101.190 |
| huginn | confirm_huginn_primary_PD_start3_rate_matched_v2_n128 | compute_end_to_commit | 128 | 20.435 | 23.824 |
| huginn | confirm_huginn_primary_PD_start3_rate_matched_v2_n128 | commit_to_D_received | 128 | 7.569 | 9.367 |
| huginn | confirm_huginn_primary_PD_start3_rate_matched_v2_n128 | P_compute_end_to_source_release_nonadditive | 128 | 29.147 | 33.910 |

These intervals use coordinator message receipts. They are not exclusive DMA or GPU compute measurements.
The compute-end-to-commit span can include descriptor preparation, submission, transport progress, and message delivery.
Source-release spans retain native ACK dependencies. They overlap the additive host phases.
A long source-retention span can consume a P staging slot after computation ends.
The data do not prove that removing all such spans is safe or achieves an equivalent latency reduction.
No early release or unacknowledged state reclamation is implemented.
The request timeline shows host observations. The separate kernel timeline shows actual observed device activity.

## Source-retention capacity control

Ouro 8192/32 uses two matched N24 diagnostic starts at nominal 2.5 requests/s.
Only the supported extra-draining-slot setting changes from zero to four.
Both starts use the same arrival schedule. The realized interval rate is about 3.121 requests/s.
The additional staging-capacity byte and output gate passes 8/8 requests.
The actual recurrent row count is 263,136/263,136.

| Extra draining slots | Completed/N | Completed req/s | TTFT p95 ms | Request TPOT p95 ms | Token ITL p99 ms | Central P completed req/s |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 24/24 | 1.778436 | 6069.688 | 39.595 | 42.097 | 1.938854 |
| 4 | 24/24 | 1.782023 | 6904.670 | 41.645 | 57.385 | 1.938854 |

The same payload is 154,618,920,960 bytes per start. Each start records 2,328 completed transfer fragments.
Central P completion rate does not improve in this control.
Arrival-to-drain throughput changes by about +0.20%, without independent-start uncertainty for this diagnostic.
The larger staging allowance therefore does not establish a practical improvement.
The control retains the native ACK and source-buffer lifetime rules.
It does not prove that source retention never matters under other conditions.

## Profile completion and finalization failure

The first Ouro P+D N128 profile completes every response but fails native artifact finalization.
That job remains FAILED in the matrix.
Its stack records native event-tree construction and subsequent gzip packaging after the response interval.
The harness also uses a fixed 300-second artifact wait despite a longer declared finalization allowance.
The failure log and partial artifacts remain intact.

The minimal subsequent exporter retains the complete native Chrome trace and native archive verification.
It omits only the unused derived `profiler.events()` table for CUDA-only collection.
The archive uses lossless gzip level 1. Extended CPU/shapes/stacks/memory collection retains the original path.
The harness now passes the declared finalization timeout to the native artifact wait.
Three CPU regression tests check this export path and exact archive member bytes.
The subsequent Ouro P-only N128 profile completes and passes the +3.957% wall-perturbation check.
No GPU arithmetic or collection activity changes in this export-only fix.
The earlier historical Huginn 80/128 and 14/128 stops remain causally unresolved.

See `profile_artifact_export_root_cause.md` for the operation-level evidence and its limits.

## KV credit does not imply capacity for another long request

The diagnostic credit screen asks whether blocks remain for a structural 64-token, one-output probe.
It does not establish capacity for another full 2560-token Huginn request or an actual migration.
In the qualified Huginn N24 interval, five reservations occupy 25,920 of 28,288 native blocks.
The remaining 2,368 blocks are about 5.962 GiB.
A new 2560/32 request needs 5,184 blocks, about 13.052 GiB, under the current uniform allocation layout.
The smaller hypothetical probe fits its block screen while the next long request does not.
The sampled refusal frequency is 28/43, not an exclusive E2E waiting fraction or a request failure rate.
All actual borrowing and continuation-admission claims remain NOT_MEASURED.
