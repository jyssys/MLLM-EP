# D-local prefill and cross-depth batching: final PoC summary

## 1. Decision

| Model | Incremental batching verdict | Generic D-local N32 state gates |
| --- | --- | --- |
| ouro | NO_GO_LOOP_BATCHING | full PASS, chunked PASS, paced PASS |
| huginn | NO_GO_LOOP_BATCHING | full PASS, chunked PASS, paced PASS |

The campaign did not establish a new loop-specific online serving advantage.

Native shared-core batching can reduce separate-execution cost. Same-depth batching also captures that benefit.

A stronger same-depth control can advance one legal prefill step before joining decode.

This control separates arithmetic batching gains from an avoidable depth-alignment wait.

Do not develop a new dynamic scheduler from these results alone.

The verdict applies to this bounded workload and runtime. It does not prove that every deflection policy lacks headroom.

The batching verdict concerns the incremental microbenchmark gate. Qualified online decode-safe kernel gaps remain unmeasured.

Thus, the campaign does not establish a structural impossibility result for D-local prefill.

## 2. Environment and budget

Start: `2026-11-06T08:21:36+09:00`. GPU campaign stop: `2026-11-06T12:29:59.366901+09:00`.

Elapsed through GPU closeout: 4.140 hours. The final wall-clock limit stays 14:21:36 KST.

Guarded worker jobs started: 97. Complete job receipts: 90.

Failure or correctness-block records: 19. Configured jobs without a worker start: 5.

A primitive grid is one worker job with multiple legal-state cells. It is not a matrix of online serving runs.

Clean serving uses isolated H100 pairs. Ouro uses physical GPUs 0/1. Huginn uses 4/5.

Other GPUs stay quiet during timing. All eight approved burners must return after closeout.

The serving harness uses the native engine API and real GPU processes. It is not an HTTP endpoint benchmark.

Ouro retains BF16 weights and R4. Huginn retains BF16 weights and R32.

The Huginn prelude, recurrent injection, coda, and per-depth KV semantics remain unchanged.

The campaign does not use early exit, quantization, approximation, or model training.

Publication base: `034b4ef9d2a8f0fe1a9d993370693d349cd03838`. Publication branch: `flashvep/d-local-cross-depth-batching-poc`.

Executed checkout HEAD: `f4ecea59f36f620c69c4fef09c590bf790802cd9`. Private RLT HEAD: `6b164d3844d35ab8e753df1864f0c87dc180401a`.

Preserved private diff SHA256: `4904a311358c5cf6467678d0e1c30af38b5a96064cae4c96ed680943d7e694c7`.

Ouro checkpoint revision: `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.

Huginn checkpoint revision: `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.

Execution preservation metadata: [ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/preservation.json](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/preservation.json).

Software versions: [software_inventory.json](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/software_inventory.json).

The checkout stayed on its existing branch. Process-local hooks and an alternate Git index protect unrelated work.

## 3. Workload and SLO contract

| Model | Input/output tokens | Low/medium/heavy nominal QPS | P/D slots |
| --- | --- | --- | --- |
| Ouro | 4096/32 | 1.23460875 / 3.4569045 / 6.356 | 2/8 |
| Huginn | 2048/16 | 0.37903355455 / 1.06129395274 / 1.971 | 2/8 |

The input pool is the preserved ShareGPT/LongBench controlled-prefix pool. Each N128 cohort uses 128 distinct prompt IDs.

The campaign does not add an Azure trace or claim natural response lengths. Fixed output caps use `ignore_eos=True`.

Prefix caching stays disabled. Multiple starts reuse the pool and do not establish new-text generalization.

Arrivals are open-loop Poisson. The primary arrival seed is 2026110701.

Raw records separate nominal QPS, actual send QPS, completed QPS, and send delay.

| Model | Frozen TTFT ms | Frozen request TPOT ms | Additional-work TPOT p95 screen ms | ITL p99 screen ms |
| --- | --- | --- | --- | --- |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

The statistical screen does not guarantee each token deadline. Maximum ITL and individual deadline misses remain in the raw analysis.

Finite-cohort goodput includes the full drain interval. It is not sustainable capacity.

## 4. Whole-request D-local mechanism and correctness

A selected request starts prefill on D. It never starts prefill on P.

D retains the request for decode. The local request has no intermediate P-to-D KV handoff.

The input still reaches D through the host control path. Native requests retain native NIXL transfer and ACK handling.

All-local N8 tests precede N32 qualification. N32 means simultaneous submission, not 32 simultaneous GPU residents.

The gate compares every output token, valid prompt KV plane, prompt-last hidden state, finish reason, and fixed depth.

Runtime reference parity is not original HF equivalence. Original HF equivalence remains NOT_MEASURED.

CPU contract checks: [cpu_regression_final.json](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/cpu_regression_final.json).

Exactness records: [qualification.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/qualification.csv). Work and retirement checks: [work_accounting.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/work_accounting.csv).

The early paced path failed because a partial fused prelude did not retain the complete staged hidden chunk.

The N32-qualified path uses the native zero-loop embedding stage before bounded recurrent steps.

The failed namespace remains blocked. Its timings did not enter the serving comparison.

## 5. Marker-free serving results

| Run | N | QPS | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | E2E p95 ms | Joint goodput req/s | Decode screen |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_ouro_native_n128_s2026110701_off | 128 | 3.457 | 818.443 | 35.693 | 38.243 | 1742.175 | 3.474 | True |
| serve_ouro_native_paired_full32 | 32 | 3.457 | 920.684 | 36.735 | 40.320 | 2043.067 | 3.488 | True |
| serve_ouro_full_c128_fixed25_v5 | 32 | 3.457 | 2461.551 | 44.553 | 239.729 | 3430.253 | 2.509 | False |
| serve_ouro_chunked_c128_fixed25_v5 | 128 | 3.457 | 49778.162 | 28.435 | 31.770 | 50600.924 | 0.954 | True |
| serve_ouro_paced_c128_fixed25_v5 | 128 | 3.457 | 50298.140 | 28.634 | 30.542 | 51125.669 | 0.935 | True |
| serve_ouro_paced_heavy_fixed25_clean_restart_v2 | 128 | 6.356 | 51871.473 | 29.186 | 34.235 | 52698.388 | 0.065 | True |
| local_huginn_native_n128_s2026110701_off | 128 | 1.061 | 2516.486 | 115.786 | 119.795 | 4251.447 | 0.943 | True |
| serve_huginn_native_paired_full32 | 32 | 1.061 | 3121.279 | 116.905 | 123.896 | 4769.308 | 0.721 | True |
| serve_huginn_full_c128_fixed25_v5 | 32 | 1.061 | 2983.479 | 181.362 | 791.890 | 5065.494 | 0.719 | False |
| serve_huginn_chunked_c128_fixed25_v5 | 128 | 1.061 | 93021.471 | 105.545 | 108.096 | 94423.468 | 0.355 | True |
| serve_huginn_paced_c128_fixed25_v5 | 128 | 1.061 | 98990.265 | 116.049 | 120.437 | 100397.660 | 0.362 | True |
| serve_huginn_paced_heavy_fixed25_v1 | 128 | 1.971 | 93832.866 | 106.656 | 111.383 | 95244.074 | 0.013 | True |

Compare N32 naive-full with its paired N32 native control. Compare the N128 generic arms with N128 native.

V5 uses common full-context decode-geometry warm-up and an eight-entry graph budget. V4 remains a tuning diagnostic.

The generic controls fix 25% of request IDs on D. Their local active limit is one.

That limit includes the local decode lifetime. It can serialize local requests.

Strict ready-decode priority can also starve prefill. A passing global decode screen does not prove useful concurrent borrowing.

These controls are not the official Kairos algorithm or a complete deadline-prediction policy.

Their poor latency must not become a structural rejection of every optimized generic deflection policy.

The measured generic controls did not improve the primary Native comparison. They do not establish GENERIC_ONLY success.

All latency percentiles, rates, misses, failures, and token IDs: [serving_runs.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/serving_runs.csv) and `raw/*/requests.jsonl`.

## 6. Why native PD protects ITL

Native D does not execute prompt slices. A migrated or local prefill sequence can block newly ready decode on the same execution stream.

Ready-queue priority cannot preempt work that the runtime already submitted. Average utilization cannot establish a safe insertion deadline.

The naive-full control isolates this interference without intermediate KV migration for local requests.

It does not isolate GPU blocking from host submission cost. The rejected traces cannot supply that causal split.

The serial slice measurements also show the effect of recurrent quantum. They do not prove online safe windows.

Slice distributions: [slice_costs.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/slice_costs.csv).

Historical Token/Loop paths also changed runtime, transfer, and graph costs. The new controls do not attribute every historical difference to one cause.

## 7. Shared-weight batching and the stronger same-depth control

The full-core grid uses legal ready states. It checks hidden rows and written query KV bytes.

It includes causal attention and graph input staging. Initial metadata construction and outer scheduling remain outside the primitive timer.

V5 keeps the decode batch intact and removes an unnecessary output concatenation from the timed separate control.

Older state checks remain valid. Older separate timings are not primary.

At identical input, weight pointer, row count, and graph, a dense depth label does not change arithmetic.

Actual cross-depth value must therefore come from ready-work availability or scheduling, not the label itself.

The larger-chunk control includes C128 and C248. Each method gets the same graph and token budgets.

The stronger same-depth control executes at most one legal prefill step to remove a depth-distance-one alignment wait.

That step can delay ready decode. Its measured TPOT and ITL remain visible. It is not strict priority.

| Model | Chunk | Physical GPU | Method | Local first token ms | Cohort ms | BG TPOT p95 ms | BG ITL p99 ms | Output exact | Mixed-phase different-depth submissions | Local prefill inside BG lifetime |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | 128 | 0 | separate | 1686.543 | 2557.960 | 29.514 | 30.916 | True | 0 | 0/3 |
| ouro | 128 | 0 | same | 1096.115 | 1966.975 | 33.563 | 34.020 | True | 0 | 0/3 |
| ouro | 128 | 0 | same_align | 1077.106 | 1948.325 | 33.825 | 38.674 | True | 0 | 0/3 |
| ouro | 128 | 0 | cross | 1077.638 | 1948.737 | 33.652 | 33.898 | True | 123 | 0/3 |
| ouro | 248 | 0 | separate | 1353.237 | 1968.476 | 29.566 | 30.875 | True | 0 | 0/3 |
| ouro | 248 | 0 | same | 650.476 | 1621.298 | 33.448 | 36.835 | True | 0 | 3/3 |
| ouro | 248 | 0 | same_align | 625.108 | 1626.912 | 33.585 | 41.514 | True | 0 | 3/3 |
| ouro | 248 | 0 | cross | 628.350 | 1623.394 | 33.459 | 36.804 | True | 68 | 3/3 |
| ouro | 248 | 1 | separate | 1363.183 | 1980.911 | 29.832 | 31.162 | True | 0 | 0/3 |
| ouro | 248 | 1 | same | 653.137 | 1632.207 | 33.673 | 36.976 | True | 0 | 3/3 |
| ouro | 248 | 1 | same_align | 627.327 | 1636.086 | 33.797 | 41.743 | True | 0 | 3/3 |
| ouro | 248 | 1 | cross | 631.179 | 1632.973 | 33.699 | 36.921 | True | 68 | 3/3 |
| huginn | 128 | 4 | separate | 2705.844 | 4333.723 | 94.171 | 98.499 | True | 0 | 0/3 |
| huginn | 128 | 4 | same | 2109.376 | 3738.785 | 129.182 | 136.279 | True | 0 | 0/3 |
| huginn | 128 | 4 | same_align | 2069.812 | 3699.063 | 132.131 | 136.699 | True | 0 | 0/3 |
| huginn | 128 | 4 | cross | 2070.940 | 3699.919 | 131.682 | 136.814 | True | 479 | 0/3 |
| huginn | 248 | 4 | separate | 2257.873 | 3725.972 | 94.372 | 99.954 | True | 0 | 0/3 |
| huginn | 248 | 4 | same | 1471.690 | 3353.006 | 130.365 | 162.375 | True | 0 | 3/3 |
| huginn | 248 | 4 | same_align | 1379.397 | 3323.412 | 130.162 | 163.092 | True | 0 | 3/3 |
| huginn | 248 | 4 | cross | 1383.800 | 3368.357 | 130.298 | 162.254 | True | 288 | 3/3 |
| huginn | 248 | 5 | separate | 2252.004 | 3715.264 | 93.919 | 99.520 | True | 0 | 0/3 |
| huginn | 248 | 5 | same | 1468.651 | 3347.664 | 129.665 | 161.933 | True | 0 | 3/3 |
| huginn | 248 | 5 | same_align | 1375.161 | 3313.577 | 129.639 | 161.910 | True | 0 | 3/3 |
| huginn | 248 | 5 | cross | 1379.972 | 3358.123 | 129.882 | 161.511 | True | 288 | 3/3 |

These are standalone closed-loop controls with two background decode streams and one local prompt. They are not online PD goodput.

The local micro request produces two tokens. Background caps stay 32 for Ouro and 16 for Huginn.

The separate ready-state primitive grid caps its background streams at eight tokens. Those cells are not primary serving workloads.

The initial micro device is physical GPU 0 or 4. The D-device confirmation uses physical GPU 1 or 5.

The lifetime column counts whole local-prefill intervals inside existing background token-to-finish intervals. It is not a GPU idle-time measurement.

Independent C32 starts have separate receipts. Their paired confidence intervals are preliminary and do not establish online capacity.

Primitive data: [matched_core_micro.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/matched_core_micro.csv). Dense controls: [dense_micro.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/dense_micro.csv).

Readiness controls: [readiness_opportunity.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/readiness_opportunity.csv). Paired-start intervals: [phase_c_gate.json](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/phase_c_gate.json).

The qualified numerical patch retains 256-row tiles for sensitive projections. Ordinary batching can amortize that extra work.

This runtime effect is not a new loop-specific contribution. Actual weight-read bytes remain NOT_MEASURED.

| Model | Depth compatibility | Separate core ms | Batched core ms | Batch saving % | Eager fallbacks |
| --- | --- | --- | --- | --- | --- |
| ouro | same | 11.705 | 6.599 | 43.618 | 0 |
| ouro | cross | 11.705 | 6.594 | 43.662 | 0 |
| huginn | same | 4.392 | 2.348 | 46.544 | 0 |
| huginn | cross | 4.392 | 2.348 | 46.532 | 0 |

Each row compares the same ready work with separate or batched execution. Same-depth and cross-depth rows can have different attention metadata.

The matched dense-label control instead keeps the tensor and graph identical. It isolates the depth label from attention geometry.

| Model | Diagnostic method | Kernels | Summed recorded kernel ms | Named attention ms | Named dense ms |
| --- | --- | --- | --- | --- | --- |
| ouro | batched | 1411 | 6.020 | 1.590 | 0.000 |
| ouro | separate | 3012 | 10.686 | 1.763 | 0.000 |
| huginn | batched | 240 | 2.171 | 0.201 | 0.000 |
| huginn | separate | 510 | 4.112 | 0.223 | 0.000 |

These profiler diagnostics support kernel attribution only. Their times do not establish clean online speedup or exclusive device occupancy.

Profile scopes: [micro_kernel_diagnostics.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/micro_kernel_diagnostics.csv).

## 8. Slack evidence and instrumentation limits

Collector coverage and low-overhead serving are separate gates. Zero dropped kernels do not establish acceptable overhead.

The current kernel-activity traces failed the matched local-tail screen. They remain diagnostic, not clean slack proof.

CUDA-event model intervals do not replace exclusive kernel activity. SM cycles, achieved bandwidth, and qualified clean kernel gaps remain NOT_MEASURED.

First-entry markers provide diagnostic Service TTFT. Marker-free runs do not contain a measured first-model timestamp.

Their Service TTFT and queue decomposition therefore remain NOT_MEASURED. Do not fabricate those components from host admission.

Per-request marked decompositions retain all later handoff and continuation costs. Component p95 values must not be added.

Interference records: [instrumentation_qualification.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/instrumentation_qualification.csv).

First-entry marker checks: [service_marker_qualification.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/service_marker_qualification.csv).

Sampled retirement lower bounds: [sampled_retired_overlap.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/sampled_retired_overlap.csv).

A zero sampled overlap bound does not prove that no overlap occurred. Sparse snapshots can miss retired boundaries.

Do not combine a separate P saturation run with a standalone D microbenchmark into a same-run success claim.

| Marked native run | TTFT p95 ms | Pre-model latency p95 ms | Service TTFT p95 ms | Eligibility |
| --- | --- | --- | --- | --- |
| local_huginn_native_service_once128 | 2485.708 | 1759.158 | 861.549 | DIAGNOSTIC_ONLY |
| local_huginn_native_service_pool128 | 2545.248 | 1757.004 | 956.864 | DIAGNOSTIC_ONLY |
| local_ouro_native_service_once128 | 746.835 | 538.558 | 242.741 | DIAGNOSTIC_ONLY |
| local_ouro_native_service_pool128 | 754.730 | 541.610 | 254.139 | DIAGNOSTIC_ONLY |

These components come from per-request timestamps. Their p95 values do not add to TTFT p95.

## 9. KV, HBM, and work accounting

Each D-local request retains full fixed-depth KV. Removing intermediate KV handoff does not remove local KV memory.

Huginn retains its physical dead-plane cost. The campaign does not apply ExactLoop-style KV compaction.

Coordinator reservation credit, allocated KV, compute readiness, and active execution are separate quantities.

Memory tables use GiB and a stated HBM denominator. Unique-storage components overlap allocator totals and must not be counted twice.

Opaque graph and driver allocations are not fully attributed. NVML reports sampled memory peaks, not continuous exact peaks.

Memory: [gpu_memory_inventory.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/gpu_memory_inventory.csv). Credits and host readiness: [system_summary.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/system_summary.csv).

Real recurrent token rows are not FLOPs. They exclude prelude, coda, and numerical-padding work.

Local completion counts must not include old independent 16-token probes or migrated P requests.

The runtime records transfer byte counters. Summed transport-observation durations are not exclusive communication time.

| Model | Role | Weights GiB / HBM % | KV pool GiB / HBM % | Workspace GiB / HBM % | NIXL staging GiB / HBM % | Allocator GiB / HBM % |
| --- | --- | --- | --- | --- | --- | --- |
| ouro | prefill | 2.672 / 3.355% | 48.000 / 60.266% | 1.254 / 1.574% | 0.000 / 0.000% | 52.082 / 65.391% |
| ouro | decode | 2.672 / 3.355% | 48.000 / 60.266% | 0.627 / 0.787% | 0.000 / 0.000% | 51.362 / 64.487% |
| huginn | prefill | 6.642 / 8.339% | 41.250 / 51.791% | 0.342 / 0.429% | 0.000 / 0.000% | 48.549 / 60.955% |
| huginn | decode | 6.642 / 8.339% | 67.354 / 84.565% | 0.169 / 0.213% | 0.001 / 0.001% | 74.278 / 93.258% |

These percentages use NVML physical HBM. CUDA allocatable capacity is a separate field in the memory snapshots.

| Model | Selected local | Retired local | Quota blocks/request | Quota GiB/request | Quota / HBM % | Pool-only quotient | Fixed rows exact | Final device/KV empty |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | 32 | 32 | 1032 | 3.023 | 3.796 | 15 | True | True |
| huginn | 32 | 32 | 4128 | 10.393 | 13.049 | 6 | True | True |

The pool-only quotient is structural accounting from measured block geometry. It ignores native reservations and compute-slot limits.

A quota is not measured live attention bytes. Unused physical KV planes remain allocated.

## 10. Prior work and novelty boundary

[Load-Aware Prefill Deflection](https://arxiv.org/abs/2607.02043) already studies whole-request D placement and decode-aware prefill chunks. The fixed controls here are not its official system.

[FluidPD](https://arxiv.org/abs/2610.06917) uses pressure-aware mechanisms, including prefix/suffix work transfer. This PoC does not run FluidRole or FluidToken.

[CDB](https://arxiv.org/abs/2608.09444) already uses tied-core work at different recurrent depths. Native RLT also exposes this capability.

[ExactLoop](https://arxiv.org/abs/2611.05150) studies exactness-graded runtime optimizations. Their symmetric application and remaining headroom are not measured here.

The reference LoopServe code uses a different vLLM V1 ABI. Its state structures were not substituted into RLT.

Combining deflection and existing CDB does not, by itself, establish a new method.

| Prior work | Relevant mechanism | Use in this PoC | Unverified extension |
| --- | --- | --- | --- |
| Load-Aware Prefill Deflection | Whole-request D placement and decode-aware chunks | Placement control and priority baseline | Official policy and calibrated deadline prediction |
| FluidPD / FluidToken | Pressure-aware prefix/suffix transfer | Comparison boundary only | Official FluidToken execution |
| CDB | Shared-core work at different recurrent depths | Native RLT depth mixing and matched controls | New contribution beyond existing depth scheduling |
| ExactLoop | Exactness-graded runtime optimization | Exactness boundary and source review | Symmetric optimization ablation |

Detailed audit: [source_and_prior_work_audit.md](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/source_and_prior_work_audit.md).

## 11. Failures, missing tests, and restart

Warm-up ownership, post-admission coda readiness, and partial-prelude state retention required separate corrections. All failed logs remain.

A daemon restart removed dispatch controllers. One worker completed its cohort without its dispatcher receipt.

Its recovered receipt has an unknown process exit code and an unqualified timing label. The raw cohort is not a clean primary comparison.

A fresh named heavy control avoids reusing that interrupted timing as primary. The original result remains preserved.

Several completed jobs log UCX CUDA-context error 4 during teardown. Their process exits, output checks, work counts, and saved release checks pass.

The teardown error cause remains INCONCLUSIVE. Completion checks do not establish that the warning is harmless.

The campaign omitted optional Huginn C32 N32 cells to prioritize mandatory C128 N32 and warm-matched serving. Their N8 tests remain.

Cross-depth online N128, N512 persistence, natural-EOS variability, and the ExactLoop ablation did not pass the expansion gate.

The campaign does not claim a complete calibrated generic deflection controller or a production scheduler.

Every execution and missing configuration: [configuration_matrix.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/configuration_matrix.csv). Failures: [failure_inventory.csv](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/failure_inventory.csv).

Full measurement limits: [measurement_contract_and_limitations.md](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/measurement_contract_and_limitations.md).

## 12. Figures and reproduction

- [figure_1_decode_safe_windows](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_1_decode_safe_windows.png). Qualified activity-only observations. No simulated idle intervals.

- [figure_2_d_local_ownership_and_hbm](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_2_d_local_ownership_and_hbm.png). Measured storage inventory and N32 exactness. No added KV pool.

- [figure_3_same_vs_cross_depth_micro](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_3_same_vs_cross_depth_micro.png). Measured legal ready-state operations. Timings include graph input staging. Not online serving.

- [figure_4_serving_latency_and_goodput](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_4_serving_latency_and_goodput.png). Finite-cohort measurements. Missing arms are not zero.

- [figure_5_ready_work_opportunity_control](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_5_ready_work_opportunity_control.png). Native priorities are common. Only recurrent compatibility changes. No modeled speedup.

- [figure_6_slice_execution_quantum](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_6_slice_execution_quantum.png). CUDA-event enclosures, not exclusive kernel time or an online speedup.

- [figure_7_native_load_controls](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_7_native_load_controls.png). Marker-free Native PD controls. No invented generic or cross-depth curve.

- [figure_8_unqualified_diagnostic_timeline](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_8_unqualified_diagnostic_timeline.png). Measured under rejected instrumentation. Never use this figure as clean kernel activity or safe slack proof.

- [figure_9_stronger_same_depth_control](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_9_stronger_same_depth_control.png). Measured larger-chunk micro controls. Same align executes one real prefill step and reports its decode interference.

- [figure_10_fixed_placement_backlog](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/figures/figure_10_fixed_placement_backlog.png). Marker-free per-request latency and sampled coordinator credit. No queue-to-kernel attribution.


Use the qualified private runtime and the pinned checkpoints. Existing names return saved receipts, not fresh experiments.

Use a fresh configuration name for a real rerun. Do not overwrite a prior namespace.

```bash
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.run NEW_RUN_NAME
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.analyze
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.micro_analysis
/home/esjung/.venvs/looped-pd-abc/bin/python -m poc_flashvep.d_local_cross_depth_batching.figures
```

## 13. Publication and resource restoration

Branch: `flashvep/d-local-cross-depth-batching-poc`. The working checkout and real staged index remain untouched.

The committed [publication receipt](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/publication_receipt.json) records the evidence commit and verified remote SHA.

The local [final remote check](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/final_remote_verification.json) records the final metadata commit.

Burner restoration and actual GPU-process checks: [resource restoration](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/final_resource_restoration.json).

The first GPU 1 burner startup met a TCPStore port collision. The retry leaves the other seven burner sessions running.

Resource recovery evidence: [burner_startup_retry.json](ouro_huginn_loop_aware_d_local_prefill_cross_phase_batching/state/burner_startup_retry.json).
