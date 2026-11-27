# Native H200 operator resource profiling

CUPTI GPU activities exported by PyTorch Kineto are aligned to the client monotonic clock as `baseTimeNanoseconds × 10⁻⁹ + kernel.ts × 10⁻⁶ − clock_epoch_minus_mono`. Chrome event durations use microseconds. Kernel-only CUPTI data instead use the affine map from worker start and end CUPTI readings to their monotonic-clock bracket midpoints. The inner bounds of the brackets must cover the ROI for each selected role, with matching export counts and zero dropped or invalid records. Kineto coverage is checked against the selected profiler markers. Both roles are required to measure temporal intersection. A profile earns `TRACE_ANALYSIS` only when matched clean-before/profile/clean-after cohorts complete with equal token sequences and the absolute change is at most 5% on every pair for TTFT p95, TPOT p95, ITL p99, and finite request throughput. Other traces are `DIAGNOSTIC`. The three runs share workers and are an observer control, not independent server starts.

| Profile | Collector | Traced roles | Trace status | Observer gate | Largest paired change % | Full selected-role coverage | Record check |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| dense-controls/p-light/16k_light_profile | cupti_kernel_only | P0, D0 | TRACE_ANALYSIS | PASS | 2.87 | True | True |
| native-p-only/online/16k_p_only_profile | cupti_kernel_only | P0 | DIAGNOSTIC | FAIL | 26.20 | True | True |
| native-pd/online/16k_trace_profile | kineto | P0, D0 | DIAGNOSTIC | FAIL | 68.18 | True | True |
| native-steady/online/16k_steady_profile | cupti_kernel_only | P0, D0 | DIAGNOSTIC | FAIL | 50.36 | True | True |

16k_light TPOT p95 shifted +1.91% versus clean-before and +0.81% versus clean-after. Clean-after versus clean-before shifted +1.09%. Matched emitted token sequences were identical.

16k_p_only TPOT p95 shifted +5.81% versus clean-before and +19.98% versus clean-after. Clean-after versus clean-before shifted -11.81%. Matched emitted token sequences were identical.

16k_trace TPOT p95 shifted +68.18% versus clean-before and +42.77% versus clean-after. Clean-after versus clean-before shifted +17.80%. Matched emitted token sequences were identical.

16k_steady TPOT p95 shifted +17.41% versus clean-before and +21.65% versus clean-after. Clean-after versus clean-before shifted -3.48%. Matched emitted token sequences were identical.

## Operator families on the common clock

Kernel name heuristics separate Attention, GEMM, explicit normalization, activation and cache operations. The linear family includes cuBLAS and NVJet names. A generic GEMM kernel name does not identify the QKV, output projection or FFN call site. The same warning applies to fused operators and GPU graphs. Each value below is a within-family interval union. Different family intervals may coincide. Neither their sum nor the sum of kernel durations is a serving critical path.

| Profile | Role | GPU family | Union ms | Event-sum ms | Share of role's active kernel intervals | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 16k_light | D0 | Attention | 197.043 | 200.832 | 17.97% | TRACE_ANALYSIS |
| 16k_light | D0 | FFN activation | 10.067 | 10.067 | 0.92% | TRACE_ANALYSIS |
| 16k_light | D0 | KV cache write | 15.046 | 15.046 | 1.37% | TRACE_ANALYSIS |
| 16k_light | D0 | Linear / GEMM | 816.221 | 816.427 | 74.42% | TRACE_ANALYSIS |
| 16k_light | D0 | Norm | 29.792 | 29.792 | 2.72% | TRACE_ANALYSIS |
| 16k_light | D0 | Other / unresolved | 29.135 | 29.521 | 2.66% | TRACE_ANALYSIS |
| 16k_light | D0 | Sampling | 0.815 | 0.815 | 0.07% | TRACE_ANALYSIS |
| 16k_light | P0 | Attention | 1826.232 | 1826.232 | 28.94% | TRACE_ANALYSIS |
| 16k_light | P0 | FFN activation | 130.400 | 130.400 | 2.07% | TRACE_ANALYSIS |
| 16k_light | P0 | KV cache write | 48.446 | 48.446 | 0.77% | TRACE_ANALYSIS |
| 16k_light | P0 | Linear / GEMM | 4125.450 | 4125.450 | 65.38% | TRACE_ANALYSIS |
| 16k_light | P0 | Norm | 116.964 | 116.964 | 1.85% | TRACE_ANALYSIS |
| 16k_light | P0 | Other / unresolved | 62.050 | 62.050 | 0.98% | TRACE_ANALYSIS |
| 16k_light | P0 | Sampling | 0.116 | 0.116 | 0.00% | TRACE_ANALYSIS |
| 16k_p_only | P0 | Attention | 5344.552 | 5344.552 | 29.06% | DIAGNOSTIC |
| 16k_p_only | P0 | FFN activation | 377.689 | 377.689 | 2.05% | DIAGNOSTIC |
| 16k_p_only | P0 | KV cache write | 140.737 | 140.737 | 0.77% | DIAGNOSTIC |
| 16k_p_only | P0 | Linear / GEMM | 12008.662 | 12008.662 | 65.30% | DIAGNOSTIC |
| 16k_p_only | P0 | Norm | 338.339 | 338.339 | 1.84% | DIAGNOSTIC |
| 16k_p_only | P0 | Other / unresolved | 179.362 | 179.362 | 0.98% | DIAGNOSTIC |
| 16k_p_only | P0 | Sampling | 0.332 | 0.332 | 0.00% | DIAGNOSTIC |
| 16k_trace | D0 | Attention | 193.593 | 196.950 | 17.87% | DIAGNOSTIC |
| 16k_trace | D0 | FFN activation | 9.906 | 9.906 | 0.91% | DIAGNOSTIC |
| 16k_trace | D0 | KV cache write | 14.260 | 14.260 | 1.32% | DIAGNOSTIC |
| 16k_trace | D0 | Linear / GEMM | 809.256 | 809.480 | 74.71% | DIAGNOSTIC |
| 16k_trace | D0 | Norm | 29.448 | 29.448 | 2.72% | DIAGNOSTIC |
| 16k_trace | D0 | Other / unresolved | 27.273 | 27.598 | 2.52% | DIAGNOSTIC |
| 16k_trace | D0 | Sampling | 0.744 | 0.744 | 0.07% | DIAGNOSTIC |
| 16k_trace | P0 | Attention | 1830.172 | 1830.172 | 29.01% | DIAGNOSTIC |
| 16k_trace | P0 | FFN activation | 130.107 | 130.107 | 2.06% | DIAGNOSTIC |
| 16k_trace | P0 | KV cache write | 48.288 | 48.288 | 0.77% | DIAGNOSTIC |
| 16k_trace | P0 | Linear / GEMM | 4122.386 | 4122.386 | 65.34% | DIAGNOSTIC |
| 16k_trace | P0 | Norm | 116.583 | 116.583 | 1.85% | DIAGNOSTIC |
| 16k_trace | P0 | Other / unresolved | 61.773 | 61.773 | 0.98% | DIAGNOSTIC |
| 16k_trace | P0 | Sampling | 0.117 | 0.117 | 0.00% | DIAGNOSTIC |
| 16k_steady | D0 | Attention | 582.189 | 593.289 | 17.97% | DIAGNOSTIC |
| 16k_steady | D0 | FFN activation | 29.702 | 29.702 | 0.92% | DIAGNOSTIC |
| 16k_steady | D0 | KV cache write | 44.447 | 44.447 | 1.37% | DIAGNOSTIC |
| 16k_steady | D0 | Linear / GEMM | 2411.662 | 2412.292 | 74.42% | DIAGNOSTIC |
| 16k_steady | D0 | Norm | 88.025 | 88.025 | 2.72% | DIAGNOSTIC |
| 16k_steady | D0 | Other / unresolved | 86.267 | 87.393 | 2.66% | DIAGNOSTIC |
| 16k_steady | D0 | Sampling | 2.400 | 2.400 | 0.07% | DIAGNOSTIC |
| 16k_steady | P0 | Attention | 5356.120 | 5356.120 | 29.09% | DIAGNOSTIC |
| 16k_steady | P0 | FFN activation | 377.659 | 377.659 | 2.05% | DIAGNOSTIC |
| 16k_steady | P0 | KV cache write | 140.811 | 140.811 | 0.76% | DIAGNOSTIC |
| 16k_steady | P0 | Linear / GEMM | 12016.159 | 12016.159 | 65.27% | DIAGNOSTIC |
| 16k_steady | P0 | Norm | 339.029 | 339.029 | 1.84% | DIAGNOSTIC |
| 16k_steady | P0 | Other / unresolved | 179.318 | 179.318 | 0.97% | DIAGNOSTIC |
| 16k_steady | P0 | Sampling | 0.334 | 0.334 | 0.00% | DIAGNOSTIC |

## Actual engine iteration geometry

A CPU `execute_context_…_generation_…` annotation counts scheduled context requests and token rows, plus generation requests and rows. Counts include annotation starts within the common ROI. The reducer uses CPU annotations once per iteration and excludes duplicate GPU annotations. Rows with no scheduled tokens include waiting or communication progress. These counts supply online shapes for operator controls. A trace with a failed observer gate establishes diagnostic shapes only.

| Profile | Role | Context requests / rows | Generation requests / rows | Iterations | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| 16k_trace | D0 | 0 / 0 | 0 / 0 | 205 | DIAGNOSTIC |
| 16k_trace | D0 | 0 / 0 | 1 / 1 | 180 | DIAGNOSTIC |
| 16k_trace | D0 | 1 / 1 | 0 / 0 | 12 | DIAGNOSTIC |
| 16k_trace | P0 | 0 / 0 | 0 / 0 | 377 | DIAGNOSTIC |
| 16k_trace | P0 | 1 / 8192 | 0 / 0 | 26 | DIAGNOSTIC |

Qwen3-8B TP1 has 32 query heads, eight KV heads, head dimension 128, hidden width 4,096 and FFN width 12,288. For an annotated group of M token rows, the logical GEMM geometry before padding is QKV 6,144 × 4,096, attention output 4,096 × 4,096, fused gate/up 24,576 × 4,096, and down projection 4,096 × 12,288 in N × K order. These dimensions come from the checkpoint and runtime model definition. The role-specific M values above come from the actual trace. [Pinned checkpoint geometry](https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/config.json) [vLLM Qwen3 implementation](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)

## Strided scheduler geometry observer

The worker samples every fourth dispatch and records each request's scheduled query rows, computed prefix positions plus query rows as its logical KV length, and context/generation flags. The raw arrays preserve sampled ragged shapes. Weighting by samples leaves unsampled iterations outside the population claim. This CPU observer uses the same clean-before/observer/clean-after 5% and token-parity contract.

| Cohort | Role | ROI samples | Batch size range | Scheduled-query total p50 / p95 | Context-only / generation-only / mixed | Logical KV rows per request range | Observer max change % | Evidence |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense-controls/p-light/8k_near_shape | P0 | 12 | 1–1 | 8192 / 8192 | 12 / 0 / 0 | 8192–8192 | 19.38 | DIAGNOSTIC |
| dense-controls/p-light/8k_near_shape | D0 | 178 | 1–1 | 1 / 1 | 12 / 166 / 0 | 8192–8207 | 19.38 | DIAGNOSTIC |

[Per-worker logical shape distributions](../pd_resource_complementary_parallelism/results/native/native_scheduler_shapes.json)

## D engine token accounting in clean cohorts

The Prometheus `iteration_tokens_total` histogram accumulates computed prompt and generation tokens in engine output accounting. The ≤1 bucket can contain zero-token updates. This evidence helps locate small decode geometry and leaves actual CUDA launch batch size to the worker observers and trace annotations.

| Cohort | D iterations in histogram | Accounted tokens | Iterations reporting ≤1 token |
| --- | ---: | ---: | ---: |
| dense-controls/p-light/16k_near | 1024 | 1024 | 1024 |
| dense-controls/p-light/16k_over | 1024 | 1024 | 1024 |
| dense-controls/p-light/8k_near_clean_before | 1024 | 1024 | 1024 |
| native-p-only/online/16k_p_only_clean_before | 1024 | 1024 | 1024 |
| native-pd/online/16k_near | 1024 | 1024 | 1024 |
| native-pd/online/16k_over | 1024 | 1024 | 1024 |
| native-pd/online/8k_near | 1024 | 1024 | 1024 |
| native-pd/online/8k_over | 1022 | 1024 | 1020 |
| native-static-3gpu/p1_d2/16k_near | 1024 | 1024 | 1024 |
| native-static-3gpu/p1_d2/16k_over | 1024 | 1024 | 1024 |
| native-static-3gpu/p1_d2/8k_near | 1024 | 1024 | 1024 |
| native-static-3gpu/p2_d1/16k_near | 794 | 1024 | 564 |
| native-static-3gpu/p2_d1/16k_over | 735 | 1024 | 446 |
| native-static-3gpu/p2_d1/8k_near | 749 | 1024 | 488 |
| native-static-3gpu/p_tp2_d1/16k_near | 685 | 1024 | 376 |
| native-static-3gpu/p_tp2_d1/16k_over | 608 | 1024 | 249 |
| native-static-3gpu/p_tp2_d1/8k_near | 427 | 1024 | 157 |
| native-steady/online/16k_near | 1024 | 1024 | 1024 |
| native-steady/online/16k_steady_clean_before | 1024 | 1024 | 1024 |
| native-steady/online/mixed_heavy | 1826 | 2048 | 1604 |

## Same-window NVML GPM device samples

These hardware samples are interval averages over roughly 250 ms. Means are weighted by central-ROI exposure. They do not resolve individual operator or decode deadlines. Device RX and TX are aggregate rates across NVLinks, which can account for the same transfer at both endpoints. [NVML GPM metric definitions](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlGpmEnums.html)

| Cohort | Role | SM active % | SM occupancy % | Tensor active % | DRAM bandwidth % | NVLink RX / TX MiB/s | Peak sampled VRAM GiB | SM sample coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense-controls/dp2/16k_near | R0 | 54.16 | 10.49 | 42.21 | 13.42 | 0.00 / 0.00 | 66.40 | 100.00% |
| dense-controls/dp2/16k_near | R1 | 45.20 | 8.75 | 35.19 | 11.18 | 0.00 / 0.00 | 66.43 | 100.00% |
| dense-controls/dp2/16k_over | R0 | 71.95 | 13.99 | 57.19 | 16.53 | 0.00 / 0.00 | 66.40 | 100.00% |
| dense-controls/dp2/16k_over | R1 | 64.14 | 12.42 | 50.39 | 15.44 | 0.00 / 0.00 | 66.43 | 100.00% |
| dense-controls/dp2/32k_near | R0 | 50.91 | 9.86 | 40.59 | 8.96 | 0.00 / 0.00 | 66.40 | 100.00% |
| dense-controls/dp2/32k_near | R1 | 28.73 | 5.55 | 22.66 | 5.44 | 0.00 / 0.00 | 66.43 | 100.00% |
| dense-controls/dp2/8k_near | R0 | 60.17 | 11.50 | 43.53 | 20.13 | 0.00 / 0.00 | 66.40 | 100.00% |
| dense-controls/dp2/8k_near | R1 | 51.11 | 9.79 | 37.42 | 16.80 | 0.00 / 0.00 | 66.43 | 100.00% |
| dense-controls/dp2-chunk2048/16k_near | R0 | 52.51 | 10.05 | 40.37 | 14.01 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-chunk2048/16k_near | R1 | 43.65 | 8.34 | 33.42 | 11.84 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-chunk2048/16k_over | R0 | 70.23 | 13.55 | 55.69 | 17.19 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-chunk2048/16k_over | R1 | 63.67 | 12.26 | 50.15 | 16.05 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-chunk2048/8k_near | R0 | 60.08 | 11.37 | 43.50 | 20.70 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-chunk2048/8k_near | R1 | 48.96 | 9.21 | 34.66 | 17.70 | 0.00 / 0.00 | 65.55 | 100.00% |
| dense-controls/dp2-no-chunk/16k_near | R0 | 54.47 | 10.60 | 42.44 | 13.04 | 0.00 / 0.00 | 70.38 | 100.00% |
| dense-controls/dp2-no-chunk/16k_near | R1 | 44.42 | 8.61 | 34.41 | 10.89 | 0.00 / 0.00 | 68.12 | 100.00% |
| dense-controls/dp2-no-chunk/16k_over | R0 | 72.23 | 14.12 | 57.18 | 16.41 | 0.00 / 0.00 | 70.38 | 100.00% |
| dense-controls/dp2-no-chunk/16k_over | R1 | 63.70 | 12.41 | 50.45 | 14.44 | 0.00 / 0.00 | 70.38 | 100.00% |
| dense-controls/dp2-no-chunk/8k_near | R0 | 60.77 | 11.68 | 44.18 | 20.05 | 0.00 / 0.00 | 68.12 | 100.00% |
| dense-controls/dp2-no-chunk/8k_near | R1 | 50.05 | 9.60 | 36.35 | 17.10 | 0.00 / 0.00 | 68.12 | 100.00% |
| dense-controls/p-light/16k_light_clean_after | P0 | 79.62 | 15.77 | 67.03 | 14.10 | 0.00 / 3543.19 | 66.46 | 100.00% |
| dense-controls/p-light/16k_light_clean_after | D0 | 9.42 | 1.29 | 0.67 | 9.31 | 3618.46 / 0.00 | 65.50 | 100.00% |
| dense-controls/p-light/16k_light_clean_before | P0 | 79.07 | 15.67 | 66.57 | 14.13 | 0.00 / 3605.07 | 66.45 | 100.00% |
| dense-controls/p-light/16k_light_clean_before | D0 | 9.50 | 1.30 | 0.67 | 9.37 | 3747.56 / 0.00 | 65.49 | 100.00% |
| dense-controls/p-light/16k_near | P0 | 86.79 | 17.17 | 73.15 | 15.19 | 0.00 / 4084.26 | 66.46 | 100.00% |
| dense-controls/p-light/16k_near | D0 | 10.47 | 1.44 | 0.74 | 10.33 | 4029.57 / 0.00 | 65.50 | 100.00% |
| dense-controls/p-light/16k_over | P0 | 90.73 | 17.96 | 76.46 | 15.99 | 0.00 / 4345.69 | 66.46 | 100.00% |
| dense-controls/p-light/16k_over | D0 | 10.79 | 1.48 | 0.77 | 10.65 | 4300.06 / 0.00 | 65.50 | 100.00% |
| dense-controls/p-light/8k_near_clean_after | P0 | 87.53 | 17.46 | 74.77 | 17.74 | 0.00 / 4699.45 | 66.46 | 100.00% |
| dense-controls/p-light/8k_near_clean_after | D0 | 22.66 | 3.13 | 1.24 | 22.05 | 4835.35 / 0.00 | 65.50 | 100.00% |
| dense-controls/p-light/8k_near_clean_before | P0 | 87.98 | 17.54 | 75.17 | 17.81 | 0.00 / 4769.06 | 66.46 | 100.00% |
| dense-controls/p-light/8k_near_clean_before | D0 | 22.08 | 3.05 | 1.21 | 21.50 | 4815.50 / 0.00 | 65.50 | 100.00% |
| dense-controls/pd-cross/16k_near | P0 | 86.42 | 17.11 | 72.82 | 15.25 | 0.00 / 4187.75 | 66.45 | 100.00% |
| dense-controls/pd-cross/16k_near | D0 | 10.55 | 1.45 | 0.75 | 10.41 | 4161.86 / 0.00 | 65.49 | 100.00% |
| dense-controls/pd-cross/16k_over | P0 | 90.78 | 17.96 | 76.51 | 15.87 | 0.00 / 4320.33 | 66.45 | 100.00% |
| dense-controls/pd-cross/16k_over | D0 | 10.60 | 1.46 | 0.75 | 10.48 | 4268.90 / 0.00 | 65.49 | 100.00% |
| dense-controls/pd-cross/8k_near | P0 | 87.05 | 17.36 | 74.37 | 17.92 | 0.00 / 4856.43 | 66.45 | 100.00% |
| dense-controls/pd-cross/8k_near | D0 | 22.88 | 3.16 | 1.26 | 22.31 | 4752.70 / 0.00 | 65.49 | 100.00% |
| dense-controls/pd-deflect/16k_near | P0 | 77.92 | 15.44 | 65.62 | 13.98 | 0.00 / 3726.90 | 66.45 | 100.00% |
| dense-controls/pd-deflect/16k_near | D0 | 24.65 | 4.29 | 13.12 | 12.32 | 3736.88 / 0.00 | 66.46 | 100.00% |
| dense-controls/pd-deflect/16k_over | P0 | 90.85 | 17.96 | 76.53 | 16.02 | 0.00 / 4237.08 | 66.45 | 100.00% |
| dense-controls/pd-deflect/16k_over | D0 | 37.24 | 6.64 | 21.67 | 16.86 | 4192.93 / 0.00 | 66.46 | 100.00% |
| dense-controls/pd-deflect/8k_near | P0 | 83.76 | 16.71 | 71.51 | 17.26 | 0.00 / 4654.45 | 66.45 | 100.00% |
| dense-controls/pd-deflect/8k_near | D0 | 32.59 | 5.12 | 9.63 | 23.92 | 4753.79 / 0.00 | 66.46 | 100.00% |
| dense-controls/pd-layout-probe/8k_low | D0 | 7.92 | 1.09 | 0.44 | 7.76 | 1736.37 / 0.00 | 65.52 | 100.00% |
| dense-controls/pd-layout-probe/8k_low | P0 | 27.83 | 5.61 | 23.68 | 6.55 | 0.00 / 1675.80 | 66.48 | 100.00% |
| native-p-only/online/16k_p_only_clean_after | P0 | 90.10 | 17.82 | 75.90 | 15.94 | 0.00 / 4121.66 | 66.46 | 100.00% |
| native-p-only/online/16k_p_only_clean_after | D0 | 10.86 | 1.49 | 0.77 | 10.72 | 4230.53 / 0.00 | 65.49 | 100.00% |
| native-p-only/online/16k_p_only_clean_before | P0 | 90.71 | 17.94 | 76.43 | 15.96 | 0.00 / 4340.88 | 66.45 | 100.00% |
| native-p-only/online/16k_p_only_clean_before | D0 | 10.90 | 1.50 | 0.77 | 10.74 | 4290.42 / 0.00 | 65.49 | 100.00% |
| native-pd/online/16k_low | P0 | 29.15 | 5.79 | 24.52 | 5.45 | 0.00 / 1426.91 | 66.48 | 100.00% |
| native-pd/online/16k_low | D0 | 3.78 | 0.52 | 0.27 | 3.74 | 1435.41 / 0.00 | 65.52 | 100.00% |
| native-pd/online/16k_near | P0 | 86.82 | 17.17 | 73.18 | 15.27 | 0.00 / 4192.35 | 66.48 | 100.00% |
| native-pd/online/16k_near | D0 | 10.36 | 1.42 | 0.74 | 10.26 | 4192.74 / 0.00 | 65.52 | 100.00% |
| native-pd/online/16k_o64 | P0 | 63.95 | 12.68 | 53.86 | 11.60 | 0.00 / 3156.40 | 66.48 | 100.00% |
| native-pd/online/16k_o64 | D0 | 32.95 | 4.53 | 2.35 | 31.30 | 3137.89 / 0.00 | 65.52 | 100.00% |
| native-pd/online/16k_over | P0 | 91.10 | 18.02 | 76.77 | 16.02 | 0.00 / 4336.71 | 66.48 | 100.00% |
| native-pd/online/16k_over | D0 | 11.17 | 1.54 | 0.80 | 11.17 | 4144.24 / 0.00 | 65.52 | 100.00% |
| native-pd/online/16k_trace_clean_after | P0 | 79.90 | 15.83 | 67.28 | 14.46 | 0.00 / 3780.23 | 66.49 | 100.00% |
| native-pd/online/16k_trace_clean_after | D0 | 8.71 | 1.20 | 0.62 | 8.64 | 3981.97 / 0.00 | 65.53 | 100.00% |
| native-pd/online/16k_trace_clean_before | P0 | 79.44 | 15.72 | 66.88 | 14.08 | 0.00 / 3631.65 | 66.48 | 100.00% |
| native-pd/online/16k_trace_clean_before | D0 | 9.27 | 1.27 | 0.66 | 9.18 | 3949.22 / 0.00 | 65.52 | 100.00% |
| native-pd/online/2k_o256_decode | P0 | 10.98 | 2.23 | 9.24 | 3.57 | 0.00 / 840.61 | 66.48 | 100.00% |
| native-pd/online/2k_o256_decode | D0 | 65.52 | 9.09 | 3.69 | 61.38 | 848.59 / 0.00 | 65.52 | 100.00% |
| native-pd/online/32k_low | P0 | 44.77 | 8.77 | 36.98 | 6.35 | 0.00 / 1734.70 | 66.48 | 100.00% |
| native-pd/online/32k_low | D0 | 2.50 | 0.34 | 0.25 | 2.56 | 1844.80 / 0.00 | 65.52 | 100.00% |
| native-pd/online/32k_near | P0 | 71.06 | 13.92 | 58.80 | 10.05 | 0.00 / 2701.98 | 66.48 | 100.00% |
| native-pd/online/32k_near | D0 | 3.41 | 0.46 | 0.34 | 3.50 | 2724.98 / 0.00 | 65.52 | 100.00% |
| native-pd/online/32k_over | P0 | 85.83 | 16.80 | 70.96 | 12.07 | 0.00 / 3129.98 | 66.48 | 100.00% |
| native-pd/online/32k_over | D0 | 4.67 | 0.64 | 0.47 | 4.77 | 3175.43 / 0.00 | 65.52 | 100.00% |
| native-pd/online/4k_o64_mix | P0 | 25.33 | 5.14 | 21.43 | 6.94 | 0.00 / 1759.27 | 66.48 | 100.00% |
| native-pd/online/4k_o64_mix | D0 | 44.93 | 6.23 | 2.29 | 42.17 | 1801.84 / 0.00 | 65.52 | 100.00% |
| native-pd/online/8k_low | P0 | 28.81 | 5.80 | 24.52 | 6.62 | 0.00 / 1773.97 | 66.48 | 100.00% |
| native-pd/online/8k_low | D0 | 8.01 | 1.11 | 0.44 | 7.89 | 1689.79 / 0.00 | 65.52 | 100.00% |
| native-pd/online/8k_near | P0 | 87.71 | 17.48 | 74.93 | 17.79 | 0.00 / 4787.06 | 66.48 | 100.00% |
| native-pd/online/8k_near | D0 | 22.44 | 3.10 | 1.23 | 21.88 | 4816.28 / 0.00 | 65.52 | 100.00% |
| native-pd/online/8k_over | P0 | 90.73 | 18.08 | 77.52 | 18.37 | 0.00 / 5004.12 | 66.48 | 100.00% |
| native-pd/online/8k_over | D0 | 23.28 | 3.22 | 1.28 | 22.72 | 4921.05 / 0.00 | 65.52 | 100.00% |
| native-static-3gpu/p1_d2/16k_near | P0 | 86.61 | 17.13 | 72.98 | 15.26 | 0.00 / 4235.14 | 66.45 | 100.00% |
| native-static-3gpu/p1_d2/16k_near | D0 | 10.66 | 1.46 | 0.76 | 10.53 | 4058.48 / 0.00 | 65.49 | 100.00% |
| native-static-3gpu/p1_d2/16k_near | D1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 65.48 | 100.00% |
| native-static-3gpu/p1_d2/16k_over | P0 | 90.62 | 17.91 | 76.36 | 15.91 | 0.00 / 4234.00 | 66.45 | 100.00% |
| native-static-3gpu/p1_d2/16k_over | D0 | 10.69 | 1.47 | 0.76 | 10.57 | 4274.80 / 0.00 | 65.49 | 100.00% |
| native-static-3gpu/p1_d2/16k_over | D1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 65.48 | 100.00% |
| native-static-3gpu/p1_d2/8k_near | P0 | 87.48 | 17.43 | 74.73 | 17.78 | 0.00 / 4705.70 | 66.45 | 100.00% |
| native-static-3gpu/p1_d2/8k_near | D0 | 22.65 | 3.13 | 1.24 | 22.05 | 4746.52 / 0.00 | 65.49 | 100.00% |
| native-static-3gpu/p1_d2/8k_near | D1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 / 0.00 | 65.48 | 100.00% |
| native-static-3gpu/p2_d1/16k_near | P0 | 55.19 | 10.96 | 46.43 | 10.14 | 0.00 / 2615.93 | 66.48 | 100.00% |
| native-static-3gpu/p2_d1/16k_near | P1 | 35.31 | 7.04 | 29.69 | 6.74 | 0.00 / 1877.18 | 66.45 | 100.00% |
| native-static-3gpu/p2_d1/16k_near | D0 | 9.72 | 1.33 | 0.76 | 9.64 | 4561.04 / 0.00 | 65.52 | 100.00% |
| native-static-3gpu/p2_d1/16k_over | P0 | 70.62 | 14.01 | 59.43 | 12.90 | 0.00 / 3454.72 | 66.48 | 100.00% |
| native-static-3gpu/p2_d1/16k_over | P1 | 57.19 | 11.36 | 48.11 | 10.46 | 0.00 / 2729.33 | 66.45 | 100.00% |
| native-static-3gpu/p2_d1/16k_over | D0 | 12.09 | 1.65 | 1.00 | 11.93 | 6069.24 / 0.00 | 65.52 | 100.00% |
| native-static-3gpu/p2_d1/8k_near | P0 | 51.96 | 10.45 | 44.24 | 11.79 | 0.00 / 3155.25 | 66.48 | 100.00% |
| native-static-3gpu/p2_d1/8k_near | P1 | 39.28 | 7.92 | 33.42 | 8.96 | 0.00 / 2442.41 | 66.45 | 100.00% |
| native-static-3gpu/p2_d1/8k_near | D0 | 19.99 | 2.76 | 1.19 | 19.35 | 5526.42 / 0.00 | 65.52 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_near | P0/TP0 | 53.06 | 17.39 | 35.91 | 11.21 | 27309.17 / 20448.96 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_near | P0/TP1 | 53.28 | 17.57 | 35.94 | 11.23 | 27364.37 / 20485.27 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_near | D0 | 8.16 | 1.12 | 0.70 | 8.10 | 4557.78 / 0.00 | 65.49 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_over | P0/TP0 | 74.15 | 24.15 | 50.39 | 15.28 | 37221.16 / 27830.56 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_over | P0/TP1 | 73.72 | 24.12 | 49.99 | 15.24 | 37140.42 / 27809.74 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/16k_over | D0 | 9.33 | 1.27 | 0.85 | 9.34 | 6088.71 / 0.00 | 65.49 | 100.00% |
| native-static-3gpu/p_tp2_d1/8k_near | P0/TP0 | 55.70 | 19.55 | 36.66 | 13.71 | 33478.25 / 25038.09 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/8k_near | P0/TP1 | 55.71 | 19.52 | 36.70 | 13.69 | 33464.53 / 25067.89 | 60.54 | 100.00% |
| native-static-3gpu/p_tp2_d1/8k_near | D0 | 10.93 | 1.50 | 0.87 | 10.80 | 5349.62 / 0.00 | 65.49 | 100.00% |
| native-steady/online/16k_near | P0 | 86.98 | 17.20 | 73.29 | 15.30 | 0.00 / 4170.03 | 66.46 | 100.00% |
| native-steady/online/16k_near | D0 | 10.45 | 1.43 | 0.74 | 10.35 | 4053.77 / 0.00 | 65.50 | 100.00% |
| native-steady/online/16k_steady_clean_after | P0 | 90.66 | 17.91 | 76.41 | 15.85 | 0.00 / 4336.01 | 66.46 | 100.00% |
| native-steady/online/16k_steady_clean_after | D0 | 11.00 | 1.51 | 0.78 | 10.84 | 4282.13 / 0.00 | 65.50 | 100.00% |
| native-steady/online/16k_steady_clean_before | P0 | 90.95 | 17.97 | 76.63 | 15.96 | 0.00 / 4222.29 | 66.45 | 100.00% |
| native-steady/online/16k_steady_clean_before | D0 | 10.62 | 1.46 | 0.75 | 10.49 | 4170.45 / 0.00 | 65.49 | 100.00% |
| native-steady/online/mixed | P0 | 39.08 | 7.71 | 32.61 | 6.61 | 0.00 / 1664.52 | 66.46 | 100.00% |
| native-steady/online/mixed | D0 | 32.14 | 4.39 | 2.87 | 30.36 | 1722.17 / 0.00 | 65.50 | 100.00% |
| native-steady/online/mixed_heavy | P0 | 85.45 | 16.82 | 71.39 | 13.66 | 0.00 / 3786.53 | 66.46 | 100.00% |
| native-steady/online/mixed_heavy | D0 | 15.01 | 2.06 | 1.17 | 14.51 | 3734.87 / 0.00 | 65.50 | 100.00% |

## Counter and geometry scope

This capture identifies GPU kernel intervals and PyTorch-observed memory-copy intervals. NIXL copy-engine activity may execute outside the traced worker, so observed memcpy cannot measure all communication. NVML GPM supplies the device-level activity and bandwidth ratios. Achieved FLOPs, actual HBM read/write bytes, L2 and per-operator hardware counters remain unmeasured. The vLLM theoretical FLOP and byte counters are excluded. The iteration annotations identify row counts, while operation-specific GEMM N/K follow the model weight geometry. The trace does not map each GEMM invocation to its layer/operand or recover every active attention KV-length distribution. The Prometheus iteration token counter is an engine output-accounting aggregate. Computed prompt tokens can be credited together on output, so these sums cannot recover chunk-level dispatch shapes. Stream or kernel overlap is temporal evidence. These traces leave physical SM partitioning unverified.

Per-role offsets, interval union, same-ROI occupancy bins, observer shifts, event counts, and leading concrete kernel names are preserved. [CUPTI/Kineto reduction data](../pd_resource_complementary_parallelism/results/native/native_profiles.json)

[native operator time breakdown](../pd_resource_complementary_parallelism/results/native/figures/native_operator_time_breakdown.svg) [native pd temporal kernel timeline 0](../pd_resource_complementary_parallelism/results/native/figures/native_pd_temporal_kernel_timeline_0.svg) [native pd temporal kernel timeline 1](../pd_resource_complementary_parallelism/results/native/figures/native_pd_temporal_kernel_timeline_1.svg) [native pd temporal kernel timeline 2](../pd_resource_complementary_parallelism/results/native/figures/native_pd_temporal_kernel_timeline_2.svg) [native pd temporal kernel timeline 3](../pd_resource_complementary_parallelism/results/native/figures/native_pd_temporal_kernel_timeline_3.svg) [native tensor and dram gpm](../pd_resource_complementary_parallelism/results/native/figures/native_tensor_and_dram_gpm.svg)
