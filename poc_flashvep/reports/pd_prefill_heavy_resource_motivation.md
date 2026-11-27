# H200 PD prefill-heavy resource motivation

## Measurement contract

The client uses frozen token IDs, a seeded open-loop Poisson clock and greedy BF16 native vLLM. PD uses a physical NIXL lease returned by P and consumed by D. DP sends the same token contract to a native colocated worker. Primary cohorts disable automatic prefix caching. Fixed-length serving returns at least 16 tokens. The independent prefill reference returns one token as an internal phase capacity probe.

## Dataset lineage

The controlled trace source is EADA's `qwen_128.json`, SHA256 `c3472d660ffa61988e04205a1dcbb02b0ff7737ece729a1fdd001cb5ba85b904`. The public LongBench v1 archive at revision `5e628be450b7e67fb7ae6e201bd6d8f7056f7672` has SHA256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`. The earlier cohort builder applies the publisher's task templates and the pinned Qwen2.5-7B-Instruct tokenizer at `a09a35458c702b33eeacc393d103063234e8bc28`. It filters to 2,048–28,672 templated tokens, sorts within each task by the SHA256 of the EADA salt and official request ID, and retains 32 each from Qasper, HotpotQA, NarrativeQA and GovReport. The stored template maps equal the publisher's maps, with two-space UTF-8 JSON formatting and a final newline. The cohort was reproduced byte-for-byte on CPU against the pinned assets. [Pinned public dataset](https://huggingface.co/datasets/zai-org/LongBench/tree/5e628be450b7e67fb7ae6e201bd6d8f7056f7672) [Pinned publisher template](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/config/dataset2prompt.json) [Prior cohort builder](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/encoder_assisted_decode_attention/longbench.py) [Source ID and hash receipt](https://github.com/jyssys/MLLM-EP/blob/07892cbe7193366ec422f1df82006c6e69f91cec/poc_flashvep/encoder_assisted_decode_attention/results/longbench_manifest_receipt.json)

The current trace builder shuffles the source with seed 20261129, re-tokenizes its user bodies with the model-specific non-thinking chat template, and truncates or extends the token bodies to exact input lengths. Extensions use subsequent corpus rows. Arrival offsets use seed 20261128 and are scaled by offered load. The 16K/16 cohort extends 17 of 24, 49 of 64 and 103 of 128 inputs. The 128-request variant has 123 distinct token streams. APC remains disabled in the primary arms. These controlled traces measure serving behavior for LongBench-derived text. Task quality requires untouched task inputs. [Current trace builder](../pd_resource_complementary_parallelism/workload.py)

## Metric definitions

TTFT starts at scheduled arrival. TPOT is the mean inter-token interval per request. ITL is pooled across client-observed token timestamps. SSE multi-token events share one arrival timestamp. The reducer preserves their counts and resulting zero intervals. The frozen joint SLO is TTFT ≤5,000 ms, TPOT ≤33 ms and every ITL ≤34 ms. Failures count in the offered cohort and never contribute to goodput. Successful-request percentiles and finite arrival-to-drain throughput use different denominators. These finite cohorts cannot certify sustainable throughput.

The queue and GPU-profile ROI uses the common arrival window with each end trimmed by 10% of its span, capped at two seconds. Sampled waiting measures the scheduler's published queue. It does not resolve P-ready work, D-credit stalls, or transfer wait. The P-reference ratio uses completed input tokens per finite wall second. It differs from a same-ROI independent P service-rate comparison.

The archived initial `native-pd` sweep polled GPM from its client loop. The later `dense-controls` comparisons use a separate background sampler. Use the latter matched pairs for tail-latency comparisons.

In the archived campaign, the steady shell wrapper reported exit 2 while parsing its trailing clause after recording all six complete cohorts. The per-cohort receipts and their successful request counts remain the serving evidence.

## Independent P reference

| P witness | Input | Batch token budget | Offered req/s | OK / offered | Wall s | Completed req/s | Input tokens/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p-reference/online/16k_high | [16384] | 8192 | 12 | 24 / 24 | 12.18 | 1.971 | 32287 |
| p-reference/online/32k_high | [32768] | 8192 | 8 | 24 / 24 | 30.95 | 0.776 | 25413 |
| p-reference/online/8k_high | [8192] | 8192 | 12 | 24 / 24 | 5.28 | 4.546 | 37237 |
| p-tuned/pb16384/16k_high | [16384] | 16384 | 12 | 24 / 24 | 12.20 | 1.967 | 32235 |
| p-tuned/pb16384/32k_high | [32768] | 16384 | 8 | 24 / 24 | 30.81 | 0.779 | 25522 |
| p-tuned/pb16384/8k_high | [8192] | 16384 | 12 | 24 / 24 | 5.30 | 4.532 | 37128 |
| p-tuned/pb32768/16k_high | [16384] | 32768 | 12 | 24 / 24 | 12.15 | 1.975 | 32363 |
| p-tuned/pb32768/32k_high | [32768] | 32768 | 8 | 24 / 24 | 30.83 | 0.779 | 25511 |
| p-tuned/pb32768/8k_high | [8192] | 32768 | 12 | 24 / 24 | 5.29 | 4.536 | 37162 |

## Clean native serving

| Run | GPUs | Offered req/s | APC | OK / offered | TTFT p95 ms | TPOT p95 ms | ITL p99 ms | Joint SLO | Finite req/s | Goodput req/s / per GPU | P waiting mean | P-ref finite ratio matched / best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense-controls/dp2/16k_near | 2 | 1.9 | off | 64 / 64 | 1009.23 | 72.16 | 294.19 | 45 / 64 | 1.923 | 1.352 / 0.676 | — | — / — |
| dense-controls/dp2/16k_over | 2 | 2.6 | off | 64 / 64 | 1333.89 | 155.58 | 296.92 | 33 / 64 | 2.614 | 1.348 / 0.674 | — | — / — |
| dense-controls/dp2/32k_near | 2 | 0.8 | off | 24 / 24 | 1896.84 | 87.67 | 364.02 | 21 / 24 | 0.703 | 0.615 / 0.308 | — | — / — |
| dense-controls/dp2/8k_near | 2 | 4.5 | off | 64 / 64 | 532.43 | 46.81 | 191.73 | 44 / 64 | 4.539 | 3.120 / 1.560 | — | — / — |
| dense-controls/dp2-chunk2048/16k_near | 2 | 1.9 | off | 64 / 64 | 1048.56 | 63.09 | 83.47 | 44 / 64 | 1.923 | 1.322 / 0.661 | — | — / — |
| dense-controls/dp2-chunk2048/16k_over | 2 | 2.6 | off | 64 / 64 | 1396.63 | 66.23 | 84.74 | 30 / 64 | 2.559 | 1.200 / 0.600 | — | — / — |
| dense-controls/dp2-chunk2048/8k_near | 2 | 4.5 | off | 64 / 64 | 452.14 | 42.38 | 66.28 | 43 / 64 | 4.537 | 3.048 / 1.524 | — | — / — |
| dense-controls/dp2-no-chunk/16k_near | 2 | 1.9 | off | 64 / 64 | 999.22 | 71.29 | 488.44 | 45 / 64 | 1.923 | 1.352 / 0.676 | — | — / — |
| dense-controls/dp2-no-chunk/16k_over | 2 | 2.6 | off | 64 / 64 | 1474.45 | 166.63 | 973.18 | 35 / 64 | 2.614 | 1.430 / 0.715 | — | — / — |
| dense-controls/dp2-no-chunk/8k_near | 2 | 4.5 | off | 64 / 64 | 487.89 | 46.01 | 208.11 | 44 / 64 | 4.538 | 3.120 / 1.560 | — | — / — |
| dense-controls/p-light/16k_light_clean_after | 2 | 2.6 | off | 24 / 24 | 3523.09 | 4.63 | 7.33 | 24 / 24 | 1.731 | 1.731 / 0.865 | 0.50 | 87.83% / 87.62% |
| dense-controls/p-light/16k_light_clean_before | 2 | 2.6 | off | 24 / 24 | 3424.95 | 4.58 | 7.39 | 24 / 24 | 1.733 | 1.733 / 0.867 | 0.45 | 87.95% / 87.74% |
| dense-controls/p-light/16k_near | 2 | 1.9 | off | 64 / 64 | 3893.40 | 5.02 | 6.54 | 64 / 64 | 1.762 | 1.762 / 0.881 | 2.37 | 89.42% / 89.21% |
| dense-controls/p-light/16k_over | 2 | 2.6 | off | 64 / 64 | 9768.08 | 4.49 | 8.03 | 27 / 64 | 1.873 | 0.790 / 0.395 | 6.63 | 95.03% / 94.81% |
| dense-controls/p-light/8k_near_clean_after | 2 | 4.5 | off | 64 / 64 | 1933.13 | 4.81 | 7.26 | 64 / 64 | 4.070 | 4.070 / 2.035 | 2.13 | 89.54% / 89.54% |
| dense-controls/p-light/8k_near_clean_before | 2 | 4.5 | off | 64 / 64 | 1961.70 | 4.89 | 6.85 | 64 / 64 | 4.052 | 4.052 / 2.026 | 1.99 | 89.14% / 89.14% |
| dense-controls/pd-cross/16k_near | 2 | 1.9 | off | 64 / 64 | 3868.43 | 4.91 | 9.19 | 64 / 64 | 1.768 | 1.768 / 0.884 | 2.25 | 89.72% / 89.51% |
| dense-controls/pd-cross/16k_over | 2 | 2.6 | off | 64 / 64 | 9963.36 | 4.59 | 8.17 | 26 / 64 | 1.863 | 0.757 / 0.378 | 6.66 | 94.51% / 94.29% |
| dense-controls/pd-cross/8k_near | 2 | 4.5 | off | 64 / 64 | 1919.38 | 4.89 | 8.95 | 63 / 64 | 4.066 | 4.003 / 2.001 | 1.88 | 89.46% / 89.46% |
| dense-controls/pd-deflect/16k_near | 2 | 1.9 | off | 64 / 64 | 2100.93 | 5.41 | 20.79 | 61 / 64 | 1.908 | 1.819 / 0.909 | 0.39 | — / — |
| dense-controls/pd-deflect/16k_over | 2 | 2.6 | off | 64 / 64 | 4497.97 | 13.63 | 25.56 | 55 / 64 | 2.285 | 1.964 / 0.982 | 2.99 | — / — |
| dense-controls/pd-deflect/8k_near | 2 | 4.5 | off | 64 / 64 | 1416.36 | 9.15 | 20.82 | 56 / 64 | 4.347 | 3.803 / 1.902 | 0.62 | — / — |
| dense-controls/pd-layout-probe/8k_low | 2 | 2 | off | 24 / 24 | 555.25 | 4.54 | 6.30 | 24 / 24 | 1.798 | 1.798 / 0.899 | 0.00 | 39.55% / 39.55% |
| native-p-only/online/16k_p_only_clean_after | 2 | 2.6 | off | 64 / 64 | 9755.83 | 4.00 | 6.91 | 26 / 64 | 1.877 | 0.763 / 0.381 | 6.68 | 95.26% / 95.04% |
| native-p-only/online/16k_p_only_clean_before | 2 | 2.6 | off | 64 / 64 | 9739.42 | 4.53 | 9.37 | 27 / 64 | 1.872 | 0.790 / 0.395 | 6.61 | 94.98% / 94.76% |
| native-pd/online/16k_low | 2 | 0.8 | off | 24 / 24 | 1215.10 | 4.77 | 12.18 | 21 / 24 | 0.726 | 0.636 / 0.318 | 0.00 | 36.86% / 36.77% |
| native-pd/online/16k_near | 2 | 1.9 | off | 64 / 64 | 3861.32 | 5.41 | 24.96 | 56 / 64 | 1.766 | 1.546 / 0.773 | 2.25 | 89.63% / 89.42% |
| native-pd/online/16k_o64 | 2 | 1.9 | off | 24 / 24 | 2018.47 | 7.26 | 43.08 | 10 / 24 | 1.484 | 0.618 / 0.309 | 0.21 | 75.30% / 75.12% |
| native-pd/online/16k_over | 2 | 2.6 | off | 64 / 64 | 9795.82 | 4.92 | 15.91 | 26 / 64 | 1.872 | 0.761 / 0.380 | 6.66 | 95.01% / 94.79% |
| native-pd/online/16k_trace_clean_after | 2 | 2.6 | off | 24 / 24 | 3478.79 | 4.83 | 10.70 | 22 / 24 | 1.737 | 1.592 / 0.796 | 0.45 | 88.13% / 87.93% |
| native-pd/online/16k_trace_clean_before | 2 | 2.6 | off | 24 / 24 | 3490.86 | 4.10 | 11.15 | 23 / 24 | 1.730 | 1.658 / 0.829 | 0.47 | 87.77% / 87.56% |
| native-pd/online/2k_o256_decode | 2 | 4 | off | 24 / 24 | 188.93 | 6.10 | 24.65 | 0 / 24 | 3.042 | 0.000 / 0.000 | 0.00 | — / — |
| native-pd/online/32k_low | 2 | 0.45 | off | 24 / 24 | 3001.89 | 3.26 | 11.89 | 23 / 24 | 0.402 | 0.385 / 0.193 | 0.05 | 51.84% / 51.62% |
| native-pd/online/32k_near | 2 | 0.8 | off | 24 / 24 | 5685.50 | 4.37 | 14.41 | 22 / 24 | 0.620 | 0.569 / 0.284 | 0.34 | 80.01% / 79.66% |
| native-pd/online/32k_over | 2 | 1.15 | off | 24 / 24 | 9771.03 | 4.95 | 13.20 | 13 / 24 | 0.724 | 0.392 / 0.196 | 0.91 | 93.40% / 93.00% |
| native-pd/online/4k_o64_mix | 2 | 4 | off | 24 / 24 | 316.78 | 6.13 | 32.40 | 13 / 24 | 3.447 | 1.867 / 0.934 | 0.00 | — / — |
| native-pd/online/8k_low | 2 | 2 | off | 24 / 24 | 557.70 | 4.99 | 15.08 | 23 / 24 | 1.800 | 1.725 / 0.862 | 0.00 | 39.59% / 39.59% |
| native-pd/online/8k_near | 2 | 4.5 | off | 64 / 64 | 1906.44 | 8.94 | 24.11 | 56 / 64 | 4.066 | 3.558 / 1.779 | 2.03 | 89.45% / 89.45% |
| native-pd/online/8k_over | 2 | 6 | off | 64 / 64 | 4375.08 | 6.86 | 18.23 | 57 / 64 | 4.269 | 3.802 / 1.901 | 5.34 | 93.93% / 93.93% |
| native-static-3gpu/p1_d2/16k_near | 3 | 1.9 | off | 64 / 64 | 3882.21 | 4.17 | 8.44 | 64 / 64 | 1.773 | 1.773 / 0.591 | 2.27 | 89.96% / 89.75% |
| native-static-3gpu/p1_d2/16k_over | 3 | 2.6 | off | 64 / 64 | 9875.84 | 4.38 | 6.78 | 26 / 64 | 1.867 | 0.758 / 0.253 | 6.68 | 94.74% / 94.51% |
| native-static-3gpu/p1_d2/8k_near | 3 | 4.5 | off | 64 / 64 | 1943.37 | 4.85 | 8.93 | 64 / 64 | 4.066 | 4.066 / 1.355 | 1.99 | 89.45% / 89.45% |
| native-static-3gpu/p2_d1/16k_near | 3 | 1.9 | off | 64 / 64 | 1251.77 | 11.29 | 20.34 | 59 / 64 | 1.911 | 1.761 / 0.587 | 0.04 | — / — |
| native-static-3gpu/p2_d1/16k_over | 3 | 2.6 | off | 64 / 64 | 1379.42 | 11.59 | 23.29 | 58 / 64 | 2.586 | 2.344 / 0.781 | 0.15 | — / — |
| native-static-3gpu/p2_d1/8k_near | 3 | 4.5 | off | 64 / 64 | 686.58 | 8.87 | 62.64 | 53 / 64 | 4.495 | 3.723 / 1.241 | 0.00 | — / — |
| native-static-3gpu/p_tp2_d1/16k_near | 3 | 1.9 | off | 64 / 64 | 1529.74 | 40.39 | 265.18 | 31 / 64 | 1.902 | 0.921 / 0.307 | 0.19 | — / — |
| native-static-3gpu/p_tp2_d1/16k_over | 3 | 2.6 | off | 64 / 64 | 1891.72 | 51.78 | 262.28 | 21 / 64 | 2.573 | 0.844 / 0.281 | 0.71 | — / — |
| native-static-3gpu/p_tp2_d1/8k_near | 3 | 4.5 | off | 64 / 64 | 1122.32 | 40.42 | 134.19 | 43 / 64 | 4.477 | 3.008 / 1.003 | 0.15 | — / — |
| native-steady/online/16k_near | 2 | 1.9 | off | 64 / 64 | 3865.19 | 4.66 | 8.24 | 64 / 64 | 1.772 | 1.772 / 0.886 | 2.22 | 89.94% / 89.73% |
| native-steady/online/16k_steady_clean_after | 2 | 2.6 | off | 64 / 64 | 9884.91 | 4.42 | 10.66 | 26 / 64 | 1.870 | 0.760 / 0.380 | 6.73 | 94.90% / 94.68% |
| native-steady/online/16k_steady_clean_before | 2 | 2.6 | off | 64 / 64 | 9814.63 | 4.58 | 7.09 | 26 / 64 | 1.874 | 0.761 / 0.381 | 6.66 | 95.09% / 94.86% |
| native-steady/online/mixed | 2 | 1.25 | off | 24 / 24 | 1965.26 | 6.87 | 12.05 | 17 / 24 | 1.077 | 0.763 / 0.381 | 0.00 | — / — |
| native-steady/online/mixed_heavy | 2 | 1.45 | off | 64 / 64 | 4282.93 | 6.34 | 15.13 | 53 / 64 | 1.386 | 1.148 / 0.574 | 1.54 | — / — |

Interrupted cohorts have raw-stage receipts without a valid terminal run record. The incomplete cohort manifest preserves them without assigning latency or success rates. [Incomplete raw manifest](../pd_resource_complementary_parallelism/results/native/native_incomplete.json)

## Matched native output controls

Pairs share the model, physical GPU set, trace digest, offered load, APC, sampling regime and generation contract. Output parity compares every emitted token of requests successful in both arms. A match supports output reproducibility for that cohort. The table does not infer bitwise logit equality or task accuracy.

| First run | Second run | Exact / both OK / total | TTFT p95 first / second ms | Goodput first / second req/s |
| --- | --- | ---: | ---: | ---: |
| dense-controls/dp2/16k_near | dense-controls/dp2-chunk2048/16k_near | 62 / 64 / 64 | 1009.23 / 1048.56 | 1.352 / 1.322 |
| dense-controls/dp2/16k_near | dense-controls/dp2-no-chunk/16k_near | 63 / 64 / 64 | 1009.23 / 999.22 | 1.352 / 1.352 |
| dense-controls/dp2/16k_near | dense-controls/p-light/16k_near | 60 / 64 / 64 | 1009.23 / 3893.40 | 1.352 / 1.762 |
| dense-controls/dp2/16k_near | dense-controls/pd-cross/16k_near | 60 / 64 / 64 | 1009.23 / 3868.43 | 1.352 / 1.768 |
| dense-controls/dp2/16k_near | dense-controls/pd-deflect/16k_near | 60 / 64 / 64 | 1009.23 / 2100.93 | 1.352 / 1.819 |
| dense-controls/dp2/16k_near | native-steady/online/16k_near | 60 / 64 / 64 | 1009.23 / 3865.19 | 1.352 / 1.772 |
| dense-controls/dp2-chunk2048/16k_near | dense-controls/dp2-no-chunk/16k_near | 63 / 64 / 64 | 1048.56 / 999.22 | 1.322 / 1.352 |
| dense-controls/dp2-chunk2048/16k_near | dense-controls/p-light/16k_near | 60 / 64 / 64 | 1048.56 / 3893.40 | 1.322 / 1.762 |
| dense-controls/dp2-chunk2048/16k_near | dense-controls/pd-cross/16k_near | 60 / 64 / 64 | 1048.56 / 3868.43 | 1.322 / 1.768 |
| dense-controls/dp2-chunk2048/16k_near | dense-controls/pd-deflect/16k_near | 60 / 64 / 64 | 1048.56 / 2100.93 | 1.322 / 1.819 |
| dense-controls/dp2-chunk2048/16k_near | native-steady/online/16k_near | 60 / 64 / 64 | 1048.56 / 3865.19 | 1.322 / 1.772 |
| dense-controls/dp2-no-chunk/16k_near | dense-controls/p-light/16k_near | 61 / 64 / 64 | 999.22 / 3893.40 | 1.352 / 1.762 |
| dense-controls/dp2-no-chunk/16k_near | dense-controls/pd-cross/16k_near | 61 / 64 / 64 | 999.22 / 3868.43 | 1.352 / 1.768 |
| dense-controls/dp2-no-chunk/16k_near | dense-controls/pd-deflect/16k_near | 61 / 64 / 64 | 999.22 / 2100.93 | 1.352 / 1.819 |
| dense-controls/dp2-no-chunk/16k_near | native-steady/online/16k_near | 61 / 64 / 64 | 999.22 / 3865.19 | 1.352 / 1.772 |
| dense-controls/p-light/16k_near | dense-controls/pd-cross/16k_near | 64 / 64 / 64 | 3893.40 / 3868.43 | 1.762 / 1.768 |
| dense-controls/p-light/16k_near | dense-controls/pd-deflect/16k_near | 64 / 64 / 64 | 3893.40 / 2100.93 | 1.762 / 1.819 |
| dense-controls/p-light/16k_near | native-steady/online/16k_near | 64 / 64 / 64 | 3893.40 / 3865.19 | 1.762 / 1.772 |
| dense-controls/pd-cross/16k_near | dense-controls/pd-deflect/16k_near | 64 / 64 / 64 | 3868.43 / 2100.93 | 1.768 / 1.819 |
| dense-controls/pd-cross/16k_near | native-steady/online/16k_near | 64 / 64 / 64 | 3868.43 / 3865.19 | 1.768 / 1.772 |
| dense-controls/pd-deflect/16k_near | native-steady/online/16k_near | 64 / 64 / 64 | 2100.93 / 3865.19 | 1.819 / 1.772 |
| dense-controls/dp2/16k_over | dense-controls/dp2-chunk2048/16k_over | 60 / 64 / 64 | 1333.89 / 1396.63 | 1.348 / 1.200 |
| dense-controls/dp2/16k_over | dense-controls/dp2-no-chunk/16k_over | 62 / 64 / 64 | 1333.89 / 1474.45 | 1.348 / 1.430 |
| dense-controls/dp2/16k_over | dense-controls/p-light/16k_over | 60 / 64 / 64 | 1333.89 / 9768.08 | 1.348 / 0.790 |
| dense-controls/dp2/16k_over | dense-controls/pd-cross/16k_over | 60 / 64 / 64 | 1333.89 / 9963.36 | 1.348 / 0.757 |
| dense-controls/dp2/16k_over | dense-controls/pd-deflect/16k_over | 59 / 64 / 64 | 1333.89 / 4497.97 | 1.348 / 1.964 |
| dense-controls/dp2/16k_over | native-p-only/online/16k_p_only_clean_before | 60 / 64 / 64 | 1333.89 / 9739.42 | 1.348 / 0.790 |
| dense-controls/dp2/16k_over | native-steady/online/16k_steady_clean_before | 60 / 64 / 64 | 1333.89 / 9814.63 | 1.348 / 0.761 |
| dense-controls/dp2-chunk2048/16k_over | dense-controls/dp2-no-chunk/16k_over | 60 / 64 / 64 | 1396.63 / 1474.45 | 1.200 / 1.430 |
| dense-controls/dp2-chunk2048/16k_over | dense-controls/p-light/16k_over | 58 / 64 / 64 | 1396.63 / 9768.08 | 1.200 / 0.790 |
| dense-controls/dp2-chunk2048/16k_over | dense-controls/pd-cross/16k_over | 58 / 64 / 64 | 1396.63 / 9963.36 | 1.200 / 0.757 |
| dense-controls/dp2-chunk2048/16k_over | dense-controls/pd-deflect/16k_over | 57 / 64 / 64 | 1396.63 / 4497.97 | 1.200 / 1.964 |
| dense-controls/dp2-chunk2048/16k_over | native-p-only/online/16k_p_only_clean_before | 58 / 64 / 64 | 1396.63 / 9739.42 | 1.200 / 0.790 |
| dense-controls/dp2-chunk2048/16k_over | native-steady/online/16k_steady_clean_before | 58 / 64 / 64 | 1396.63 / 9814.63 | 1.200 / 0.761 |
| dense-controls/dp2-no-chunk/16k_over | dense-controls/p-light/16k_over | 60 / 64 / 64 | 1474.45 / 9768.08 | 1.430 / 0.790 |
| dense-controls/dp2-no-chunk/16k_over | dense-controls/pd-cross/16k_over | 60 / 64 / 64 | 1474.45 / 9963.36 | 1.430 / 0.757 |
| dense-controls/dp2-no-chunk/16k_over | dense-controls/pd-deflect/16k_over | 59 / 64 / 64 | 1474.45 / 4497.97 | 1.430 / 1.964 |
| dense-controls/dp2-no-chunk/16k_over | native-p-only/online/16k_p_only_clean_before | 60 / 64 / 64 | 1474.45 / 9739.42 | 1.430 / 0.790 |
| dense-controls/dp2-no-chunk/16k_over | native-steady/online/16k_steady_clean_before | 60 / 64 / 64 | 1474.45 / 9814.63 | 1.430 / 0.761 |
| dense-controls/p-light/16k_over | dense-controls/pd-cross/16k_over | 64 / 64 / 64 | 9768.08 / 9963.36 | 0.790 / 0.757 |
| dense-controls/p-light/16k_over | dense-controls/pd-deflect/16k_over | 63 / 64 / 64 | 9768.08 / 4497.97 | 0.790 / 1.964 |
| dense-controls/p-light/16k_over | native-p-only/online/16k_p_only_clean_before | 64 / 64 / 64 | 9768.08 / 9739.42 | 0.790 / 0.790 |
| dense-controls/p-light/16k_over | native-steady/online/16k_steady_clean_before | 64 / 64 / 64 | 9768.08 / 9814.63 | 0.790 / 0.761 |
| dense-controls/pd-cross/16k_over | dense-controls/pd-deflect/16k_over | 63 / 64 / 64 | 9963.36 / 4497.97 | 0.757 / 1.964 |
| dense-controls/pd-cross/16k_over | native-p-only/online/16k_p_only_clean_before | 64 / 64 / 64 | 9963.36 / 9739.42 | 0.757 / 0.790 |
| dense-controls/pd-cross/16k_over | native-steady/online/16k_steady_clean_before | 64 / 64 / 64 | 9963.36 / 9814.63 | 0.757 / 0.761 |
| dense-controls/pd-deflect/16k_over | native-p-only/online/16k_p_only_clean_before | 63 / 64 / 64 | 4497.97 / 9739.42 | 1.964 / 0.790 |
| dense-controls/pd-deflect/16k_over | native-steady/online/16k_steady_clean_before | 63 / 64 / 64 | 4497.97 / 9814.63 | 1.964 / 0.761 |
| native-p-only/online/16k_p_only_clean_before | native-steady/online/16k_steady_clean_before | 64 / 64 / 64 | 9739.42 / 9814.63 | 0.790 / 0.761 |
| dense-controls/dp2/8k_near | dense-controls/dp2-chunk2048/8k_near | 60 / 64 / 64 | 532.43 / 452.14 | 3.120 / 3.048 |
| dense-controls/dp2/8k_near | dense-controls/dp2-no-chunk/8k_near | 63 / 64 / 64 | 532.43 / 487.89 | 3.120 / 3.120 |
| dense-controls/dp2/8k_near | dense-controls/p-light/8k_near_clean_before | 62 / 64 / 64 | 532.43 / 1961.70 | 3.120 / 4.052 |
| dense-controls/dp2/8k_near | dense-controls/pd-cross/8k_near | 62 / 64 / 64 | 532.43 / 1919.38 | 3.120 / 4.003 |
| dense-controls/dp2/8k_near | dense-controls/pd-deflect/8k_near | 62 / 64 / 64 | 532.43 / 1416.36 | 3.120 / 3.803 |
| dense-controls/dp2-chunk2048/8k_near | dense-controls/dp2-no-chunk/8k_near | 59 / 64 / 64 | 452.14 / 487.89 | 3.048 / 3.120 |
| dense-controls/dp2-chunk2048/8k_near | dense-controls/p-light/8k_near_clean_before | 60 / 64 / 64 | 452.14 / 1961.70 | 3.048 / 4.052 |
| dense-controls/dp2-chunk2048/8k_near | dense-controls/pd-cross/8k_near | 60 / 64 / 64 | 452.14 / 1919.38 | 3.048 / 4.003 |
| dense-controls/dp2-chunk2048/8k_near | dense-controls/pd-deflect/8k_near | 58 / 64 / 64 | 452.14 / 1416.36 | 3.048 / 3.803 |
| dense-controls/dp2-no-chunk/8k_near | dense-controls/p-light/8k_near_clean_before | 61 / 64 / 64 | 487.89 / 1961.70 | 3.120 / 4.052 |
| dense-controls/dp2-no-chunk/8k_near | dense-controls/pd-cross/8k_near | 61 / 64 / 64 | 487.89 / 1919.38 | 3.120 / 4.003 |
| dense-controls/dp2-no-chunk/8k_near | dense-controls/pd-deflect/8k_near | 61 / 64 / 64 | 487.89 / 1416.36 | 3.120 / 3.803 |
| dense-controls/p-light/8k_near_clean_before | dense-controls/pd-cross/8k_near | 64 / 64 / 64 | 1961.70 / 1919.38 | 4.052 / 4.003 |
| dense-controls/p-light/8k_near_clean_before | dense-controls/pd-deflect/8k_near | 62 / 64 / 64 | 1961.70 / 1416.36 | 4.052 / 3.803 |
| dense-controls/pd-cross/8k_near | dense-controls/pd-deflect/8k_near | 62 / 64 / 64 | 1919.38 / 1416.36 | 4.003 / 3.803 |
| native-static-3gpu/p1_d2/16k_near | native-static-3gpu/p2_d1/16k_near | 62 / 64 / 64 | 3882.21 / 1251.77 | 1.773 / 1.761 |
| native-static-3gpu/p1_d2/16k_near | native-static-3gpu/p_tp2_d1/16k_near | 61 / 64 / 64 | 3882.21 / 1529.74 | 1.773 / 0.921 |
| native-static-3gpu/p2_d1/16k_near | native-static-3gpu/p_tp2_d1/16k_near | 61 / 64 / 64 | 1251.77 / 1529.74 | 1.761 / 0.921 |
| native-static-3gpu/p1_d2/16k_over | native-static-3gpu/p2_d1/16k_over | 62 / 64 / 64 | 9875.84 / 1379.42 | 0.758 / 2.344 |
| native-static-3gpu/p1_d2/16k_over | native-static-3gpu/p_tp2_d1/16k_over | 58 / 64 / 64 | 9875.84 / 1891.72 | 0.758 / 0.844 |
| native-static-3gpu/p2_d1/16k_over | native-static-3gpu/p_tp2_d1/16k_over | 60 / 64 / 64 | 1379.42 / 1891.72 | 2.344 / 0.844 |
| native-static-3gpu/p1_d2/8k_near | native-static-3gpu/p2_d1/8k_near | 58 / 64 / 64 | 1943.37 / 686.58 | 4.066 / 3.723 |
| native-static-3gpu/p1_d2/8k_near | native-static-3gpu/p_tp2_d1/8k_near | 60 / 64 / 64 | 1943.37 / 1122.32 | 4.066 / 3.008 |
| native-static-3gpu/p2_d1/8k_near | native-static-3gpu/p_tp2_d1/8k_near | 58 / 64 / 64 | 686.58 / 1122.32 | 3.723 / 3.008 |

## Stage and handoff observations

The client observes P-response completion and the first D token. Ready-to-submit is an application boundary. D submit-to-first includes transfer, scheduling and model work. NIXL byte and time deltas are measured by the connector. The client issues the D RPC after the P response. P can still retain block leases from previously completed prefills. Transport-time sums can include overlapping transfers. The D waiting gauge includes every published reason, and the preemption counter does not count all admission waits.

| Run | P RPC p95 ms | D submit-to-first p95 ms | D wait mean | Transfer count | Connector KV GiB | Mean transfer / post ms | Descriptors / transfer | P / D preemptions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense-controls/p-light/16k_light_clean_after | 3369.80 | 236.74 | 0.00 | 24 | 54.00 | 121.62 / 96.42 | 29214 | 0 / 0 |
| dense-controls/p-light/16k_light_clean_before | 3318.82 | 231.43 | 0.00 | 24 | 54.00 | 113.59 / 89.87 | 26919 | 0 / 0 |
| dense-controls/p-light/16k_near | 3790.56 | 229.39 | 0.04 | 64 | 144.00 | 121.23 / 100.33 | 29979 | 0 / 0 |
| dense-controls/p-light/16k_over | 9562.83 | 229.56 | 0.11 | 64 | 144.00 | 104.64 / 86.97 | 25676 | 0 / 0 |
| dense-controls/p-light/8k_near_clean_after | 1795.45 | 133.07 | 0.05 | 64 | 72.00 | 60.26 / 45.43 | 14129 | 0 / 0 |
| dense-controls/p-light/8k_near_clean_before | 1828.79 | 129.95 | 0.12 | 64 | 72.00 | 64.46 / 47.62 | 14990 | 0 / 0 |
| dense-controls/pd-cross/16k_near | 3672.59 | 234.63 | 0.04 | 64 | 144.00 | 128.67 / 106.50 | 29973 | 0 / 0 |
| dense-controls/pd-cross/16k_over | 9732.07 | 250.96 | 0.05 | 64 | 144.00 | 130.75 / 102.49 | 30552 | 0 / 0 |
| dense-controls/pd-cross/8k_near | 1791.01 | 129.19 | 0.10 | 64 | 72.00 | 66.35 / 49.40 | 15855 | 0 / 0 |
| dense-controls/pd-deflect/16k_near | 1866.81 | 575.54 | 0.07 | 53 | 119.25 | 149.70 / 93.97 | 26482 | 0 / 0 |
| dense-controls/pd-deflect/16k_over | 4283.27 | 555.76 | 0.08 | 52 | 117.00 | 146.25 / 96.49 | 29101 | 0 / 0 |
| dense-controls/pd-deflect/8k_near | 1249.05 | 253.04 | 0.15 | 58 | 65.25 | 73.26 / 46.14 | 14311 | 0 / 0 |
| dense-controls/pd-layout-probe/8k_low | 431.45 | 132.97 | 0.03 | 24 | 27.00 | 60.47 / 45.42 | 14600 | 0 / 0 |
| native-p-only/online/16k_p_only_clean_after | 9541.41 | 242.02 | 0.10 | 64 | 144.00 | 120.16 / 92.14 | 27953 | 0 / 0 |
| native-p-only/online/16k_p_only_clean_before | 9568.27 | 235.25 | 0.03 | 64 | 144.00 | 120.81 / 93.61 | 28528 | 0 / 0 |
| native-pd/online/16k_low | 985.75 | 298.60 | 0.02 | 24 | 54.00 | 152.79 / 129.93 | 34570 | 0 / 0 |
| native-pd/online/16k_near | 3700.50 | 224.45 | 0.04 | 64 | 144.00 | 113.43 / 95.69 | 28556 | 0 / 0 |
| native-pd/online/16k_o64 | 1864.66 | 217.54 | 0.05 | 24 | 54.00 | 118.32 / 99.44 | 29817 | 0 / 0 |
| native-pd/online/16k_over | 9566.51 | 232.25 | 0.03 | 64 | 144.00 | 132.01 / 111.09 | 33423 | 0 / 0 |
| native-pd/online/16k_trace_clean_after | 3337.70 | 225.72 | 0.00 | 24 | 54.00 | 113.30 / 93.08 | 27166 | 0 / 0 |
| native-pd/online/16k_trace_clean_before | 3350.75 | 235.79 | 0.00 | 24 | 54.00 | 120.26 / 97.96 | 29742 | 0 / 0 |
| native-pd/online/2k_o256_decode | 95.83 | 119.27 | 0.00 | 24 | 6.75 | 18.00 / 10.26 | 3598 | 0 / 0 |
| native-pd/online/32k_low | 2717.20 | 431.47 | 0.00 | 24 | 108.00 | 249.94 / 220.49 | 58434 | 0 / 0 |
| native-pd/online/32k_near | 5340.69 | 376.06 | 0.02 | 24 | 108.00 | 199.01 / 172.32 | 50812 | 0 / 0 |
| native-pd/online/32k_over | 9459.50 | 401.98 | 0.03 | 24 | 108.00 | 192.66 / 168.34 | 49278 | 0 / 0 |
| native-pd/online/4k_o64_mix | 211.87 | 91.50 | 0.00 | 24 | 13.50 | 32.04 / 21.39 | 7690 | 0 / 0 |
| native-pd/online/8k_low | 436.92 | 143.56 | 0.11 | 24 | 27.00 | 59.04 / 46.72 | 14600 | 0 / 0 |
| native-pd/online/8k_near | 1781.18 | 145.38 | 0.15 | 64 | 72.00 | 65.34 / 46.52 | 13838 | 0 / 0 |
| native-pd/online/8k_over | 4251.92 | 138.31 | 0.26 | 64 | 72.00 | 65.31 / 47.94 | 14990 | 0 / 0 |
| native-static-3gpu/p1_d2/16k_near | 3661.14 | 231.60 | 0.02 | 64 | 144.00 | 119.93 / 98.86 | 28239 | 0 / 0 |
| native-static-3gpu/p1_d2/16k_over | 9665.38 | 231.24 | 0.06 | 64 | 144.00 | 113.14 / 93.23 | 27666 | 0 / 0 |
| native-static-3gpu/p1_d2/8k_near | 1798.43 | 132.97 | 0.00 | 64 | 72.00 | 67.32 / 50.41 | 15845 | 0 / 0 |
| native-static-3gpu/p2_d1/16k_near | 954.82 | 350.02 | 0.08 | 64 | 144.00 | 146.76 / 102.50 | 29452 | 0 / 0 |
| native-static-3gpu/p2_d1/16k_over | 1137.73 | 328.04 | 0.06 | 64 | 144.00 | 137.93 / 94.55 | 28347 | 0 / 0 |
| native-static-3gpu/p2_d1/8k_near | 438.57 | 193.48 | 0.12 | 64 | 72.00 | 78.60 / 51.31 | 13566 | 0 / 0 |
| native-static-3gpu/p_tp2_d1/16k_near | 906.60 | 646.83 | 0.20 | 128 | 144.00 | 207.22 / 122.60 | 36864 | 0 / 0 |
| native-static-3gpu/p_tp2_d1/16k_over | 1427.93 | 630.90 | 0.14 | 128 | 144.00 | 201.89 / 122.19 | 36864 | 0 / 0 |
| native-static-3gpu/p_tp2_d1/8k_near | 408.66 | 739.24 | 0.63 | 128 | 72.00 | 136.07 / 59.96 | 18432 | 0 / 0 |
| native-steady/online/16k_near | 3663.51 | 232.16 | 0.07 | 64 | 144.00 | 114.98 / 92.19 | 27673 | 0 / 0 |
| native-steady/online/16k_steady_clean_after | 9675.87 | 234.32 | 0.05 | 64 | 144.00 | 119.39 / 91.91 | 27965 | 0 / 0 |
| native-steady/online/16k_steady_clean_before | 9605.40 | 233.44 | 0.05 | 64 | 144.00 | 116.95 / 94.54 | 27959 | 0 / 0 |
| native-steady/online/mixed | 1626.74 | 380.59 | 0.00 | 24 | 39.09 | 90.35 / 65.05 | 19126 | 0 / 0 |
| native-steady/online/mixed_heavy | 4085.32 | 406.37 | 0.08 | 64 | 162.00 | 136.78 / 114.94 | 31804 | 0 / 0 |

## Physical connector layout audit

The experimental cross-layer flag records a request. The table reports the actual allocated shared tensor from the local worker RPC. The NIXL compatibility hashes in the JSON use the transfer topology, including its cross-layer mode.

| Native start | Role | Logged runner | Connector / layout | Connector prefers cross-layer | Allocated cross-layer tensor | Shape | Stride |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dense-controls/pd-layout-probe | P0 | V2 | HTTP 500 | — | — | — | — |
| dense-controls/pd-layout-probe | D0 | V2 | HTTP 500 | — | — | — | — |
| native-p-only/online | P0 | V2 | NixlPullConnector / HND | False | False | — | — |
| native-p-only/online | D0 | V2 | NixlPullConnector / HND | False | False | — | — |

## Scheduler capacity and deferred waits

`capacity` counts requests in the ordinary waiting queue. `deferred` counts skipped requests, including transient constraints such as remote KV receipt. A capacity wait alone leaves its scheduler cause open. The mean gauges are weighted over the common ROI and KV values are sampled peaks of allocated cache-pool utilization. The preemption counters span the full finite cohort.

| Run | P capacity / deferred wait | D capacity / deferred wait | P / D KV peak (sampled) | P / D preemptions |
| --- | ---: | ---: | ---: | ---: |
| dense-controls/pd-cross/16k_near | 2.245 / 0.000 | 0.000 / 0.038 | 11.72% / 4.69% | 0 / 0 |
| dense-controls/pd-cross/16k_over | 6.658 / 0.000 | 0.000 / 0.055 | 11.72% / 4.69% | 0 / 0 |
| dense-controls/pd-cross/8k_near | 1.884 / 0.000 | 0.000 / 0.097 | 9.38% / 2.35% | 0 / 0 |
| dense-controls/pd-deflect/16k_near | 0.393 / 0.000 | 0.019 / 0.056 | 14.06% / 14.07% | 0 / 0 |
| dense-controls/pd-deflect/16k_over | 2.988 / 0.000 | 0.026 / 0.053 | 14.06% / 9.38% | 0 / 0 |
| dense-controls/pd-deflect/8k_near | 0.618 / 0.000 | 0.000 / 0.148 | 9.38% / 4.70% | 0 / 0 |
| dense-controls/pd-layout-probe/8k_low | 0.000 / 0.000 | 0.000 / 0.030 | 4.69% / 2.34% | 0 / 0 |
| native-p-only/online/16k_p_only_clean_after | 6.679 / 0.000 | 0.000 / 0.104 | 11.72% / 4.69% | 0 / 0 |
| native-p-only/online/16k_p_only_clean_before | 6.608 / 0.000 | 0.000 / 0.027 | 11.72% / 4.69% | 0 / 0 |
| native-p-only/online/16k_p_only_profile | 6.652 / 0.000 | 0.000 / 0.027 | 11.72% / 4.69% | 0 / 0 |
| native-static-3gpu/p1_d2/16k_near | 2.269 / 0.000 | 0.000 / 0.019 | 11.72% / 4.69% | 0 / 0 |
| native-static-3gpu/p1_d2/16k_over | 6.683 / 0.000 | 0.000 / 0.057 | 11.72% / 4.69% | 0 / 0 |
| native-static-3gpu/p1_d2/8k_near | 1.987 / 0.000 | 0.000 / 0.000 | 9.38% / 2.35% | 0 / 0 |
| native-static-3gpu/p2_d1/16k_near | 0.038 / 0.000 | 0.000 / 0.076 | 11.72% / 9.38% | 0 / 0 |
| native-static-3gpu/p2_d1/16k_over | 0.153 / 0.000 | 0.000 / 0.060 | 14.06% / 9.38% | 0 / 0 |
| native-static-3gpu/p2_d1/8k_near | 0.000 / 0.000 | 0.000 / 0.122 | 7.03% / 4.70% | 0 / 0 |
| native-static-3gpu/p_tp2_d1/16k_near | 0.186 / 0.000 | 0.000 / 0.204 | 9.38% / 18.76% | 0 / 0 |
| native-static-3gpu/p_tp2_d1/16k_over | 0.712 / 0.000 | 0.000 / 0.142 | 9.38% / 14.08% | 0 / 0 |
| native-static-3gpu/p_tp2_d1/8k_near | 0.146 / 0.000 | 0.000 / 0.630 | 5.86% / 16.42% | 0 / 0 |
| native-steady/online/16k_near | 2.219 / 0.000 | 0.000 / 0.074 | 11.72% / 4.69% | 0 / 0 |
| native-steady/online/16k_steady_clean_after | 6.726 / 0.000 | 0.000 / 0.054 | 11.72% / 4.69% | 0 / 0 |
| native-steady/online/16k_steady_clean_before | 6.665 / 0.000 | 0.000 / 0.054 | 11.72% / 4.69% | 0 / 0 |
| native-steady/online/16k_steady_profile | 6.656 / 0.000 | 0.000 / 0.027 | 11.72% / 4.69% | 0 / 0 |
| native-steady/online/mixed | 0.000 / 0.000 | 0.000 / 0.000 | 11.72% / 11.79% | 0 / 0 |
| native-steady/online/mixed_heavy | 1.535 / 0.000 | 0.000 / 0.082 | 18.75% / 9.38% | 0 / 0 |

Mean host-segment durations below use the same successful handoff rows and sum to the mean TTFT of those requests.

| Run | Arrival → client start ms | Client start → P ready ms | P ready → D submit ms | D submit → first token ms | Mean TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense-controls/p-light/16k_light_clean_after | 1.03 | 1605.52 | 0.01 | 194.45 | 1801.00 |
| dense-controls/p-light/16k_light_clean_before | 0.64 | 1582.33 | 0.01 | 185.19 | 1768.18 |
| dense-controls/p-light/16k_near | 0.77 | 2185.31 | 0.01 | 186.99 | 2373.08 |
| dense-controls/p-light/16k_over | 0.97 | 5004.24 | 0.01 | 178.88 | 5184.11 |
| dense-controls/p-light/8k_near_clean_after | 0.85 | 1078.93 | 0.01 | 107.43 | 1187.22 |
| dense-controls/p-light/8k_near_clean_before | 0.83 | 1098.99 | 0.01 | 109.18 | 1209.01 |
| dense-controls/pd-cross/16k_near | 0.78 | 2134.17 | 0.01 | 194.82 | 2329.78 |
| dense-controls/pd-cross/16k_over | 0.85 | 5085.73 | 0.01 | 206.63 | 5293.23 |
| dense-controls/pd-cross/8k_near | 0.70 | 1078.87 | 0.01 | 110.26 | 1189.84 |
| dense-controls/pd-deflect/16k_near | 1.03 | 962.03 | 0.01 | 253.67 | 1216.74 |
| dense-controls/pd-deflect/16k_over | 0.73 | 2491.11 | 0.01 | 244.98 | 2736.83 |
| dense-controls/pd-deflect/8k_near | 0.87 | 633.45 | 0.01 | 129.83 | 764.16 |
| dense-controls/pd-layout-probe/8k_low | 1.03 | 293.95 | 0.13 | 108.59 | 403.71 |
| native-p-only/online/16k_p_only_clean_after | 0.99 | 4994.18 | 0.21 | 195.32 | 5190.70 |
| native-p-only/online/16k_p_only_clean_before | 0.94 | 4988.96 | 0.20 | 192.38 | 5182.48 |
| native-pd/online/16k_low | 6.82 | 673.75 | 0.01 | 232.79 | 913.38 |
| native-pd/online/16k_near | 8.40 | 2153.42 | 0.01 | 180.61 | 2342.44 |
| native-pd/online/16k_o64 | 7.57 | 1056.85 | 0.01 | 192.73 | 1257.16 |
| native-pd/online/16k_over | 4.27 | 5002.36 | 0.01 | 206.56 | 5213.21 |
| native-pd/online/16k_trace_clean_after | 4.36 | 1592.37 | 0.01 | 183.02 | 1779.76 |
| native-pd/online/16k_trace_clean_before | 2.71 | 1604.81 | 0.01 | 201.95 | 1809.48 |
| native-pd/online/2k_o256_decode | 4.00 | 71.33 | 0.01 | 61.28 | 136.61 |
| native-pd/online/32k_low | 1.04 | 1713.07 | 0.01 | 376.19 | 2090.32 |
| native-pd/online/32k_near | 1.19 | 2822.56 | 0.01 | 328.66 | 3152.43 |
| native-pd/online/32k_over | 3.20 | 4449.76 | 0.01 | 318.94 | 4771.92 |
| native-pd/online/4k_o64_mix | 6.30 | 141.12 | 0.01 | 71.66 | 219.09 |
| native-pd/online/8k_low | 7.55 | 293.31 | 0.01 | 110.79 | 411.66 |
| native-pd/online/8k_near | 2.89 | 1070.38 | 0.01 | 114.39 | 1187.67 |
| native-pd/online/8k_over | 5.11 | 2225.62 | 0.01 | 114.73 | 2345.47 |
| native-static-3gpu/p1_d2/16k_near | 0.88 | 2119.53 | 0.18 | 190.37 | 2310.97 |
| native-static-3gpu/p1_d2/16k_over | 0.96 | 5064.09 | 0.19 | 181.85 | 5247.09 |
| native-static-3gpu/p1_d2/8k_near | 0.76 | 1081.37 | 0.13 | 114.53 | 1196.80 |
| native-static-3gpu/p2_d1/16k_near | 1.22 | 633.37 | 0.01 | 234.97 | 869.56 |
| native-static-3gpu/p2_d1/16k_over | 0.83 | 708.20 | 0.01 | 238.78 | 947.82 |
| native-static-3gpu/p2_d1/8k_near | 0.81 | 283.05 | 0.01 | 152.73 | 436.60 |
| native-static-3gpu/p_tp2_d1/16k_near | 0.85 | 538.84 | 0.19 | 424.29 | 964.17 |
| native-static-3gpu/p_tp2_d1/16k_over | 0.90 | 713.00 | 0.19 | 433.63 | 1147.71 |
| native-static-3gpu/p_tp2_d1/8k_near | 0.80 | 251.43 | 0.12 | 371.07 | 623.42 |
| native-steady/online/16k_near | 0.95 | 2121.77 | 0.20 | 181.01 | 2303.93 |
| native-steady/online/16k_steady_clean_after | 0.97 | 5073.33 | 0.16 | 190.49 | 5264.95 |
| native-steady/online/16k_steady_clean_before | 0.83 | 5006.16 | 0.19 | 185.71 | 5192.88 |
| native-steady/online/mixed | 0.71 | 641.70 | 0.15 | 210.88 | 853.44 |
| native-steady/online/mixed_heavy | 0.78 | 2053.15 | 0.19 | 209.37 | 2263.48 |

## Same-window motivation gate

| Profile | Evidence | P kernel busy | D kernel busy | P active and D kernel inactive | P waited | P ROI rate / fastest finite P witness | Decode SLO | Overall gate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| dense-controls/p-light/16k_light_profile | TRACE_ANALYSIS | 81.29% | 14.13% | 68.75% | 31.53% | 84.79% | 24 / 24 | INCONCLUSIVE_SERVICE_AND_CREDIT_DECOMPOSITION |
| native-p-only/online/16k_p_only_profile | DIAGNOSTIC | 92.52% | — | — | 78.61% | 94.24% | 64 / 64 | DIAGNOSTIC_PROFILE |
| native-pd/online/16k_trace_profile | DIAGNOSTIC | 81.29% | 13.96% | 68.73% | 28.08% | 84.79% | 22 / 24 | DIAGNOSTIC_PROFILE |
| native-steady/online/16k_steady_profile | DIAGNOSTIC | 92.62% | 16.30% | 77.28% | 78.40% | 94.24% | 64 / 64 | DIAGNOSTIC_PROFILE |

16k_light [TRACE_ANALYSIS] records simultaneous P/D kernel activity for 12.54% of the ROI. The kernel-only collector excludes memory-copy events. Captured intervals do not resolve interruption-safe time.

16k_light recorded 27440.5 P input tokens/s across the central window, 84.79% of the fastest same-input finite P witness. The ROI mean sampled D capacity/deferred gauges were —/—. P/D peak sampled cache occupancy was 9.38%/0.00%.

16k_p_only captured P kernels only. D kernel occupancy and P/D temporal intersection are unavailable for this arm.

16k_trace [DIAGNOSTIC] records simultaneous P/D kernel activity for 12.56% of the ROI. P was active with no recorded D kernel, copy or memset for 58.68%. Captured intervals do not resolve interruption-safe time.

16k_steady [DIAGNOSTIC] records simultaneous P/D kernel activity for 15.34% of the ROI. The kernel-only collector excludes memory-copy events. Captured intervals do not resolve interruption-safe time.

A qualifying P busy interval alone establishes kernel occupancy. A stricter prefill-heavy claim also requires independently matched P service rate, P-ready backlog and exclusion of D KV admission and transfer credit stalls. The ROI rate counts prefill completions inside the central window, while the independent witness uses its finite drain. D kernel inactivity leaves DMA, hardware counter headroom and future deadline safety unestablished.

Regenerate both reports and figures from the native raw directory with `python poc_flashvep/pd_resource_complementary_parallelism/analysis_native.py --raw-root <runs> --out poc_flashvep/pd_resource_complementary_parallelism/results/native --reports poc_flashvep/reports --figures`. The plots and streaming trace reductions were generated with Matplotlib 3.11.2 and ijson 3.6.0 in a separate CPU environment.

Reducer outputs include complete request counts, all configured percentiles, grouped SSE events, P and D host-boundary times and Prometheus deltas. [Native reduction data](../pd_resource_complementary_parallelism/results/native/native_summary.json)

[native pd kernel busy](../pd_resource_complementary_parallelism/results/native/figures/native_pd_kernel_busy.svg) [native pd operating region](../pd_resource_complementary_parallelism/results/native/figures/native_pd_operating_region.svg) [native qps vs decode tails](../pd_resource_complementary_parallelism/results/native/figures/native_qps_vs_decode_tails.svg) [native qps vs goodput](../pd_resource_complementary_parallelism/results/native/figures/native_qps_vs_goodput.svg) [native qps vs ttft](../pd_resource_complementary_parallelism/results/native/figures/native_qps_vs_ttft.svg)
