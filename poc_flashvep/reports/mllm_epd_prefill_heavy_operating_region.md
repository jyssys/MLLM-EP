# MLLM EPD prefill-heavy operating region

**Motivation assessment** `INCONCLUSIVE_PROFILING`

Native online complete cohorts 74/74. Observer-qualified three-role captures 0/6. Same-window M1 passes 0.

6 diagnostic captures satisfy the non-observer M1 checks in one window. Their physical measurements remain conditional evidence because the matched latency guard failed.

## Contract

The reducer uses one monotonic interval from the scheduled arrival at index floor(N×.20) to floor(N×.80). These endpoints stay inside the offered load and exclude drain. The cohort-level serving Goodput denominator runs from the first scheduled arrival to the last completed response. P service proximity divides P-ready completion tokens per ROI second by the strongest finite independent P reference for that cohort. Because completions are discrete and request sizes differ, a short-window ratio can exceed 100% and does not by itself establish saturation. The Decode SLO in the trace applies to successful decode outputs entirely inside this ROI. The online row also preserves all-cohort absolute decode success and joint SLO.

Observer qualification uses duration, TTFT p50/p95, TPOT p95 and ITL p99. Both the relative clean spread and the profiled deviation from the clean center must be at most 5% on each metric. Request order, prompt contract, interarrival offsets and greedy tokens must match. Each role must record valid CUPTI events across the entire selected window. NVML SM activity is an independent weighted device counter. It can guide load selection. The M1 decision requires physical kernel spans.

## Actual online sweep

| Epoch and cohort | QPS | Median images | Median LM input | Median output | Median visual share | E/P/D SM % | P service proximity | Decode SLO | TTFT p95 ms | Joint req/s | Observer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `calibration-sweep/mix_q072` | 1.894 | 18.0 | 13,461 | 16.0 | 99.39% | 74.1/74.0/6.2 | 70.23% | 24/24 | 3,272.8 | 2.208 | clean |
| `calibration-sweep/mix_q096` | 2.525 | 18.0 | 13,461 | 16.0 | 99.39% | 76.7/81.3/8.3 | 77.08% | 24/24 | 5,528.4 | 1.696 | clean |
| `calibration-sweep/mix_q120` | 3.157 | 18.0 | 13,461 | 16.0 | 99.39% | 88.2/75.0/8.4 | 62.52% | 23/24 | 5,667.5 | 1.540 | clean |
| `calibration-sweep/mix_q148` | 3.893 | 18.0 | 13,461 | 16.0 | 99.39% | 89.4/76.2/9.7 | 77.10% | 24/24 | 6,559.0 | 1.619 | clean |
| `calibration-sweep/mix_o64` | 2.499 | 18.0 | 13,461 | 64.0 | 99.39% | 78.3/80.8/26.4 | 76.29% | 24/24 | 5,473.9 | 1.939 | clean |
| `calibration-sweep/mix_o256` | 2.499 | 18.0 | 13,461 | 256.0 | 99.39% | 77.4/80.1/62.8 | 76.29% | 19/24 | 5,454.4 | 1.286 | clean |
| `calibration-sweep/long_q072` | 0.879 | 42.5 | 24,790 | 16.0 | 99.51% | 62.9/80.7/5.7 | 83.52% | 24/24 | 5,775.7 | 0.896 | clean |
| `calibration-sweep/long_q096` | 1.172 | 42.5 | 24,790 | 16.0 | 99.51% | 82.0/92.6/6.2 | 91.14% | 23/24 | 8,220.4 | 0.673 | clean |
| `calibration-sweep/long_q120` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.7/93.3/5.9 | 103.27% | 24/24 | 10,990.1 | 0.433 | clean |
| `calibration-sweep/long_q148` | 1.807 | 42.5 | 24,790 | 16.0 | 99.51% | 97.2/87.0/5.3 | 77.20% | 24/24 | 12,483.3 | 0.238 | clean |
| `calibration-sweep/upper_q072` | 1.085 | 33.5 | 20,242 | 16.0 | 99.50% | 55.3/68.7/5.7 | 78.19% | 24/24 | 5,644.1 | 0.947 | clean |
| `calibration-sweep/upper_q096` | 1.446 | 33.5 | 20,242 | 16.0 | 99.50% | 71.1/85.5/7.3 | 101.63% | 24/24 | 7,601.1 | 0.960 | clean |
| `calibration-sweep/upper_q120` | 1.808 | 33.5 | 20,242 | 16.0 | 99.50% | 87.1/86.9/6.7 | 106.62% | 24/24 | 8,493.0 | 0.889 | clean |
| `calibration-sweep/upper_q148` | 2.230 | 33.5 | 20,242 | 16.0 | 99.50% | 95.8/82.8/5.1 | 95.13% | 24/24 | 9,464.7 | 0.710 | clean |
| `controlled-sweep/v25_q082` | 1.826 | 7.0 | 16,384 | 16.0 | 24.72% | 23.1/88.1/10.1 | 94.69% | 24/24 | 3,277.6 | 2.075 | clean |
| `controlled-sweep/v25_q112` | 2.494 | 7.0 | 16,384 | 16.0 | 24.72% | 27.6/92.4/10.2 | 105.81% | 24/24 | 5,525.9 | 1.626 | clean |
| `controlled-sweep/v25_q140` | 3.117 | 7.0 | 16,384 | 16.0 | 24.72% | 32.0/94.3/10.5 | 102.86% | 24/24 | 6,233.8 | 1.595 | clean |
| `controlled-sweep/v50_q082` | 1.797 | 13.0 | 16,384 | 16.0 | 49.44% | 37.2/90.4/11.6 | 94.84% | 24/24 | 3,173.1 | 2.071 | clean |
| `controlled-sweep/v50_q112` | 2.454 | 13.0 | 16,384 | 16.0 | 49.44% | 48.9/95.0/10.4 | 105.97% | 24/24 | 5,043.7 | 1.947 | clean |
| `controlled-sweep/v50_q140` | 3.068 | 13.0 | 16,384 | 16.0 | 49.44% | 62.3/95.3/10.1 | 103.04% | 24/24 | 6,185.5 | 1.592 | clean |
| `controlled-sweep/v50_o32` | 2.454 | 13.0 | 16,384 | 32.0 | 49.44% | 50.6/95.2/20.6 | 105.97% | 24/24 | 4,997.6 | 1.941 | clean |
| `controlled-sweep/v50_o128` | 2.454 | 13.0 | 16,384 | 128.0 | 49.44% | 51.7/95.9/61.3 | 105.97% | 24/24 | 5,001.7 | 1.860 | clean |
| `controlled-sweep/v75_q082` | 1.771 | 20.0 | 16,384 | 16.0 | 75.20% | 58.2/86.6/10.4 | 94.81% | 24/24 | 3,450.3 | 2.001 | clean |
| `controlled-sweep/v75_q112` | 2.419 | 20.0 | 16,384 | 16.0 | 75.20% | 71.3/93.5/9.5 | 105.95% | 24/24 | 5,277.1 | 1.886 | clean |
| `controlled-sweep/v75_q140` | 3.024 | 20.0 | 16,384 | 16.0 | 75.20% | 84.1/91.6/10.6 | 103.02% | 24/24 | 6,893.4 | 1.396 | clean |
| `controlled-sweep/v95_q082` | 1.754 | 25.0 | 16,384 | 16.0 | 93.93% | 68.4/86.5/9.8 | 103.46% | 24/24 | 3,627.5 | 1.988 | clean |
| `controlled-sweep/v95_q112` | 2.395 | 25.0 | 16,384 | 16.0 | 93.93% | 91.5/92.3/10.4 | 105.95% | 24/24 | 5,465.8 | 1.686 | clean |
| `controlled-sweep/v95_q140` | 2.994 | 25.0 | 16,384 | 16.0 | 93.93% | 97.4/90.4/8.2 | 103.02% | 24/24 | 6,629.0 | 1.441 | clean |
| `controlled-sweep/triplet_q112` | 2.476 | 3.0 | 16,384 | 16.0 | 50.01% | 55.9/91.9/11.9 | 105.98% | 24/24 | 5,210.4 | 1.926 | clean |
| `controlled-sweep/triplet_q140` | 3.095 | 3.0 | 16,384 | 16.0 | 50.01% | 66.1/95.2/9.4 | 103.04% | 24/24 | 6,245.4 | 1.586 | clean |
| `controlled-sweep/tiny_images_q112` | 2.461 | 13.0 | 16,384 | 16.0 | 49.44% | 48.0/95.1/11.3 | 94.19% | 24/24 | 5,094.8 | 1.944 | clean |
| `controlled-sweep/tiny_images_q140` | 3.077 | 13.0 | 16,384 | 16.0 | 49.44% | 58.7/90.1/10.9 | 103.04% | 24/24 | 6,734.3 | 1.353 | clean |
| `controlled-sweep/long_few_q112` | 1.568 | 5.0 | 23,350 | 16.0 | 99.83% | 98.0/91.3/6.9 | 94.19% | 24/24 | 9,039.9 | 0.596 | clean |
| `controlled-sweep/long_few_q140` | 1.960 | 5.0 | 23,350 | 16.0 | 99.83% | 98.2/77.1/4.8 | 72.48% | 24/24 | 11,390.5 | 0.260 | clean |
| `controlled-sweep/long_many_q112` | 1.550 | 40.0 | 23,168 | 16.0 | 99.47% | 89.2/92.7/6.5 | 94.11% | 24/24 | 8,361.7 | 0.649 | clean |
| `controlled-sweep/long_many_q140` | 1.938 | 40.0 | 23,168 | 16.0 | 99.47% | 94.6/89.6/7.6 | 91.35% | 24/24 | 10,344.9 | 0.486 | clean |
| `deflection-chunk4096/inline_native_n24` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.1/90.3/5.9 | 90.43% | 24/24 | 10,602.3 | 0.431 | clean |
| `deflection-chunk4096/native_eos_n64` | 2.955 | 3.0 | 9,867 | 24.5 | 99.40% | 70.7/86.5/43.9 | 100.67% | 63/64 | 3,557.2 | 2.870 | clean |
| `deflection-confirm/confirm_native_n64` | 1.465 | 41.0 | 24,790 | 16.0 | 99.52% | 89.4/92.0/6.2 | 98.63% | 64/64 | 14,716.8 | 0.131 | clean |
| `deflection-inline/inline_native_n24` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.4/90.5/5.1 | 89.67% | 24/24 | 10,518.8 | 0.384 | clean |
| `deflection-inline/mixed_native_n24` | 2.600 | 20.0 | 16,384 | 16.0 | 62.47% | 39.3/93.9/9.3 | — | 24/24 | 5,414.1 | 1.843 | clean |
| `image-ready/image_ready_n24` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 86.4/79.6/4.2 | 76.18% | 24/24 | 12,428.4 | 0.277 | native image-ready polling |
| `late-native/v25_primed_n24_clean_before` | 2.494 | 7.0 | 16,384 | 16.0 | 24.72% | 27.1/94.7/11.2 | 105.81% | 24/24 | 5,056.2 | 1.964 | clean |
| `late-native/v25_primed_n24_profile` | 2.494 | 7.0 | 16,384 | 16.0 | 24.72% | 26.8/95.9/11.4 | 105.81% | 24/24 | 4,898.6 | 2.073 | cupti |
| `late-native/v25_primed_n24_clean_after` | 2.494 | 7.0 | 16,384 | 16.0 | 24.72% | 28.1/95.0/11.1 | 105.81% | 24/24 | 4,971.2 | 1.977 | clean |
| `late-native/long_scheduled_burst24` | 1.679 | 42.5 | 24,790 | 16.0 | 99.51% | 86.4/86.6/5.5 | 89.81% | 24/24 | 7,523.2 | 0.325 | clean |
| `length-capacity/size4096_q085` | 6.929 | 3.0 | 4,096 | 16.0 | 49.00% | 31.9/75.0/26.0 | 107.24% | 24/24 | 1,186.2 | 7.368 | clean |
| `length-capacity/size4096_q115` | 9.374 | 3.0 | 4,096 | 16.0 | 49.00% | 43.8/76.7/23.8 | 96.72% | 24/24 | 1,292.5 | 8.203 | clean |
| `length-capacity/size8192_q085` | 3.930 | 7.0 | 8,192 | 16.0 | 49.44% | 41.1/83.4/20.7 | 98.29% | 24/24 | 1,571.6 | 4.393 | clean |
| `length-capacity/size8192_q115` | 5.317 | 7.0 | 8,192 | 16.0 | 49.44% | 53.8/89.8/19.7 | 108.80% | 24/24 | 2,446.1 | 4.473 | clean |
| `length-capacity/size24576_q085` | 1.138 | 20.0 | 24,576 | 16.0 | 50.13% | 40.2/95.3/6.8 | 98.26% | 24/24 | 5,150.4 | 1.195 | clean |
| `length-capacity/size24576_q115` | 1.540 | 20.0 | 24,576 | 16.0 | 50.13% | 49.4/95.9/6.3 | 96.71% | 24/24 | 8,446.8 | 0.812 | clean |
| `length-capacity/size32512_q085` | 0.796 | 26.0 | 32,512 | 16.0 | 49.26% | 39.2/93.6/4.5 | 98.33% | 24/24 | 8,079.8 | 0.593 | clean |
| `length-capacity/size32512_q115` | 1.077 | 26.0 | 32,512 | 16.0 | 49.26% | 44.9/91.1/4.4 | 96.76% | 24/24 | 12,428.5 | 0.264 | clean |
| `native-population/full_main400_eos` | 1.600 | 3.0 | 9,804 | 25.0 | 99.39% | 39.9/45.4/23.8 | — | 400/400 | 3,030.5 | 1.826 | clean |
| `natural-confirmation/long_confirm24_clean_before` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 86.0/78.0/4.3 | 89.02% | 24/24 | 12,559.1 | 0.273 | clean |
| `natural-confirmation/long_confirm24_profile` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.4/92.8/5.6 | 103.27% | 24/24 | 10,335.4 | 0.434 | cupti |
| `natural-confirmation/long_confirm24_clean_after` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 96.5/92.9/5.0 | 90.43% | 24/24 | 10,361.6 | 0.434 | clean |
| `natural-confirmation/long_persistence64` | 1.465 | 41.0 | 24,790 | 16.0 | 99.52% | 88.9/81.4/5.3 | 89.54% | 64/64 | 17,001.5 | 0.161 | clean |
| `natural-confirmation/long_persistence128` | 1.465 | 42.5 | 24,816 | 16.0 | 99.52% | 86.6/91.8/6.0 | 97.80% | 127/128 | 30,076.2 | 0.086 | clean |
| `natural-confirmation/mainupper_n64` | 1.894 | 27.0 | 20,073 | 16.0 | 99.51% | 92.4/89.9/8.0 | 109.90% | 64/64 | 11,900.9 | 0.386 | clean |
| `natural-confirmation/mainmix_eos_n64` | 2.955 | 3.0 | 9,867 | 24.5 | 99.40% | 70.0/87.6/45.2 | 100.67% | 60/64 | 3,141.8 | 2.739 | clean |
| `natural-retry/long_primed_n24_clean_before` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.5/92.3/4.9 | 90.43% | 24/24 | 10,488.9 | 0.434 | clean |
| `natural-retry/long_primed_n24_profile` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 96.5/92.5/5.7 | 103.27% | 24/24 | 10,477.6 | 0.334 | cupti |
| `natural-retry/long_primed_n24_clean_after` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.7/91.0/5.7 | 103.27% | 23/24 | 10,548.2 | 0.382 | clean |
| `nixl-ready/nixl_readiness_n24_clean_before` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.7/91.0/5.0 | 90.43% | 24/24 | 10,500.4 | 0.431 | native NIXL READ observation |
| `nixl-ready/nixl_readiness_n24_profile` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.5/91.3/4.7 | 90.43% | 24/24 | 10,573.4 | 0.430 | native NIXL READ observation |
| `nixl-ready/nixl_readiness_n24_clean_after` | 1.465 | 42.5 | 24,790 | 16.0 | 99.51% | 97.7/91.6/5.2 | 90.43% | 24/24 | 10,392.9 | 0.431 | native NIXL READ observation |
| `observer-long64/long_primed_n64_clean_before` | 1.465 | 41.0 | 24,790 | 16.0 | 99.52% | 89.1/89.8/5.8 | 95.79% | 64/64 | 14,955.9 | 0.149 | clean |
| `observer-long64/long_primed_n64_profile` | 1.465 | 41.0 | 24,790 | 16.0 | 99.52% | 88.9/86.2/5.8 | 95.79% | 64/64 | 15,620.4 | 0.165 | cupti |
| `observer-long64/long_primed_n64_clean_after` | 1.465 | 41.0 | 24,790 | 16.0 | 99.52% | 89.2/90.6/6.4 | 99.50% | 64/64 | 14,712.7 | 0.168 | clean |
| `observer-upper48/upper_48g_n64_clean_before` | 1.894 | 27.0 | 20,073 | 16.0 | 99.51% | 90.2/92.8/7.7 | 107.64% | 63/64 | 12,561.4 | 0.377 | clean |
| `observer-upper48/upper_48g_n64_profile` | 1.894 | 27.0 | 20,073 | 16.0 | 99.51% | 91.6/90.6/7.7 | 110.39% | 64/64 | 11,327.8 | 0.387 | cupti |
| `observer-upper48/upper_48g_n64_clean_after` | 1.894 | 27.0 | 20,073 | 16.0 | 99.51% | 91.0/90.9/7.6 | 107.94% | 62/64 | 11,237.5 | 0.391 | clean |

## Text-only backbone control

The same checkpoint and BF16 native workers serve these no-image prompts through the 1P/1D path. They skip the encoder and keep the model's original LM forward and NIXL handoff.

| Epoch and cohort | Offered req/s | Median LM input | P/D SM % | P service proximity | Decode SLO | TTFT p95 ms | Joint req/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `controlled-sweep/v00_q082` | 1.867 | 16,384 | 92.2/9.8 | 94.83% | 24/24 | 2,810.5 | 2.201 |
| `controlled-sweep/v00_q112` | 2.550 | 16,384 | 97.7/10.8 | 94.20% | 24/24 | 4,632.2 | 2.243 |
| `controlled-sweep/v00_q140` | 3.188 | 16,384 | 97.8/11.9 | 103.04% | 24/24 | 5,785.5 | 1.771 |
| `late-native/text_e_ready_rate_poisson` | 3.057 | 16,384 | 97.7/12.1 | 98.81% | 24/24 | 5,619.8 | 1.857 |
| `late-native/text_e_ready_replayed` | 3.057 | 16,384 | 97.7/11.3 | 99.14% | 24/24 | 2,737.2 | 2.239 |

[Controlled visual-ratio activity](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/controlled_visual_ratio_sm_activity.svg)

## Mixed multimodal and text traffic

The native routing control submits image prompts to E before P. Text prompts reach P directly. Both use the same D worker and native NIXL.

| Epoch and cohort | Offered req/s | Completed image | Completed text | Completed | E/P/D SM % | Decode SLO | TTFT p95 ms | Joint req/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `late-native/mixed_multimodal_text_retry` | 1.800 | 24 | 24 | 48/48 | 32.9/67.2/16.4 | 48/48 | 3,389.5 | 2.056 |
| `native-population/mixed_multimodal_text` | 1.800 | 23 | 24 | 47/48 | 32.6/71.4/17.9 | 47/48 | 5,246.8 | 1.805 |

## Profiled observations and decision

| Epoch and cohort | E/P/D kernel busy | E/P/D physical region | Evidence | ROI with a recent P-ready sample | P service proximity | Same-window Decode SLO | M1 missing or unmet |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `late-native/v25_primed_n24_profile` | 31.54%/97.48%/18.41% | R2 | DIAGNOSTIC | 97.49% | 105.81% | 9/9 | same_window_observer_pass |
| `natural-confirmation/long_confirm24_profile` | 98.32%/95.50%/9.46% | R3 | DIAGNOSTIC | 86.25% | 103.27% | 7/7 | same_window_observer_pass |
| `natural-retry/long_primed_n24_profile` | 97.20%/95.06%/9.17% | R3 | DIAGNOSTIC | 76.12% | 103.27% | 7/7 | same_window_observer_pass |
| `nixl-ready/nixl_readiness_n24_profile` | 98.31%/94.03%/8.65% | R3 | DIAGNOSTIC | 89.11% | 90.43% | 7/7 | same_window_observer_pass |
| `observer-long64/long_primed_n64_profile` | 90.04%/89.91%/9.55% | R3 | DIAGNOSTIC | 82.77% | 95.79% | 30/30 | same_window_observer_pass |
| `observer-upper48/upper_48g_n64_profile` | 92.27%/93.30%/12.99% | R3 | DIAGNOSTIC | 77.29% | 110.39% | 32/32 | same_window_observer_pass |

## Profiling observer check

Each metric passes only when both (A) the clean-before/after spread and (B) the profiled movement from the clean center stay within 5%. Cells report A/B. The threshold also requires exact outputs, complete request accounting and an intact three-role trace.

| Epoch and cohort | Duration | TTFT p50 | TTFT p95 | TPOT p95 | ITL p99 | Gate |
| --- | --- | --- | --- | --- | --- | --- |
| `late-native/v25_primed_n24_profile` | 0.65%/0.63% | 3.09%/4.28% | 1.70%/2.30% | 13.47%/42.87% | 21.10%/11.00% | FAIL |
| `natural-confirmation/long_confirm24_profile` | 5.60%/2.86% | 15.23%/8.47% | 19.17%/9.82% | 5.25%/16.80% | 5.91%/14.96% | FAIL |
| `natural-retry/long_primed_n24_profile` | 0.83%/0.51% | 0.24%/5.58% | 0.56%/0.39% | 1.28%/17.14% | 26.58%/7.30% | FAIL |
| `nixl-ready/nixl_readiness_n24_profile` | 0.03%/0.21% | 1.55%/1.57% | 1.03%/1.21% | 6.01%/16.37% | 12.45%/1.27% | AUXILIARY_OBSERVER_NOT_QUALIFIED |
| `observer-long64/long_primed_n64_profile` | 0.51%/1.49% | 3.00%/9.20% | 1.64%/5.30% | 6.28%/20.20% | 1.37%/7.95% | FAIL |
| `observer-upper48/upper_48g_n64_profile` | 3.65%/0.88% | 1.23%/9.87% | 11.13%/4.80% | 8.58%/21.35% | 14.31%/10.75% | FAIL |

The labels classify physical activity only. R1 uses E ≥80%, P <80%, D ≤30%. R2 uses E <80%, P ≥80%, D ≤30%. R3 uses E and P ≥80%, D ≤30%. R4 uses E and P ≥80%, D ≥50%. R5 uses D ≥50% and D at least as busy as either upstream stage. Other measured combinations remain outside these bands. The primary M1 gate additionally requires a ready backlog, proximity, decode SLO, and usable timing evidence from the same capture. Natural versus controlled scope follows the frozen cohort receipts.

[Reduced cohort and observation records](../mllm_epd_prefill_heavy_modality_discovery/results/region/operating_region.json) · [Online cohort table](../mllm_epd_prefill_heavy_modality_discovery/results/region/online_cohorts.csv)

[Figure A](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/figure_a_operating_region.svg)
