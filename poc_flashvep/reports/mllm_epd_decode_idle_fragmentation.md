# Decode idle fragmentation

Kernel idle is the complement of the aligned D kernel union. Proxy request-free intervals require no request between P-ready and D HTTP completion. The recent-empty engine measure intersects observed kernel idle with bounded scheduler samples of an empty queue and no running requests. It covers only the time reported by scheduler samples and can miss a truly quiet engine between schedules. An interval with no CUPTI copy activity can still include DMA handled by the native connector. Native transfer-free and deadline-safe availability require separate evidence. All raw capture classifications below retain their observed/diagnostic labels.

## Same-window idle decomposition

| Epoch and cohort | D kernel idle | Also proxy request free | Also sampled engine empty | Also CUPTI copy-free | Also outside accounted NIXL READ | P busy and D kernel idle | P busy and D proxy-quiet | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `late-native/v25_primed_n24_profile` | 81.59% | 67.48% | — | — | — | 79.15% | 65.09% | DIAGNOSTIC |
| `natural-confirmation/long_confirm24_profile` | 90.54% | 80.59% | 0.00% | — | — | 86.25% | 76.35% | DIAGNOSTIC |
| `natural-retry/long_primed_n24_profile` | 90.83% | 82.55% | 0.00% | — | — | 85.93% | 77.67% | DIAGNOSTIC |
| `nixl-ready/nixl_readiness_n24_profile` | 91.35% | 81.36% | — | 89.41% | 88.18% | 85.81% | 75.95% | DIAGNOSTIC |
| `observer-long64/long_primed_n64_profile` | 90.45% | 81.38% | — | — | — | 80.49% | 71.48% | DIAGNOSTIC |
| `observer-upper48/upper_48g_n64_profile` | 87.01% | 74.62% | — | — | — | 80.41% | 68.12% | DIAGNOSTIC |

## Observable idle causes

In `observer-long64/long_primed_n64_profile`, 81.38% of the window is both kernel-idle and without eligible D proxy work. The overlap with an active E HTTP request is 81.38% and with an active P HTTP request 81.38%. Kernel idle inside an active D HTTP request covers 9.03%. Upstream HTTP overlap shows where requests are pending in the pipeline. E and P can overlap each other and the two shares can describe different requests.


## Fragment length and hypothetical work fit

The detail view prioritizes a same-window M2 pass, then M1, then the largest qualified capture. The largest diagnostic is used when there is no qualified capture. Selected `observer-long64/long_primed_n64_profile` · `DIAGNOSTIC`.

| D idle ms | Count | Time s | Share of idle | P overlap s | E overlap s | P-ready overlap s | Next P-ready proxy p50 ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [0, 1) | 135326 | 0.178 | 0.78% | 0.175 | 0.146 | 0.159 | 684.70 |
| [1, 5) | 207 | 0.315 | 1.39% | 0.314 | 0.263 | 0.298 | 696.01 |
| [5, 10) | 0 | 0.000 | 0.00% | 0.000 | 0.000 | 0.000 | — |
| [10, 25) | 0 | 0.000 | 0.00% | 0.000 | 0.000 | 0.000 | — |
| [25, 50) | 0 | 0.000 | 0.00% | 0.000 | 0.000 | 0.000 | — |
| [50, 100) | 0 | 0.000 | 0.00% | 0.000 | 0.000 | 0.000 | — |
| [100, 250) | 0 | 0.000 | 0.00% | 0.000 | 0.000 | 0.000 | — |
| [250, 500) | 5 | 1.954 | 8.60% | 1.774 | 1.772 | 1.691 | 402.55 |
| ≥500 | 26 | 20.281 | 89.24% | 17.962 | 18.430 | 16.460 | 662.53 |

The future P-ready proxy timestamp is retrospective and may follow the selected idle interval. The fit table is an `OFFLINE_ORACLE` over contiguous D-kernel-idle and proxy-request-free intervals. The P-ready column asks if a known compute-ready sample leaves a possible launch time inside the interval. Fits do not account for unknown future arrivals, scheduler delay, staging, prefill compute shape, transport, or interruption.

| Hypothetical contiguous work ms | Fits observed intervals | Fits with recent P-ready sample at possible start | Total eligible interval s |
| --- | --- | --- | --- |
| 5 | 31 | 30 | 20.4483 |
| 10 | 31 | 30 | 20.4483 |
| 25 | 31 | 30 | 20.4483 |
| 50 | 31 | 30 | 20.4483 |
| 100 | 31 | 30 | 20.4483 |
| 250 | 30 | 29 | 20.2548 |
| 500 | 26 | 24 | 18.7107 |
| 750 | 10 | 10 | 8.9576 |
| 1000 | 1 | 1 | 1.0415 |
| 1150 | 0 | 0 | 0.0000 |
| 1250 | 0 | 0 | 0.0000 |

The 1,150 and 1,250 ms rows approximate the separately measured natural N24 and N64 naïve D-local median HTTP submit-to-first-token enclosures. These comparisons apply measured work quanta to the native trace post hoc.

Measured contiguous intervals under the proxy and scheduler scopes.

| Interval condition | p50 ms | p95 ms | Maximum ms |
| --- | --- | --- | --- |
| D_kernel_idle_proxy_request_free_ms | 627.1 | 934.9 | 1,041.5 |
| D_kernel_idle_proxy_request_free_with_recent_P_ready_ms | 269.5 | 567.7 | 851.5 |

[Figure C](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/figure_c_decode_opportunity.svg) · [Idle window distribution](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/decode_idle_window_distribution.svg)

## Arrival timing through the chain

Elapsed completion gaps measure the actual arrival transformation. The stages are dependent and the cohorts can differ in model and request geometry.

| Stream in the shared ROI | Count | p50 gap ms | p95 gap ms | Gap CV |
| --- | --- | --- | --- | --- |
| scheduled_arrival | 39 | 541.5 | 1,830.0 | 0.82 |
| E_complete | 40 | 6.8 | 2,525.6 | 1.52 |
| P_submit | 40 | 6.8 | 2,525.5 | 1.53 |
| P_ready | 30 | 793.2 | 1,081.1 | 0.20 |
| D_submit | 30 | 793.6 | 1,080.9 | 0.20 |

## Physical NIXL handoff readiness

The opt-in D observer matches each original P/H connector descriptor to the native async READ by its exact producer engine and request identity. The post call entry and first DONE return bound possible READ activity. The last PROC entry bounds the interval still outstanding. The read complement is reported only when every recorded successful request has a valid READ or an explicit destination-local path and capture reports zero drops. This captures D-worker NIXL state in host monotonic time. Exclusive DMA utilization and other transports require additional observations. These rows retain a diagnostic observer classification.

| Epoch and cohort | Native READ requests | D-local no handoff | Successful clients | Whole-cohort accounted | Native bytes GiB | Possible READ ROI | Definite outstanding ROI | Proxy quiet outside accounted READs | Kernel idle outside accounted READs | Kernel idle + proxy quiet outside accounted READs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `nixl-ready/nixl_readiness_n24_clean_before` | 24 | 0 | 24 | yes | 33.55 | 2.49% | 0.74% | 84.82% | — | — |
| `nixl-ready/nixl_readiness_n24_profile` | 24 | 0 | 24 | yes | 33.55 | 3.17% | 0.88% | 81.36% | 88.18% | 81.36% |
| `nixl-ready/nixl_readiness_n24_clean_after` | 24 | 0 | 24 | yes | 33.55 | 2.41% | 0.61% | 83.87% | — | — |

The worker-ready timestamp is the return of D's original get-finished routine. Request-level ready intervals may include NIXL state polling. Native telemetry duration is available separately.

| Epoch and cohort | Native transfer p50/p95 ms | First post to worker ready p50/p95 ms | Latest DONE to worker ready median μs | E/P busy, D idle outside accounted READs |
| --- | --- | --- | --- | --- |
| `nixl-ready/nixl_readiness_n24_clean_before` | 21.823/26.142 | 21.929/26.262 | 94.68 | — |
| `nixl-ready/nixl_readiness_n24_profile` | 27.660/34.394 | 27.773/34.509 | 93.66 | 81.26% |
| `nixl-ready/nixl_readiness_n24_clean_after` | 20.619/25.251 | 20.741/25.366 | 92.70 | — |

[Native NIXL READ timeline](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/native_nixl_read_timeline.svg) · [Operation-level timestamps and telemetry](../mllm_epd_prefill_heavy_modality_discovery/results/region/native_nixl_READ_intervals.csv)

## Arrival-matched text control

This test reuses the observed E-ready offsets from a completed multimodal cohort with the same Language Model checkpoint. Text requests enter a native 1P/1D NIXL path at those offsets. The test isolates arrival spacing with comparable prompt length, while input content and the live encoder remain different.

`controlled-sweep/v95_q112` → `late-native/text_e_ready_replayed` · `MEASURED_ONLINE`. Largest scheduled offset error 0.000 μs. Largest difference between sorted input lengths 1 token(s).

| Full cohort stream | Count | Observed span rate /s | Median gap ms | p95 gap ms | Gap CV |
| --- | --- | --- | --- | --- | --- |
| source_client_arrival | 24 | 3.684 | 235.8 | 536.8 | 0.84 |
| source_E_complete | 24 | 3.057 | 324.3 | 606.4 | 1.01 |
| source_P_submit | 24 | 3.057 | 324.1 | 606.4 | 1.01 |
| source_P_ready | 24 | 2.219 | 443.7 | 498.4 | 0.05 |
| replay_client_arrival | 24 | 3.057 | 324.3 | 606.4 | 1.01 |
| replay_P_submit | 24 | 3.058 | 325.1 | 606.4 | 1.01 |
| replay_P_ready | 24 | 2.298 | 436.9 | 452.0 | 0.03 |
| nearest_text_Poisson_arrival | 24 | 4.702 | 184.8 | 420.6 | 0.84 |

Live stage activity in the central 60% of the matched E-ready and text replay arrival clocks.

| Clock | ROI s | P submits/completes | P-completed input tokens/s | D proxy request-free | P-submit to first client token p95 ms |
| --- | --- | --- | --- | --- | --- |
| source_from_E_ready | 6.20 | 15/13 | 34,341 | 73.09% | 2,850.6 |
| text_replay_from_client_schedule | 6.20 | 15/14 | 36,982 | 58.57% | 2,532.3 |

Nearest same-checkpoint text Poisson cohort `late-native/text_e_ready_rate_poisson` offered 3.057 req/s. Its offered rate differs by 0.00% from the recorded E-ready finite rate. Its finite realized schedule is reported separately from its nominal rate.

[E-ready arrival replay figure](../mllm_epd_prefill_heavy_modality_discovery/results/region/figures/encoder_ready_arrival_replay.svg)

## Historical AR context

Historical Qwen3-8B native 1P/1D N24 controlled 16K/O16 at 2.6 req/s. Model, arrival transform, media and KV budget differ. It is a descriptive comparison.

The prior raw streams are re-binned over this study's central p20–p80 interval while keeping the published whole-cohort observer result. The AR comparison gives P kernel busy 92.44%, D kernel busy 15.03%, P-busy and D-kernel-idle overlap 77.96%, and D kernel idle and proxy request free 56.33%. This supplies a descriptive reference. Model-specific or modality-specific explanations require a controlled comparison. [Published AR baseline](https://github.com/jyssys/MLLM-EP/blob/5bdd6ef5e91840e37c776c2ad937866531e6a391/poc_flashvep/reports/pd_prefill_heavy_resource_motivation.md)
