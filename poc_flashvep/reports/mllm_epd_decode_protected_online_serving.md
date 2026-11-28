# Decode-protected online serving

Finite-cohort open-loop metrics use the first planned arrival through the last recorded drain. Decode attainment requires TPOT ≤33 ms and every ITL ≤34 ms. Joint attainment also requires TTFT ≤5 s. All planned requests remain in attainment denominators. Latency percentiles describe completed requests. ITL percentiles pool recorded token gaps. A one-token response has TPOT zero under the client's contract and contributes no gap sample.

## Fresh clean client-facing arms

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dp3-b2048/fresh_dp3_n128 | 1.465 / 16 | 2048 / 2048 | 128/128 | 3.852 | 112.257 | 163.032 | 43/128 | 43/128 | 0.569 / 0.569 | — |
| dp3-b2048/heldout_dp3_n24 | 1.465 / EOS | 2048 / 2048 | 24/24 | 3.820 | 108.507 | 146.341 | 8/24 | 8/24 | 0.583 / 0.583 | — |
| matched-controls/fresh_native_restart_n128 | 1.465 / 16 | 8192 / 8192 | 128/128 | 31.326 | 4.357 | 13.076 | 125/128 | 8/128 | 1.179 / 0.075 | 0 |
| matched-controls/fresh_cost_generic_n64 | 1.465 / 16 | 8192 / 8192 | 64/64 | 10.016 | 6.674 | 23.658 | 63/64 | 20/64 | 1.415 / 0.449 | 11 |
| native-persistence/fresh_native_n256 | 1.465 / 16 | 8192 / 8192 | 256/256 | 61.866 | 4.286 | 12.669 | 254/256 | 6/256 | 1.198 / 0.028 | 0 |
| native-persistence/heldout_native_n24 | 1.465 / EOS | 8192 / 8192 | 24/24 | 7.207 | 4.285 | 10.953 | 24/24 | 17/24 | 1.365 / 0.967 | 0 |
| native-persistence/heldout_protected_generic_n24 | 1.465 / EOS | 8192 / 8192 | 24/24 | 7.311 | 5.172 | 15.002 | 24/24 | 16/24 | 1.384 / 0.922 | 4 |
| native-persistence/fresh_native_fixed32_n64 | 1.465 / 32 | 8192 / 8192 | 64/64 | 14.588 | 4.644 | 12.647 | 64/64 | 8/64 | 1.193 / 0.149 | 0 |
| native-persistence/fresh_native_fixed64_n64 | 1.465 / 64 | 8192 / 8192 | 64/64 | 14.968 | 4.878 | 13.094 | 63/64 | 7/64 | 1.175 / 0.131 | 0 |
| native-recovery/fresh_native_n64 | 1.465 / 16 | 8192 / 8192 | 64/64 | 16.129 | 4.536 | 12.470 | 64/64 | 8/64 | 1.156 / 0.144 | 0 |
| native-recovery/fresh_native_n128 | 1.465 / 16 | 8192 / 8192 | 128/128 | 29.944 | 4.690 | 13.542 | 127/128 | 9/128 | 1.209 / 0.086 | 0 |
| native-recovery/fresh_protected_generic_n64 | 1.465 / 16 | 8192 / 8192 | 64/64 | 11.283 | 4.847 | 13.294 | 64/64 | 18/64 | 1.385 / 0.390 | 9 |
| native-recovery/fresh_native_lower_n64 | 0.879 / 16 | 8192 / 8192 | 64/64 | 6.574 | 4.431 | 10.890 | 64/64 | 54/64 | 0.949 / 0.801 | 0 |

## Matched Native and additional-work arms

A strict pair requires complete cohorts, the same model, physical GPU assignment, budgets and transfer policy, and identical prompt digests and seeded arrivals. The Decode gate requires TPOT p95 and ITL p99 within +5%, at least 95% absolute Decode attainment, and at least the Native Decode pass count. Normalized text equality uses the originally streamed text after Unicode NFKC, case folding and whitespace collapse. Whole generated tokens and normalized streamed text are identity checks. Candidate task-score noninferiority evidence status is NOT_RUN.

| Native → extra work | Start | Δ TTFT p95 | Δ TPOT p95 | Δ ITL p99 | Δ E2E p95 | Δ joint goodput | Δ Decode goodput | Exact prompt / output | Normalized text exact | Strict Decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| native-recovery/fresh_native_n64 → matched-controls/fresh_cost_generic_n64 | separate | -37.90% | +47.14% | +89.72% | -36.41% | +211.07% | +22.48% | 64/64 / 62/64 | 64/64 | FAIL |
| native-persistence/heldout_native_n24 → native-persistence/heldout_protected_generic_n24 | same | +1.45% | +20.68% | +36.97% | +0.39% | -4.60% | +1.37% | 24/24 / 24/24 | 24/24 | FAIL |
| native-recovery/fresh_native_n64 → native-recovery/fresh_protected_generic_n64 | same | -30.04% | +6.86% | +6.61% | -24.85% | +169.69% | +19.86% | 64/64 / 61/64 | 62/64 | FAIL |

## Historical DP3 controls

The prior H200 DP3 runs use the same number of GPUs, each with a complete model and separate KV budget. They retain their own runtime starts and warmup. Their joint goodput and tails contextualize the serving trade-off. Strict paired gates use fresh Native arms.

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dp3-retry/colocated_long_n24 | 1.465 / 16 | 8192 / 8192 | 24/24 | 3.762 | 306.936 | 538.294 | 4/24 | 4/24 | 0.291 / 0.291 | — |
| dp3-retry/colocated_long_n64 | 1.465 / 16 | 8192 / 8192 | 64/64 | 3.531 | 331.666 | 484.737 | 22/64 | 22/64 | 0.535 / 0.535 | — |
| dp3-retry/colocated_mixed16k_n24 | 2.6 / 16 | 8192 / 8192 | 24/24 | 2.077 | 156.516 | 392.552 | 12/24 | 12/24 | 1.607 / 1.607 | — |
| dp3-retry/colocated_mainmix_eos | 2.955 / EOS | 8192 / 8192 | 64/64 | 2.497 | 51.013 | 361.911 | 38/64 | 38/64 | 1.747 / 1.747 | — |
| dp3-b2048/colocated_long_n24 | 1.465 / 16 | 2048 / 2048 | 24/24 | 3.721 | 111.083 | 173.688 | 7/24 | 7/24 | 0.499 / 0.499 | — |
| dp3-b2048/colocated_long_n64 | 1.465 / 16 | 2048 / 2048 | 64/64 | 3.628 | 110.005 | 150.751 | 28/64 | 28/64 | 0.680 / 0.680 | — |
| dp3-b2048/colocated_mixed16k_n24 | 2.6 / 16 | 2048 / 2048 | 24/24 | 2.054 | 98.435 | 140.919 | 12/24 | 12/24 | 1.579 / 1.579 | — |
| dp3-b2048/colocated_mainmix_eos | 2.955 / EOS | 2048 / 2048 | 64/64 | 2.335 | 58.167 | 120.345 | 39/64 | 39/64 | 1.793 / 1.793 | — |

## Fresh matched DP3 controls

These arms retain the original request cohort and arrival schedule on the same physical GPUs. DP3 uses different operator and communication paths and its own B2048 batch budget. The paired changes describe a topology comparison. Its strict same-runtime gate is ineligible.

| Native → extra work | Start | Δ TTFT p95 | Δ TPOT p95 | Δ ITL p99 | Δ E2E p95 | Δ joint goodput | Δ Decode goodput | Exact prompt / output | Normalized text exact | Strict Decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| matched-controls/fresh_native_restart_n128 → dp3-b2048/fresh_dp3_n128 | separate | -87.70% | +2476.35% | +1146.83% | -82.57% | +653.94% | -51.75% | 128/128 / 111/128 | 117/128 | INELIGIBLE |
| native-recovery/fresh_native_n128 → dp3-b2048/fresh_dp3_n128 | separate | -87.13% | +2293.60% | +1103.91% | -81.77% | +564.41% | -52.92% | 128/128 / 111/128 | 117/128 | INELIGIBLE |
| native-persistence/heldout_native_n24 → dp3-b2048/heldout_dp3_n24 | separate | -46.99% | +2432.02% | +1236.07% | -34.16% | -39.71% | -57.29% | 24/24 / 22/24 | 22/24 | INELIGIBLE |

## Native Fixed16 persistence

Fresh N64, N128, N256 reuse the 24 original Very-long requests at 1.465 offered req/s, so the larger finite cohorts test the longer pressure interval over the same inputs. The N24 point is historical.

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deflection-inline/inline_native_n24 | 1.465 / 16 | 8192 / 8192 | 24/24 | 10.519 | 4.504 | 11.326 | 24/24 | 8/24 | 1.152 / 0.384 | — |

## Held-out Natural EOS output agreement

The independent original requests use ordinary model generation with the serving cap. Requested output mode and actual stopping evidence are reported together. Full prompt/token identities and first divergent steps are in the paired receipt. This evidence has no scored task comparison.

| Held-out Natural EOS run | Emitted tokens, min / p50 / max | Finish reasons |
|---|---:|---|
| dp3-b2048/heldout_dp3_n24 | 2 / 15.0 / 187 | stop 24 |
| native-persistence/heldout_native_n24 | 2 / 15.0 / 24 | stop 24 |
| native-persistence/heldout_protected_generic_n24 | 2 / 15.0 / 24 | stop 24 |

## Text-only arrival controls

The text-only arms use the same language backbone and a similar LM length to the Very-long image workload. Their visible inputs are distinct from the image prompts. The baseline and replay probe the arrival stream and decode tail descriptively.

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matched-controls/fresh_text_poisson_n64 | 1.465 / 16 | 8192 / 8192 | 64/64 | 7.777 | 2.386 | 10.815 | 64/64 | 17/64 | 1.385 / 0.368 | 0 |
| matched-controls/fresh_text_ready_replay_n64 | 1.61705 / 16 | 8192 / 8192 | 64/64 | 7.400 | 2.115 | 13.126 | 64/64 | 39/64 | 1.367 / 0.833 | 0 |

matched-controls/fresh_text_ready_replay_n64 uses 64 actual full-E-ready events from native-recovery/fresh_native_n64. The event identities, frozen offsets and client scheduled arrival offsets reproduce the recorded source.

The observed finite encoder-ready cadence is 1.617 req/s, using the intervals from first to last event. The independent Poisson nominal offered rate is 1.465 req/s.

Matched text output IDs agree on 64/64 prompts across Poisson and readiness replay. Offered QPS differed by +10.38%.

## Independent stage references

These measurements come from independent stage service receipts. They describe the measured stage contract and do not contribute to the online request goodput.

| Stage reference | Stage | Requests | Requests/s | Token rate and field | Batch-one |
|---|---|---:|---:|---|---|
| matched-controls/fresh_stage_encoder | encoder | 24 | 1.639 | 42617.035/s (visual_tokens) | False |
| matched-controls/fresh_stage_prefill | prefill | 24 | 1.221 | 31878.422/s (input_tokens) | False |
| matched-controls/fresh_stage_decode_b1 | decode | 24 | 6.539 | 104.628/s (output_tokens) | True |

## Observer clean controls and profiles

Observer clean controls are retained in their own series. Instrumented arms retain diagnostic scope, including any failed requests during that arm.

### CUDA Graph observer series

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lean-graph-retry/lean_graph_long128_clean_before | 1.465 / 16 | 8192 / 8192 | 128/128 | 33.357 | 4.584 | 12.310 | 127/128 | 7/128 | 1.179 / 0.065 | 0 |
| lean-graph-retry/lean_graph_long128_profile | 1.465 / 16 | 8192 / 8192 | 0/128 | — | — | — | 0/128 | 0/128 | 0.000 / 0.000 | — |

Arms without a completion receipt: lean-graph/lean_graph_long128_clean_before (NOT_RUN), lean-graph/lean_graph_long128_profile (NOT_RUN), lean-graph/lean_graph_long128_clean_after (NOT_RUN), lean-graph-retry/lean_graph_long128_clean_after (NOT_RUN_TERMINATED).

### Kernel-only observer series

| Case | QPS / output | P/D token budget | Completed | TTFT p95 s | TPOT p95 ms | ITL p99 ms | Decode passes | Joint passes | Decode / joint goodput req/s | D-local requests |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lean-kernel-roi/lean_kernel_roi_n128_clean_before | 1.465 / 16 | 8192 / 8192 | 128/128 | 34.258 | 4.268 | 11.696 | 128/128 | 6/128 | 1.173 / 0.055 | 0 |
| lean-kernel-roi/lean_kernel_roi_n128_profile | 1.465 / 16 | 8192 / 8192 | 128/128 | 32.650 | 5.043 | 13.203 | 127/128 | 7/128 | 1.171 / 0.065 | 0 |
| lean-kernel-roi/lean_kernel_roi_n128_clean_after | 1.465 / 16 | 8192 / 8192 | 128/128 | 32.781 | 4.454 | 13.080 | 128/128 | 6/128 | 1.187 / 0.056 | 0 |

## Evidence scope

Larger-census or Natural EOS observations without a matched intervention: dp3-b2048/fresh_dp3_n128, dp3-b2048/heldout_dp3_n24, matched-controls/fresh_native_restart_n128, native-persistence/fresh_native_n256, native-recovery/fresh_native_n128.

Finite-cohort throughput measures completion and drain at the offered load. Independent paired restarts, sustainable QPS and candidate task-score noninferiority require separate evidence.

[All tails, pass counts and rates](../mllm_epd_modality_aware_idle_execution_discovery/results/online_cases.csv) · [Paired measurements and raw SHA-256 receipts](../mllm_epd_modality_aware_idle_execution_discovery/results/online_evidence.json) · [Reduction source](../mllm_epd_modality_aware_idle_execution_discovery/online_evidence.py) · [Native census persistence](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_persistence.svg) · [Goodput and Decode tail](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_pareto.svg) · [Paired metric changes](../mllm_epd_modality_aware_idle_execution_discovery/results/figures/online_native_paired.svg)
