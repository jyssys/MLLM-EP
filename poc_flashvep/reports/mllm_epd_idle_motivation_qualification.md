# MLLM EPD idle motivation qualification

## Native proxy traces

The prior GPU profiles carry the diagnostic label under the 5% observer gate. The clean proxy analysis also has diagnostic scope. The future-aware fit is an offline oracle.

11 held-out serving arms provide 581 E-ready decision timestamps inside the central p20–p80 scheduled-arrival interval. The analysis joins complete native E→P→D requests. An outstanding proxy request covers its native P-ready moment through D HTTP completion. This is the stage controller's request boundary and leaves other work, including GPU kernels and native connector activity, to separate measurements.

| Clean native cohort | Campaign | ROI s | D proxy quiet | Quiet p50/p95/max ms | E-ready calls | D quiet and P outstanding ≥2 | Eligible-call residual p50/p95/max ms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `observer-long64/long_primed_n64_clean_before` | previous | 25.1 | 83.0% | 636.7/914.1/1,023.3 | 40 | 32 | 321.6/694.2/773.0 |
| `observer-long64/long_primed_n64_clean_after` | previous | 25.1 | 81.6% | 619.5/924.8/975.8 | 40 | 38 | 417.8/583.6/587.7 |
| `natural-confirmation/long_persistence128` | previous | 47.0 | 81.7% | 614.5/900.0/941.0 | 73 | 59 | 422.0/836.9/854.1 |
| `deflection-confirm/confirm_native_n64` | previous | 25.1 | 82.3% | 615.9/888.5/899.8 | 40 | 34 | 288.8/823.2/833.2 |
| `late-native/v25_primed_n24_clean_before` | previous | 3.8 | 73.1% | 318.5/348.0/355.7 | 14 | 8 | 168.9/290.9/314.0 |
| `controlled-sweep/v95_q112` | previous | 4.0 | 71.0% | 315.5/351.8/375.3 | 11 | 8 | 144.9/272.9/308.2 |
| `native-recovery/fresh_native_n64` | current | 25.1 | 82.0% | 614.2/895.0/907.3 | 40 | 38 | 363.9/555.7/644.5 |
| `native-recovery/fresh_native_n128` | current | 47.0 | 82.0% | 606.1/902.3/1,110.8 | 74 | 47 | 315.2/612.8/857.5 |
| `native-recovery/fresh_native_lower_n64` | current | 41.9 | 86.3% | 633.7/2,069.6/2,697.7 | 40 | 20 | 360.8/705.7/861.7 |
| `native-persistence/fresh_native_n256` | current | 85.7 | 82.2% | 616.6/924.0/1,713.7 | 136 | 114 | 318.7/898.9/1,060.5 |
| `matched-controls/fresh_native_restart_n128` | current | 47.0 | 81.7% | 617.2/903.7/909.1 | 73 | 55 | 447.3/783.4/795.5 |

## Prior physical captures

These fits use physically observed D kernel intervals and preserve the original observer classification. The optional connector column also excludes the native READ enclosures with complete accounting.

| GPU capture | E/P/D kernel activity | D kernel idle and proxy quiet p50/p95/max ms | Also outside accounted native READs, max ms | Evidence |
| --- | --- | --- | --- | --- |
| `observer-long64/long_primed_n64_profile` | 90.0/89.9/9.6% | 627.1/934.9/1,041.5 | — | DIAGNOSTIC |
| `nixl-ready/nixl_readiness_n24_profile` | 98.3/94.0/8.7% | 644.7/840.4/875.0 | 875.0 | DIAGNOSTIC |
| `late-native/v25_primed_n24_profile` | 31.5/97.5/18.4% | 297.5/320.4/320.5 | — | DIAGNOSTIC |

## Current exact-kernel capture

The delayed trace is analysed over the intersection of the canonical scheduled-arrival interval and all three actual CUPTI anchor envelopes. All E/P/D activity fractions share this trimmed monotonic window. The cohort-level observer guard compares the full clean-before, profile and clean-after requests.

| Physical capture | Canonical p20–p80 s | Actually observed s | Canonical coverage | E/P/D kernel activity | E/P kernel busy and D kernel idle | D kernel idle and proxy quiet | Observer gate | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lean-kernel-roi/lean_kernel_roi_n128_profile` | 11.62–58.58 | 22.00–58.58 | 77.9% | 88.8/92.5/9.8% | 74.0% | 80.4% | FAIL | DIAGNOSTIC |

P-ready duration has sample-and-hold scope. Decode SLO covers requests with all output token emissions inside the common window. P throughput uses actual completed prompt tokens per common-window second.

| Physical capture | P state covered | P known ready | P ready plus idle/quiet D | P prompt tokens/s | Sampled P KV rejects/attempts | D absolute SLO |
| --- | --- | --- | --- | --- | --- | --- |
| `lean-kernel-roi/lean_kernel_roi_n128_profile` | 96.7% | 96.7% | 77.1% | 32,107.1 | 0/111 | 44/45 |

Host queue values are means of periodic snapshots. The D quiet envelope uses request state at the proxy.
Transfer coverage is stated separately because absent kernel activity alone leaves native reads and DMA unobserved.

| Physical capture | Host samples | E waiting | P waiting | D capacity waiting | D idle and quiet p50/p95/max ms | D CUDA copy trace | Native READ enclosures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lean-kernel-roi/lean_kernel_roi_n128_profile` | 129 | 0.20 | 10.83 | 0.00 | 624.7/882.0/1,117.3 | Unavailable | Unavailable |

The observer table gives clean spread / profile departure from the clean center. Each limit is 5%, with complete matching outputs and intact collector clocks.

| Physical capture | Duration | TTFT p50 | TTFT p95 | TPOT p95 | ITL p99 |
| --- | --- | --- | --- | --- | --- |
| `lean-kernel-roi/lean_kernel_roi_n128_profile` | 1.18%/0.01% | 2.98%/5.39% | 4.41%/2.59% | 4.26%/15.64% | 11.18%/6.58% |

## Failed D graph-observer attempt

`long:lean-graph-retry/lean_graph_long128_profile` · `DIAGNOSTIC`. Clean-before completed 128/128. The profiled arm completed 0/128 and recorded 128 explicit failures. The clean-after arm has no client receipt.
The D API accepted the profile begin RPC after the engine had already instantiated its replay graphs. The native segfault stack reached `cuGraphLaunch`, `cudaGraphLaunch`, and PyTorch graph replay. The D graph CSV and close ROI were absent. E later reached the 12 GiB backing-storage guard for live encoder tensors while the failed stream could no longer retire normally. That guard was a secondary failure.

The strongest explanation is the graph-instrumentation activation order. NVIDIA documents selecting graph tracing before graph executable instantiation. Retrofitting pre-existing graphs lacks the required timing instrumentation. A clean early-activation smoke would be needed to isolate the exact driver fault. This attempt supplies no D graph duration or observer-qualified activity window. [CUPTI trace-level lifetime](https://docs.nvidia.com/cupti/main/main.html#trace-level-selection-and-lifetime)

[Diagnostic excerpts and source log hashes](../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility/graph_failure_receipt.json)

## Admission at E readiness

The reactive comparator launches a fixed quantum when D has no outstanding proxy request and at least 2 earlier prefills remain outstanding. The causal cadence comparator also requires its predicted time to the next native handoff to exceed the quantum plus reserve. Prediction subtracts time since the most recent observed P-ready from a lower empirical handoff interval frozen on separate cohorts. Prediction stays unavailable before the first observed handoff.

The oracle uses the actual next native P-ready after each decision. The scoring keeps all original native handoffs, including the diverted candidate's native copy, and prevents hypothetical assists from overlapping on D. A crossing is an overlap with ready native work. The replay measures neither a resulting ITL violation nor a service gain. A visual operator that becomes runnable before whole-request E readiness has a different decision surface and needs its own trace. Some held-out cohorts share a workload roster, seed or server epoch with calibration. This checks distinct cohorts under those shared conditions.

| Training family | Calibration inter-handoff gaps | Frozen empirical p10 ms |
| --- | --- | --- |
| long | 19 | 727.0 |
| v16k | 16 | 405.7 |
| long-low | 10 | 453.2 |

### long, 25 ms reserve

Each entry gives admissions / within reserve / crossing the handoff. The rest consume the reserve or have an unobserved finish boundary.

| Quantum ms | Idle only · diagnostic | Past-only cadence · diagnostic | Next-ready · offline oracle |
| --- | --- | --- | --- |
| 25 | 218/202/12 | 165/165/0 | 202/202/0 |
| 50 | 207/178/16 | 144/144/0 | 178/178/0 |
| 100 | 197/158/35 | 122/120/1 | 158/158/0 |
| 250 | 189/112/68 | 77/74/3 | 112/112/0 |
| 500 | 180/47/128 | 11/11/0 | 47/47/0 |
| 750 | 159/11/145 | 0/0/0 | 12/12/0 |
| 1000 | 152/1/151 | 0/0/0 | 1/1/0 |
| 1150 | 143/0/143 | 0/0/0 | 0/0/0 |
| 1250 | 134/0/134 | 0/0/0 | 0/0/0 |

### v16k, 25 ms reserve

Each entry gives admissions / within reserve / crossing the handoff. The rest consume the reserve or have an unobserved finish boundary.

| Quantum ms | Idle only · diagnostic | Past-only cadence · diagnostic | Next-ready · offline oracle |
| --- | --- | --- | --- |
| 25 | 16/15/1 | 12/12/0 | 15/15/0 |
| 50 | 14/12/1 | 9/9/0 | 12/12/0 |
| 100 | 14/9/4 | 7/7/0 | 9/9/0 |
| 250 | 14/2/12 | 1/1/0 | 2/2/0 |
| 500 | 10/0/10 | 0/0/0 | 0/0/0 |
| 750 | 7/0/7 | 0/0/0 | 0/0/0 |
| 1000 | 6/0/6 | 0/0/0 | 0/0/0 |
| 1150 | 6/0/6 | 0/0/0 | 0/0/0 |
| 1250 | 6/0/6 | 0/0/0 | 0/0/0 |

### long-low, 25 ms reserve

Each entry gives admissions / within reserve / crossing the handoff. The rest consume the reserve or have an unobserved finish boundary.

| Quantum ms | Idle only · diagnostic | Past-only cadence · diagnostic | Next-ready · offline oracle |
| --- | --- | --- | --- |
| 25 | 20/17/3 | 4/4/0 | 17/17/0 |
| 50 | 20/16/3 | 4/4/0 | 16/16/0 |
| 100 | 20/15/4 | 3/3/0 | 15/15/0 |
| 250 | 20/12/7 | 0/0/0 | 12/12/0 |
| 500 | 17/3/14 | 0/0/0 | 3/3/0 |
| 750 | 15/1/14 | 0/0/0 | 1/1/0 |
| 1000 | 15/0/15 | 0/0/0 | 0/0/0 |
| 1150 | 14/0/14 | 0/0/0 | 0/0/0 |
| 1250 | 13/0/13 | 0/0/0 | 0/0/0 |

## Live control and required state

The strongest generic implementation uses identical bounded work, waits for a truly admissible D slot, and gates that quantum with a live P-progress-derived lower prediction of the next native handoff. It should use the minimum over in-flight P requests, reserve time for import, yield, scheduling and connector operations, and decline work when progress is stale or the prediction is uncalibrated. A modality-aware condition has incremental value only if it improves decisions or changes the physical work and movement costs under this same control.

Clean proxy records expose submission and completion. Existing profiled P scheduler rows are sparse and record the scheduler's token counters after schedule advances them. The pinned vLLM scheduler tracks completed progress via `max(0, num_computed_tokens - num_in_flight_tokens)`. A live signal needs the request key and a monotonic freshness timestamp along with that progress. It also needs D admission, import occupancy and duration of the actual non-preemptible work quantum. A prior-window fit cannot supply those guarantees.

[Figure](../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility/idle_feasibility.svg) · [Measured and replay reductions](../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility/idle_feasibility.json) · [Decision input and outcome rows](../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility/decision_inputs_and_outcomes.csv)

## Reproduction

The previous raw input can be recovered from the archived native evidence. The capture JSON records the source file digests and selected case IDs.
[Previous raw evidence guide](../mllm_epd_prefill_heavy_modality_discovery/results/raw_evidence/README.md)

```bash
module=poc_flashvep.mllm_epd_modality_aware_idle_execution_discovery
study=poc_flashvep/mllm_epd_modality_aware_idle_execution_discovery
python -m "$module.idle_feasibility" --raw-root ../h200_epd_modality_runs \
    --fresh-root ../h200_idle_execution_runs \
    --fresh-evaluate long:native-recovery/fresh_native_n64 \
    --fresh-evaluate long:native-recovery/fresh_native_n128 \
    --fresh-evaluate long-low:native-recovery/fresh_native_lower_n64 \
    --fresh-evaluate long:native-persistence/fresh_native_n256 \
    --fresh-evaluate long:matched-controls/fresh_native_restart_n128 \
    --fresh-profile long:lean-kernel-roi/lean_kernel_roi_n128_profile \
    --fresh-failed-graph long:lean-graph-retry/lean_graph_long128_profile \
    --graph-failure-receipt "$study/results/idle_feasibility/graph_failure_receipt.json" \
    --out "$study/results/idle_feasibility"
python -m "$module.idle_feasibility_report" \
    --reduction "$study/results/idle_feasibility/idle_feasibility.json" \
    --report poc_flashvep/reports/mllm_epd_idle_motivation_qualification.md \
    --link-root ../mllm_epd_modality_aware_idle_execution_discovery/results/idle_feasibility
```
