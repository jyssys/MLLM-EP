# Layer costs and completion prediction

## Scope

This report separates measured execution costs from predicted completion times.
The model remains Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`.
All methods retain BF16, exact Attention, and Full36 Decode.
The [specification](dense_ar_decode_safe_elastic_layer_pd_poc_spec.md) fixes the remaining experimental contracts.

## Measured cost tables

`MEASURED`: the destination received actual P-produced partial states.
The table contains CUDA event enclosures for the suffix forward.
These values are not exclusive kernel durations or WCET bounds.

| Destination requests | Last2 p95 (ms) | Last4 p95 (ms) | Last8 p95 (ms) | Last12 p95 (ms) |
|---|---:|---:|---:|---:|
| 4, predecessor calibration | 58.780 | 118.561 | 239.186 | 358.577 |
| 8, new two-sample calibration | 119.532 | 235.905 | 475.882 | 714.333 |
| 16, new two-sample calibration | 259.317 | 495.853 | 998.230 | 1481.049 |

The predictor uses measured maxima for the new two-sample receiver shapes.
Those maxima differ slightly from the table's interpolated p95 values.
The predictor uses a ceiling shape when the exact smaller batch has no table entry.
An artificial six-second collection delay created diagnostic destination batches.
No serving result or handoff predictor includes that artificial delay.

See [receiver costs](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/destination_batch_costs.json).
See [the frozen lookup table](dense_ar_decode_safe_elastic_layer_pd_20261113/configs/revised_cost_model_v3.json).

## Completion model

The old formula assumed overlap that the coordinator could not execute.
The revised formula serializes destination admission and handoff:

`D_start = max(P_cut_ready, D_available_upper) + handoff_cost`

The model then adds the suffix, commit cost, and completion uncertainty.
It estimates known Decode drain from batch-aware costs and remaining output work.
It also includes accepted continuations and known pending work.
The policy does not read future arrivals or future completion timestamps.

Admission budget and immediate execution budget remain separate.
The hard runtime guard waits for existing Decode to retire before genuine suffix execution.
The guard does not guarantee the latency of a Decode request that becomes ready later.
No mechanism assumes kernel preemption.

The Revised candidate predicate does not directly compare its earliest-token-deadline field.
It records that field for analysis.
Its actual selection uses known drain, next known ingress, cost uncertainty, and an execution reserve.
Existing Decode protection comes from the hard cohort guard.
Therefore, this implementation is completion/drain-aware, not a complete per-token deadline optimizer.
The distinction prevents an unsupported claim that the requested stronger policy received an exhaustive negative test.
See [the direct-deadline field audit](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/direct_deadline_implementation_audit.json).

## Frozen uncertainty

Previously viewed runs supplied uncertainty estimates before fresh evaluation.
Source bias was 37.439 ms.
Source absolute-error p95 was 51 ms.
The known-drain underprediction margin was 70 ms.

Output16 completion margins were 110/80/60/90 ms for Last2/4/8/12.
Output64/200 retained110 ms for all candidates.
These margins replaced the unsupported5 ms assumption.
They are empirical bounds, not production guarantees.

Last2 retained110 ms because its training set contained only 11 samples from one start.
The other retained samples covered three, seven, and twelve viewed starts.
Training coverage was 100% by construction.
Training coverage does not establish held-out coverage.

See [uncertainty training](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/v3_uncertainty_training.json).

## Fresh Output16/QPS4.8 prediction results

`TRACE_ANALYSIS`: each row uses realized committed groups from one independent start.
Error equals observed first output minus the central prediction.
A positive error means underprediction.

| Arrival seed | Groups | Completion MAE (ms) | Absolute error p95 (ms) | Central underprediction | Upper-estimate coverage |
|---|---:|---:|---:|---:|---:|
| 2026111381 | 27 | 57.522 | 78.422 | 74.07% | 100% |
| 2026111382 | 26 | 41.078 | 75.228 | 38.46% | 100% |
| 2026111383 | 24 | 59.087 | 91.082 | 75.00% | 100% |

The upper estimate covered all 77 observed committed groups in this cell.
Rejected candidates have no measured counterfactual completion.
Coverage of accepted completions does not prove Decode non-inferiority.
Native-relative ITL and TPOT still require independent-start confidence intervals.

Source-ready MAE was 5.291/24.949/6.237 ms across these starts.
Known active-and-pending drain MAE was 46.524/68.019/71.098 ms.
Corresponding drain absolute-error p95 was 101.442/117.772/169.519 ms.
These errors explain why `t2-t1` cannot serve as a certified safe budget.

See [prediction evidence](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/revised_prediction_validation.json).

## Roofline accounting

`STRUCTURAL_ACCOUNTING`: the accounting uses layer GEMM shapes and causal Attention arithmetic.
It uses nominal dense BF16 peak989.5 TFLOP/s and HBM bandwidth3.35 TB/s.
These values describe the [NVIDIA H100 SXM specification](https://www.nvidia.com/en-us/data-center/h100/).
The dense model does not use the doubled sparse peak.

For batch4, Last2/4/8/12 included-work lower bounds are29.197/58.395/116.790/175.185 ms.
The bounds exclude Normalization, RoPE, activations, CUDA submission costs, and additional Attention/KV traffic.
Actual HBM traffic and Tensor Core utilization remain `NOT_MEASURED`.
Admission uses empirical costs rather than these ideal bounds.

See [structural accounting](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/roofline_structural_accounting.json).

## Transfer accounting

`STRUCTURAL_ACCOUNTING`: Native transfers1125MiB of prompt KV per request.
The intermediate Hidden plus Residual adds125 MiB for a migrated request.

| Cut | Prefix KV plus intermediate state (MiB) | Difference from Native (MiB) |
|---|---:|---:|
| Last2 | 1187.5 | +62.5 |
| Last4 | 1125 | 0 |
| Last8 | 1000 | -125 |
| Last12 | 875 | -250 |

The entire Native KV payload is not additional Layer Split overhead.
NIXL telemetry and Host submit-to-DONE enclosures use different time definitions.
The report does not sum overlapping durations into an exclusive wire cost.
Source buffers remain live until ACK.
Destination buffers remain live until recorded CUDA consumption completes.

`TRACE_ANALYSIS`: all308 migrated requests in the fresh Output16/QPS4.8 cell returned their Hidden ACK and normal Source credit.
Each Hidden/Residual payload contained 131072000 bytes.
Per-start Hidden submit-to-DONE p95 was 0.526/0.921/0.541 ms.
Preparation p95 was 0.290/0.101/0.310 ms.

KV NIXL telemetry p95 was 191.913/191.906/191.896 ms across the same starts.
Those intervals include the telemetry's defined transfer lifecycle.
They are not additive exclusive wire times.
Source-ready to first-output p95 was 531.430/519.884/536.777 ms for migrated requests.
This Host interval includes destination waiting, import, suffix execution, and output delivery.

See [actual byte and lifetime accounting](dense_ar_decode_safe_elastic_layer_pd_20261113/analysis/serving_systems_accounting.json).

## Measurement limits

Clean physical kernel busy, physical idle, and spatial compute slack remain `NOT_MEASURED`.
The warmed CUPTI attempt did not produce qualified capacity evidence.
The bounded Nsight attempt caused large latency changes and failed isolation qualification.
CUDA event controls also failed the full5% tail/A-A criterion.
Their timelines remain diagnostic.

These metrology failures do not invalidate separate marker-free serving measurements.
They prohibit clean physical-idle and exclusive-kernel claims.
The [final summary](dense_ar_decode_safe_elastic_layer_pd_final_summary.md) states the independent serving verdict.
