# Huginn profiling recovery

Status: RECOVERED_FOR_TESTED_COHORTS. The historical broad-profiler failure remains causally unresolved.

## Scope

The model is Huginn-0125 at fixed R32 and BF16.
The original recovery workload remains 3500 input tokens and 32 output tokens.
The same qualified numerical patch, graph geometry, native connector, and model revision apply.
Internal output parity does not establish original-HF fidelity.

## Historical failure

The previous campaign stopped after 80/128 and 14/128 responses under broad Nsight profiling.
Its watchdog recorded `PROFILE_NO_RESPONSE_PROGRESS_180S`.
The available evidence does not identify the first blocked operation.
Those failures remain unexplained at the causal level.

The previous launcher traced across `fork` before `exec`.
The installed Nsight help warns about undefined behavior for that option.
This warning identifies a risk, not the proven cause of the historical failure.

## Selected-stage Nsight control

The new launcher starts only the selected native StageWorker under Nsight.
It passes the native IPC descriptor through `exec`.
The other stage retains its normal worker process.
This changes instrumentation scope, not model execution or transfer ownership.

All short selected-stage cohorts completed inference.
The hot capture protocol still adds substantial measurement overhead.
Capture start and stop calls can block the owner thread.
Their wall spans are not GPU kernel time.

| Protocol | Requests | Output comparison | Matched wall difference | Interpretation |
| --- | ---: | --- | ---: | --- |
| Hot capture during measured traffic | 1 | Exact | +83.76% | Diagnostic only |
| Hot capture during measured traffic | 4 | Exact | +103.47% | Diagnostic only |
| Hot capture during measured traffic | 8 | Exact | +76.43% | Diagnostic only |
| Hot capture during measured traffic | 24 | Exact | +27.86% | Diagnostic only |
| Capture starts before worker-ready | 1 | Exact | +8.12% | Fails the 5% gate |
| Capture starts before worker-ready | 4 | 4/4 exact | +4.88% | Passes the bounded comparison |
| Capture starts before worker-ready | 8 | Exact | +17.81% | Fails the 5% gate |
| Prewarmed capture, application-exit finalization | 24 | 24/24 exact | +8.98% | Valid diagnostic trace, not clean activity |
| Same protocol, CUDA flush interval 300,000 ms | 24 | 24/24 exact | +7.70% | Still fails the 5% gate |

The 24-request deferred-flush trace contains 472,272 measured-window kernels.
Its whole-window kernel union is 59.740%.
That window includes request arrivals and Decode drain.
It is not a measure of saturated P service alone.
The failed perturbation gate prevents a clean-serving saturation claim.

Exact run names and complete values remain in `profile_recovery_results.csv` and each run's `profile_analysis.json`.
The raw reports retain individual graph-node events where the profiler records them.
Graph-active spans do not substitute for kernel intervals.

## Finalization is not an inference-progress failure

Two prewarmed Nsight cohorts completed all 24 responses before native shutdown failed.
The owner then blocked during explicit profiler-stop or report finalization.
The default native shutdown bound was 30 seconds.
These failures differ from the historical 80/128 and 14/128 in-request stalls.

The application-exit protocol avoids an explicit timed stop during serving.
Its native transport configuration sets `shutdown_timeout=300`.
The response window ends before finalization and conversion.
Finalization remains a separate resource cost.

## Native CUDA-only control

The current RLT already supports `ProfileConfig(activities=('cuda',))`.
This control uses that interface before worker-ready and warm-up.
It disables CPU-operator recording, shape recording, stacks, and memory profiling.
It does not disable CUDA graphs or asynchronous execution.
It does not replace the engine.

The exported Chrome trace retains `baseTimeNanoseconds`.
An explicit epoch/monotonic clock pair aligns kernels with actual request events.
A CPU test checked this timestamp conversion before the GPU run.
The parser keeps overlapping kernels as an interval union.

The first CUDA-only GPU cohort completed four responses but failed during artifact finalization.
An orchestration field named `pd_shutdown_timeout` did not change native `PDConfig`.
The corrected control sets the actual `native_transport_limits.shutdown_timeout` field.
The failed namespace remains intact.

| Corrected native control | Requests | Output comparison | Matched wall difference | Recorded kernels | Result |
| --- | ---: | --- | ---: | ---: | --- |
| `recovery_huginn_3500_32_native_cuda_only_profile4_ownerclose300_v2` | 4 | 4/4 exact | +1.679% | 78,712 | COMPLETE, within the 5% gate |
| `recovery_huginn_3500_32_native_cuda_only_profile24_ownerclose300_v2` | 24 | 24/24 exact | +1.351% | 472,272 | COMPLETE, within the 5% gate |

The four-request whole-window kernel union is 42.617%.
This is a low-load observation, not a P saturation claim.
A successful trace does not qualify every longer serving run.
The 24-request whole-window kernel union is 63.349%.
Its measured window lasts 38.283990 seconds.
Its matched clean window lasts 37.773681 seconds.
The native profiler completes artifact export after that response window.
Export duration is not request latency.

## Short-window collector limitation

The three-second CUDA toggle control completed all four requests.
It did not record any CUDA kernel events.
The parent analysis rejected the empty activity trace.
This failure does not mean zero GPU activity.
The original exported trace and failed analysis log remain intact.
Subsequent measurements use the validated full-cohort CUDA-only collector.
They do not reuse the failed toggle trace as activity evidence.

## Remaining acceptance checks

1. Check model-specific Primary serving under the same profiling protocol.
2. Capture D graph-internal kernels where the supported interface permits them.
3. Check P queue, P compute, D credit, and D SLO in the same observation interval.
4. Retain every perturbed trace as diagnostic evidence only.

The current evidence supports forward-progress recovery for bounded profiling.
It does not yet prove the final Prefill-heavy hypothesis.
