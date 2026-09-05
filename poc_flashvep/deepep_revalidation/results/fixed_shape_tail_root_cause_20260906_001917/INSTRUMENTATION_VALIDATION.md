# Instrumentation validation

## Validated

1. Baseline and diagnostic runs use same-device CUDA events around stock `FusedMoE.apply`, DeepEP layout/dispatch/combine wrappers, and modular expert execution.
2. No absolute CUDA timestamps are subtracted across GPUs. Cross-rank analysis joins local durations by DP/local invocation/layer.
3. Every stage record carries `local_invocation_id`, `route_id`, layer, DP/EP rank, phase, and M. `scheduler_iteration_id` is intentionally marked as a local MoE-invocation proxy because the stock V1 hook has no native scheduler id.
4. Full-serving Nsight capture contains four child worker CUDA contexts and 424,716 kernel rows, including DeepEP and expert kernels.
5. Fixed-input isolated replay: 20 warmups + 100 measured iterations, exact `M=2984`, `g=30`, route unchanged; expert samples are stable (median ~0.235 ms, max 0.382 ms).
6. Controlled full DeepEP replay: 100 iterations × 4 ranks on the same request/layer; expert p50 0.481 ms, p99 0.580 ms, max 0.745 ms.

## Intervention validation

`FLASHVEP_SYNC_BEFORE_MOE=1` is diagnostic only. In decode M=1, baseline had 17/71,424 events >10 ms and 10 >20 ms; synchronized run had 6/44,352 >10 ms and 0 >20 ms. This is a 64.7% reduction in event count (43.2% reduction in rate) and eliminates the >20 ms class. Whole p99 falls from 2.409 to 2.094 ms, while mixed large-prefill shapes can become worse; therefore the intervention is evidence about outstanding state, not a proposed production fix.

## Limitations

The hook cannot name the native V1 scheduler iteration, and the dispatch wrapper measures the CUDA event span including dependency wait rather than exposing a separate `cudaEventSynchronize` wait counter. Nsight run has no NVTX rows (NVTX was disabled to avoid the prior NCCL crash). These limitations do not invalidate stage localization because the same dispatch-first pattern is visible in actual DeepEP kernel rows and fixed-route controls.
