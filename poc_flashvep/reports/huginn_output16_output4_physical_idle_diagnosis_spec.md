# Huginn Output16/4 physical-idle diagnosis

## Scope

Complete the Loop28 tail diagnosis before these GPU controls.
Use original Native-path Zero from commit `21a067b8bf2242bb8a16fc8c59ed59193e481f54`.
Keep original BF16 weights, R32, input 2048, QPS 1.971, slots 2/8, and GPU pair 4/5.
Change only the output cap between 16 and 4.
Keep the original prompt pool, arrival offsets, seed, KV pools, graphs, and qualified runtime.

The first experiment uses N64 for a short diagnosis.
Run a marker-free control and repeat for each output cap.
Run one five-second activity-only capture per cap.
The capture starts approximately 15 seconds after the measurement marker.
Warm the collector before measurement. Export records after output completion.
Do not add a model-operation Event observer to these controls.

## Measurement and qualification

Calculate kernel union across all streams, not the sum of kernel durations.
Report P and D separately. Report the exact capture denominator.
Calculate idle-window counts, percentiles, and threshold counts and lengths.
Do not extrapolate five seconds into a measured whole-cohort idle total.
Kernel inactivity does not imply inactive copy engines or a safe Decode opportunity.

Compare TTFT p95, TPOT p95, ITL p99, and maximum ITL with marker-free controls.
Keep the strict 5% sensitivity screen and report Native A/A variation.
A failed screen permits diagnostic intervals only, not clean physical-idle claims.
Keep unsupported quantities `NOT_MEASURED` or `NOT_IDENTIFIED`.

## Oracle and forecast audit

Use measured loop and tile costs only with their original timing scope.
Do not convert a partial tile into a completed loop.
Report physical packing as an offline upper bound if readiness or handoff cannot be verified.
Do not claim Decode-safe or completion-feasible migration without the required evidence.
Audit `preparation_decode_cycle_ms=93.10616207122803` and its calibration source.
Score drain predictions only against the Decode cohort known at prediction time.
Report error only when request identities and prediction time identify that cohort.

No new scheduler optimization is in scope.
Preserve earlier evidence, the staged index, and the external runtime.
Restore approved burners on all eight GPUs after each closed diagnostic block.
