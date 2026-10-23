# Profiling scope and native execution

Clean serving uses native CPU submission accounting without the CUDA-event observer.
The separate CALIBRATION audit records 990 prefill service units across16 observed policy cohorts.
Of those units,770 match adaptive selector decisions for prediction/budget comparison.
The remaining units include native Mixed, which has no adaptive decision record.

The observer records a start event after native dependency waits.
It records an end event after native execution submission on the current stream.
Inspection waits for native runner synchronization.
These intervals include preparation and host-submission gaps.
They do not isolate attention, GEMM, or other kernel-only time.
The largest audit host/device alignment bound is 0.1412ms.

## Graph counters

All 990 observed prefill units have zero capture, replay, and fallback counter increments within the unit.
This does not mean the engine disables CUDA graphs.
Decode graph counters advance between observed prefill units.
For example, one audit retains capture count6 while the replay counter advances from 180 to348 between prefill observations.
The data support native prefill/decode path separation in this audit.
They do not support attributing a measured C/G gain to a newly introduced CUDA-graph optimization.
Kernel JIT, other graph shapes, and work outside observed units remain separate costs.
Clean raw cohorts do not contain exhaustive whole-engine graph-counter deltas.

## Staged allocation

These are actual retained staging-buffer allocations observed by the runner, not inferred HBM traffic.

| Audit policy | Configured width4 MiB | Configured width8 MiB |
| --- | ---: | ---: |
| O6 Joint, cap C2048 |32 |64 |
| O6 G-only, cap C2048 |32 |64 |
| O6 C-only, cap C2048 |32 |64 |
| Completion-fair Mixed, C512 |8 |16 |

Selecting a smaller actual C does not shrink the adaptive engine's reserved C2048 staging buffer.
The O10 logical retained-state score is not a measured reduction in physical allocation.
The model's shared native K/V block pool reserves54GiB, separate from these staging buffers.
Per-cohort peak allocated/reserved HBM appears in `serving_results.csv` and `main_table.csv`.

## Timing and publication limits

The host policy timeline records submit-return timestamps and delivered output tokens.
It is not a GPU-kernel overlap trace.
The extra event-service timeline shows actual aligned audit intervals.
It excludes unobserved decode intervals and cannot establish an exhaustive E2E decomposition.
No summed kernel or event service time becomes an E2E percentage in this study.

Process-lifetime GPU utilization, clocks, power, memory, and temperature remain coarse telemetry.
They include startup, warm-up, queue waits, and drained work.
They cannot establish GPU idle critical-path time or achieved HBM bandwidth.
Exact HBM read/write traffic and a kernel-level non-overlapped attribution are NOT_AVAILABLE.
This limitation does not prevent paired clean serving comparisons.

Data: `event_path_profile.csv`, `event_budget_residuals.csv`, `GPU_process_telemetry.csv`, and immutable audit records.
FiguresF/F2 distinguish host submission metadata from measured GPU service intervals.
