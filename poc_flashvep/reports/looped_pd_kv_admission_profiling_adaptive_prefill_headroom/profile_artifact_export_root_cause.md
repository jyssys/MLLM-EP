# Post-response profile export failure

## Measured failure

Run: `long_profile_ouro_PD_start1_both_n128`.
All 128 measured requests produced their capped outputs.
The measured response window lasted 84.847854213 seconds.
The matched clean window lasted 69.161666623 seconds.
This profile already fails the fixed 5% perturbation gate.
The final dispatcher status is FAILED, not COMPLETE.
The dispatcher elapsed time was 924.7323499014601 seconds.

The exception was `TimeoutError: profile artifact processing is still running`.
The harness called `wait_for_profile_artifacts(300)`.
That fixed argument did not use the longer declared finalization allowance.

## Observed CPU locations

The owner-process stacks identify two distinct post-response operations:

1. `torch.profiler.events()` built a native `FunctionEvent` CPU hierarchy.
2. The native artifact worker parsed and compressed the exported files.

The Decode stack first entered `_populate_cpu_children` through `profiling_artifacts.export_profile`.
The later stack entered `gzip._write_raw` through the native archive worker.
The main owner thread then waited for artifact completion.

These stacks came from the exact registered worker session.
The diagnostic signals occurred after the measured response window.
They did not interrupt the measured request generation.

The exported Decode Chrome trace occupied about 5.8 GiB on disk.
Its redundant native event table occupied about 3.8 GiB.
These rounded file sizes are diagnostic observations, not GPU-memory measurements.

## Minimal remedy and scope

The CUDA-only analysis needs the complete native Chrome trace.
It does not need a second native CPU hierarchy or a derived self-time table.
The targeted export helper retains the complete Chrome trace.
It explicitly records the optional native event table as NOT_COLLECTED.
It retains native archive integrity checks.
The helper does not alter collection activities, model execution, or measured request timing.

The extended CPU/shapes/stacks/memory path retains the original exporter.
The configured finalization timeout must also reach the native artifact wait.
Subsequent runs must use new source archives and new namespaces.
The failed run remains unchanged.

## Causal limit

This result explains this new post-response export failure.
It does not explain the historical Huginn stops at 80/128 and 14/128 responses.
Those earlier failures occurred before the response cohort completed.
Their first blocked operation remains unknown.

## Evidence

- `raw/long_profile_ouro_PD_start1_both_n128/dispatcher_result.json`
- `raw/long_profile_ouro_PD_start1_both_n128/window.json`
- `raw/long_profile_ouro_PD_start1_both_n128/worker_stacks_177813.log`
- `raw/long_profile_ouro_PD_start1_both_n128/worker_stacks_177814.log`
- `raw/long_profile_ouro_PD_start1_both_n128/post_response_export_stack_probe.jsonl`
- `logs/long_profile_ouro_PD_start1_both_n128.log`
