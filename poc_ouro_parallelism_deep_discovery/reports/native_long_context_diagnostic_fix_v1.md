# Native long-context diagnostic: scoped compatibility fix

## Cause

The legacy [exporter](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_looped_recurrence_tp_4models/worker_probe.py:97) reads `teacher_index` in its returned metadata.
The native `allowed_token_ids` path does not call the legacy teacher hook.
It therefore does not create that field.
The CPU reproduction reached the actual legacy function and failed with `AttributeError` for `teacher_index`.

The v2 output retains 24 clean trial files and `memory_after.json`.
Its [console](/home/esjung/MLLM-EP-ouro-deep-discovery/poc_ouro_parallelism_deep_discovery/outputs/native_longctx_base14_P1_v2/console.log:105) identifies the later export failure.
This patch changes no v1 or v2 output.
It does not rerun or reinterpret their timings.

## Change

The campaign [adapter](../long_context_probe.py:6) overrides only `finish_parity_capture`.
Its [helper](../long_context_diagnostic.py:4) supplies `teacher_index=0` only during the post-timing export call.
The helper then restores the previous field state, including when export fails.
The returned `teacher_steps` value is `NOT_APPLICABLE`.
The return metadata also identifies native allowed-token control and the temporary compatibility field.

The helper does not call `teacher` or change the sampler.
It rejects an already installed teacher hook.
The [runner](../scripts/native_long_context_control.py:40) now selects this campaign extension.
The clean workload, output control, trial order, timing intervals, model, and cache configuration remain unchanged.
No legacy or installed file changes.

## CPU checks and limits

The [CPU suite](../tests/test_long_context_diagnostic_cpu.py) keeps the exact legacy missing-field reproduction.
It checks temporary initialization, cleanup after errors, prior-field restoration, sampler identity, metadata labels, and extension injection.
The existing long-context protocol tests also pass.
The [validation ledger](../artifacts/native_long_context_diagnostic_fix_v1.json) records commands, counts, hashes, and actual timestamps.

No GPU command or GPU execution occurred.
No timing rerun occurred.
Actual diagnostic export remains pending the reviewed GPU run.
