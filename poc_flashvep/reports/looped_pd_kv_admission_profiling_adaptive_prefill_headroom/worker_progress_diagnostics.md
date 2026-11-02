# Worker progress diagnostics

Status: IN_PROGRESS. No final Prefill-heavy verdict is available yet.

## Preserved historical failure

The previous Huginn full-run Nsight profiles stopped after 80/128 and 14/128 responses.
Their watchdog reported `PROFILE_NO_RESPONSE_PROGRESS_180S`.
Those runs did not identify the first blocked operation.
The new selected-stage capture protocol does not assume a CUDA, NIXL, or model defect.

## New observer regression

`recovery_ouro_8192_64_original_gate8` failed during observer execution.
The observer referenced `Request.num_output_tokens`, which the native class does not expose.
The correction uses `len(Request.generated_token_ids)`.
The failed namespace remains intact.
The `_v2` retry completed all eight state and work checks.
This failure does not establish a model or KV error.

## Heavy correctness is not clean serving

`recovery_huginn_3500_32_control_gate8` reached its diagnostic time limit.
Its terminal error also contains a native shutdown timeout.
The complete log and partial responses remain intact.
This run used full KV byte checks and Prompt-state hashes.
It did not use Nsight.
It cannot reproduce or explain the historical Nsight-only stall.

Worker stacks identify byte-hash work during the heavy validation.
The legacy checker performs a separate GPU-to-CPU read for each page, layer, depth, and K/V tensor.
The new diagnostic gathers one layer plane before its CPU hash.
It retains the original depth/layer/page/K/V byte order.
CPU regression tests compare its complete dictionary with the legacy checker.
The tests include partial pages and Coda's missing final position.
The retry uses a separate namespace and a larger bounded diagnostic timeout.
No byte checks are removed from clean serving because clean serving never enables these heavy checks.

The new probe also avoids repeated all-layer gathers when it hashes one layer.
This changes diagnostic data collection only.
It does not change a model operation or the native handoff payload.
No diagnostic-time reduction is claimed as a serving speedup.

## Progress evidence

Each new owner can receive `SIGUSR1` for a Python stack snapshot.
Only registered owned worker PIDs receive that signal.
Memory and request snapshots use monotonic timestamps.
The selected-stage profiler uses the unchanged native StageWorker and IPC protocol.
A CPU test confirmed that Nsight preserves the inherited IPC descriptor.
GPU capture validation is still required.

The real CUDA contexts, warm-up, source archive, physical GPU IDs, and run IDs remain separate.
The campaign retains failed namespaces instead of restarting them in place.
