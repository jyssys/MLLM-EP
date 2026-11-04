# Huginn recovery: diagnostic record

Status: ongoing. This document does not certify the recovery gate.

## Original-condition reproduction

The clean run completed 384 requests. Its output tokens match the previous clean run for all 384 requests.
The input IDs, request seeds, and output depth lists also match.
The first profile run completed 384 requests. Its output tokens match the new clean run.
The previous NIXL progress stall did not recur in this start. The native stall cause remains `ROOT_CAUSE_INCONCLUSIVE`.
No runtime fix preceded this result.

| Measured quantity | First profile run |
|---|---:|
| Clean response window | 233.263220 s |
| Profile response window | 247.390348 s |
| Response-wall difference | +6.0563% |
| Recorded capture interval | 8.195270 s |
| P kernel union in capture | 87.5205% |
| Recorded kernel invocations | 124,612 |
| `cudaProfilerStart` host call | 3.527506 s |
| `cudaProfilerStop` host call | 12.142965 s |

The profile fails the unchanged 5% perturbation gate.
The API durations are host-observed blocking durations. They are not pure GPU time or exclusive E2E components.
Their sum does not prove a causal decomposition of the response-wall difference.
The kernel union cannot certify clean-serving saturation while the perturbation gate fails.

## Ranked hypotheses and predictions

1. Collector volume or finalization causes much of the perturbation. A shorter capture should reduce its blocking duration and response-wall difference.
2. Independent-start variation contributes materially. Identical eight-second captures should show comparable variation without a runtime change.
3. Native transfer progress contributes additional stalls. Such stalls should coincide with NIXL submission, progress, or acknowledgment waits.

The first controls repeat the eight-second capture and then reduce only capture duration to three seconds.
Both controls retain the same prompts, arrivals, probe requests, geometry, and SLOs.
All rejected captures remain in the evidence directory.

## Evidence

- `raw/recovery_huginn_original_clean384/`
- `raw/recovery_huginn_original_profile384/profile_analysis.json`
- `raw/recovery_huginn_original_profile384/capture_events.jsonl`
- `state/clean_reproduction_parity.json`
- `state/reproduction_result_audited.json`

No Dynamic A or FluidToken serving result exists in this diagnostic record.
# Reproduced progress stall in the 512-request control

The paired clean execution completed 512 requests. The profile execution stopped after 175 measured responses.
The watchdog recorded `PROFILE_NO_RESPONSE_PROGRESS_60S`.
The P worker stack remained inside `nixl_cu12._api.transfer`, called by native `transport.submit` and `connector.progress`.
The capture log contains START but no STOP. This failure occurred before the collector could finish its bounded window.

This reproduces the previous symptom. It does not identify the native lock or CUDA dependency that caused the stall.
The status remains `ROOT_CAUSE_INCONCLUSIVE`.
The new diagnostic uses 128 requests and native stack collection after 30 seconds without response progress.
The diagnostic is not a clean latency measurement.

Evidence: `raw/recovery_huginn_extended1_profile512/watchdog.json` and `worker_stacks_1501539.log`.

## Native stack localization

The 128-request diagnostic completed. The 512-request diagnostic reproduced the stall.
`py-spy dump --native` succeeded during that stall.
The main thread followed this native path:

`nixlAgent::postXferReq` → `nixlUcxEngine::postXfer` → `ucp_put_nbx` → `uct_cuda_ipc_ep_put_zcopy` → `libcupti.so.12.8` → `libcuda.so.570.211.01` → `pthread_rwlock_wrlock`.

This evidence localizes the blocking call below the Python scheduler and NIXL wrapper.
It does not identify the lock owner or prove a complete deadlock cycle.
The binding released the Python GIL before the native call, as its stack symbol shows.
The diagnostic does not establish that the NIXL progress thread caused the driver lock wait.
The next controlled change disables that supported native progress thread. Native polling and acknowledgment rules remain unchanged.

Evidence: `raw/recovery_huginn_stall_native512/native_stall_stack_1520692.json`.

## Progress-thread control

`enable_prog_thread=False` preserved all eight tested output sequences and recorded prompt states.
The matched 512-request clean execution completed.
The profile execution still stopped inside the same CUDA-IPC/CUPTI/driver path.
An all-thread GDB capture recorded waits in the main thread and CUPTI helper threads.
The native binaries do not expose enough symbols to identify the complete lock cycle.

Disabling the NIXL progress thread is not a sufficient fix.
This result weakens the hypothesis that this thread alone causes the stall.
The control does not exclude other UCX asynchronous activity.

The next control restores the original NIXL setting.
It uses an activity-only CUPTI collector without Nsight's CUDA API callback tracing.
The collector self-test recorded 200 of 200 expected kernels, including 100 CUDA Graph kernels.
It reported zero dropped records, zero invalid timestamps, and exact test output.
These values validate instrumentation coverage only. They are not model performance results.
Model-run coverage, output parity, and the unchanged 5% perturbation gate remain required.

Evidence: `state/manual_progress_control.json`, `raw/recovery_huginn_manual_progress_profile512/manual_gdb_all_threads.json`, and `tools/activity_validation.json`.

## Activity-only collector: partial recovery, not a final pass

The first activity-only run completed all 512 requests. Its outputs matched the clean reference exactly.
The response-window difference was +0.394470%, below the unchanged 5% limit.
The fixed eight-second interval had 92.7622% P kernel union and 107.167% finite-service proximity.
Kernel union is not SM occupancy or peak-FLOP utilization.

The same interval failed two required conditions.
The audited queue's half-window growth was -0.75 requests.
D ITL p99 was 205.631ms, above the frozen 192.821ms limit.
The whole-run ITL p99 was 145.361ms. This value cannot replace the failed aligned-window result.
All three individual intervals above the limit overlapped probe 8.
This temporal association does not establish causation.

The collector recorded 148,850 valid kernels in total. The aligned eight-second interval contained 142,716 kernels.
It reported no dropped records and no invalid timestamps.
The loaded library was runtime CUPTI 12.9, not Nsight's bundled CUPTI 12.8.
Thus, collector style and library version both differ from the original Nsight control.

Evidence: `state/activity_recovery_control.json` and `raw/recovery_huginn_activity_only_profile512/tail_probe_overlap.json`.

## Longer capture failure

A new control increased capture duration from eight to 32 seconds.
It retained the start fraction, request trace, model configuration, and original NIXL setting.
The execution failed during capture with CUDA `unspecified launch failure` and `NIXL_ERR_BACKEND`.
It did not reach a completed trace. Its dispatcher returned code 1 without a watchdog timeout.
This is a distinct symptom from the earlier blocking-call evidence.
The earlier user update called this a progress stall. The subsequent update corrected that description.

The error appeared at a CUDA IPC operation. CUDA errors can surface after the operation that caused them.
The exact failing operation remains `ROOT_CAUSE_INCONCLUSIVE`.
No measurement from this run qualifies as a successful performance result.
The wrapper now blocks captures longer than eight seconds unless an explicit diagnostic flag is present.

The next diagnostic tests CUPTI buffer preallocation.
NVIDIA documents dynamic buffer allocation on the application's critical path.
It recommends setting buffer preallocation before context creation. See the [CUPTI usage guide](https://docs.nvidia.com/cupti/12.9/main/main.html).
The diagnostic requests 16 buffers instead of the default three.
It first checks exact output and kernel coverage with 400,000 controlled kernel invocations.
This is a falsifiable hypothesis, not an established explanation of the launch failure.

Evidence: `logs/recovery_huginn_activity_window32_profile512.log` and `state/buffer_recovery_plan.json`.

The first buffer self-test failed before model execution with process code -11.
The new Python binding passed a size value where `cuptiActivitySetAttribute` requires a pointer to that size.
The installed CUDA header identifies this implementation error.
The binding now passes `size_t*` for both attribute APIs.
The failed self-test remains in `raw/activity_buffer_stress400k`.
Its corrected repeat uses the separate name `activity_buffer_stress400k_v2`.
This self-test error does not explain the earlier model-run launch failure.

The corrected stress test recorded all 400,000 kernels, including 200,000 graph kernels.
It reported exact output, zero dropped records, and zero invalid timestamps.
The first model diagnostic then completed its 32.0844-second capture with 539,010 records.
The complete serving run and its paired perturbation check were still pending at this checkpoint.

This control also initializes CUPTI before CUDA context creation.
The original activity-only control initialized its client during serving.
Therefore, buffer count and initialization timing are not yet isolated.
An early-initialization control with three buffers is needed to separate these effects.
The observation does not prove that buffer count alone resolves the failure.

The full diagnostic completed all 512 requests with exact output parity.
Its response-window difference was +0.0404583%.
The aligned P kernel union was 95.5207%.
Finite-service proximity was 108.954%, relative to the retained 1.516134218 requests/s reference.
The audited P-ready count had +8.671875 requests of half-window growth and a maximum of 39.
D TPOT p95 was 123.528ms. The same-window ITL p99 was 190.175ms.
The frozen ITL limit was 192.821ms.
The whole-run maximum ITL was 974.963ms. This outlier remains in the report even though the p99 screen passed.
These values are a single-start operational screen, not independent confirmation or a physical peak-throughput claim.

## Queue measurement audit

The original phase-only P-ready count was zero in the preceding diagnostic.
The ownership audit found waiting requests whose request phases changed before compute completion.
A second audit uses only pending scheduler tickets. It does not use host-service spans.
However, synchronous submissions can finish before the scheduler samples a pending ticket.
Thus, this ticket sample can omit real GPU work.

The new device audit checks each Queue timestamp against actual P kernel intervals from the same execution.
It excludes timestamp-boundary uncertainty.
Only D-credit-clear P-slot waiting at a verified active-kernel timestamp contributes to this lower bound.
It retains the original phase-only, host-ownership, and ticket-only results separately.
It does not infer GPU work from a host span.
The acceptance thresholds remain positive half-window growth and a maximum of at least two.
The independent kernel-union, service-proximity, perturbation, and Decode SLO thresholds do not change.
Final confirmation requires both the ownership audit and this direct device audit.

The new audit is declared before evaluating the independent confirmation starts.
It is an instrumentation correction, not permission to replace a failed SLO or a failed device-busy threshold.

## Whole-run tail and dual-device controls

Three independent starts passed the predeclared same-window operational gate.
This does not imply whole-run Decode SLO qualification.
Clean start 3 has ITL p99 263.483 ms, above the frozen 192.821254 ms limit.
Profiled start 2 has whole-run ITL p99 204.464 ms.
Only 3 of that profiled start's 92 violating intervals overlap the collector export.
Thus export overhead alone does not explain all tail violations.
Clean start 3 includes 341 five-row and 60 six-row recurrent submissions.
These exceed its four-row fixed-capacity graph limit.
The next controls compare D slots 4 / graph rows 4 and D slots 8 / graph rows 8.
Each changed geometry requires exact state and output qualification before a clean 512-request run.
The unchanged geometry receives a matched rerun.

The dual-device capture records P kernel union 95.504% and D kernel union 58.654% in their respective capture windows.
Its common-window ITL p99 is 213.541 ms. The same-window SLO gate fails.
Its response-window perturbation is 1.465%, but that does not erase the tail failure.
This trace can show measured device activity, but it cannot serve as a successful joint motivation figure.

Early initialization with the default three CUPTI buffers failed with a CUDA/NIXL backend error.
The sixteen-buffer path completed the independent confirmations.
This supports buffer configuration as part of the mitigation. It does not identify the exact driver failure mechanism.

One method correctness controller launched its worker just before controller cancellation.
An explicit adoption guard retained the worker and restored exclusive monitoring.
The worker wrote COMPLETE, but its parent exit status is unavailable.
The run remains CONTROLLER_INTERRUPTED_DIAGNOSTIC_ONLY and does not pass formal qualification.
All output and state evidence remain intact. A new namespace is required for the formal repeat.

