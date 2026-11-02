# Atomic Prefill PD: Final Correctness Evidence

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.


| Model | Input | Sequences | Exact output | Valid KV/hidden bytes | Expected rows | Observed rows | Gate | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ouro | 8192 | 8 | True | True | 262624 | 262624 | True | state/model_gates.json |
| huginn | 3900 | 8 | True | True | 1002240 | 1002240 | True | state/model_gates.json |
| ouro | 16384 | 8 | True | True | 524512 | 524512 | True | state/optional16k_gate.json |


The eight-request byte probes are heavy correctness measurements.
Their elapsed time is not clean Serving latency.
Work counters count active recurrent rows, not padded physical GEMM rows.

## Native AR failures


| Setting | Exact sequences | Compared | Pass | Reference | Candidate |
| --- | --- | --- | --- | --- | --- |
| Default PD 8000/200, different arrival rates | 7 | 24 | False | pilot_qwen8_l8000_o200_cfull_f0p25_v1 | pilot_qwen8_l8000_o200_cfull_f0p5_v1 |
| Official async PD 8000/200 repeat | 3 | 8 | False | diagnostic_qwen8_async_pd_l8000_o200_first_v1 | diagnostic_qwen8_async_pd_l8000_o200_repeat_v1 |

The default short 8000/16 native-PD starts matched 8/8.
That finite result did not predict longer-output reproducibility.
The default native DP2 repeat also failed at 7/8 without NIXL.

The first globally divergent internal operation remains unidentified in the original failing execution.
Token differences alone do not prove BF16 rounding or KV corruption.
The failed original AR family does not support a qualified DuetServe reproduction.

New execution-family gates appear in the extension section below.

## Independent final gates


| Model | Full repeat compared | Full repeat passed | Tuned parity | Colocated parity |
| --- | --- | --- | --- | --- |
| ouro | 32 | True | True | True |
| huginn | 32 | True | True | True |

The final qualification uses 32 requests, another arrival seed, and another sampling seed.
It uses new Prompt rows where the source pool permits them.
Earlier byte checks remain separate evidence.

A missing or failed gate never becomes a performance pass.

## Complete gate and failure records

See `state/live_correctness_audits.jsonl`, `state/online_correctness_blockers.json`, and `configuration_matrix.csv`.
Workspace, wire-credit, chunk, Native Mixed, and optional-context gates have separate JSON records in `state/`.
Failed runs retain their complete logs and source archives.

## User extension: final deadline 2026-11-02 11:00 KST

Ouro and Huginn discovery had priority. The user also requested AR reproducibility work.
No new GPU job starts after 10:05 KST. Active owned measurements stop by 10:15 KST.
At 09:47 KST, the start cutoff moved five minutes later. The earlier budget file remains preserved.

The GPU stop, final deadline, correctness gates, and frozen SLOs did not change.
The remaining time is reserved for evidence review, publication, and resource restoration.
The original SLOs and bottleneck gates remain unchanged.

The new native AR execution family has a separate calibration file.
Its thresholds do not replace the failed default-family thresholds.

### AR reproduction by actual execution setting


| Setting | Exact sequences | N | Pass | Evidence |
| --- | --- | --- | --- | --- |
| Default PD, changed QPS | 7 | 24 | False | MEASURED_NATIVE_GPU_OUTPUT |
| Default arithmetic, Eager PD | 9 | 24 | False | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager PD, changed QPS | 24 | 24 | True | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager, independent N32 repeat | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager, C1024 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager, C4096 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager, PD versus DP2 | 15 | 32 | False | MEASURED_NATIVE_GPU_OUTPUT |
| True BI + Eager, DP2 repeat | 21 | 32 | False | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, graph NONE, PD repeat | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, graph NONE, PD versus DP2 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, graph NONE, DP2 repeat | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, graph NONE, C4096 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, P4 startup recheck, Full | NOT_MEASURED | 32 | False | NOT_COMPARABLE_INPUT_OR_SEED |
| BI compiled, P4 startup recheck, C4096 | NOT_MEASURED | 32 | False | NOT_COMPARABLE_INPUT_OR_SEED |
| BI compiled, P4 corrected-input recheck, Full | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, P4 corrected-input recheck, C4096 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, P4 corrected-input recheck, DP2 | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |
| BI compiled, P4 corrected-input recheck, DP2 repeat | 32 | 32 | True | MEASURED_NATIVE_GPU_OUTPUT |

These tests establish only the stated internal sequence comparisons.
An N32 PD pass does not prove DP2 parity, original-HF equivalence, or downstream accuracy.
Later performance runs also have an exact common-input audit. A new divergence blocks that execution family.

The first compiled QPS pilot failed during P-worker startup, before any request completed.
The child log reports a CUDA illegal-memory-access error during native kernel warm-up.
The capacity selector reduced engine request capacity from the qualified four to one.

A finite optimal batch size does not certify a smaller engine request capacity.
The new P4 checks restore the earlier qualified request capacity. They do not modify native warm-up or model arithmetic.
The exact failing GPU operation remains unlocalized. One startup failure does not establish that P1 alone caused it.

The failed namespace and the earlier family blocker remain preserved. P4 performance uses a separate family label.
The earlier independent finite-service references remain diagnostic references, not measured P4 sustainable capacity.
The v2 rechecks selected the default test pool, not the earlier qualification subset.

All 32 input rows differed. Those original-reference comparisons cannot diagnose numerical failure.
Corrected v3 checks explicitly select the earlier dataset pool. The invalid comparisons remain preserved.

The bounded AR search defines 9 coarse screens across three methods and three rate factors.
Completed coarse screens: 9/9.
The immutable exploratory plan retains 21 configurations across all seven requested factors.

Confirmation takes priority after the coarse grid. Unstarted extra factors and DP2 final repeats remain explicit budget omissions.

The first extension BI-labelled v1 trials actually used `ar_batch_invariant=False`.
The configuration constructor overwrote the explicit request. A regression test detected this bug.
The constructor now preserves an explicit flag. New v2 namespaces record the effective native flag.

The incorrectly labelled v1 data remain preserved. They cannot prove that BI failed.
Earlier genuine BI failures also remain preserved. This correction does not erase those failures.
See `state/ar_extension_BI_configuration_correction.json` for the affected configurations.

### Numerical probes and their limits

Evidence type: MEASURED_OPERATOR_MICROBENCHMARK and MEASURED_COLD_PREFIX_PROBE.
The probes use the actual Qwen checkpoint and captured operands. They do not use a proxy model.
Default arithmetic changes some tested O/Down outputs when only GEMM row geometry changes.

True BI v2 produces exact selected projection rows in the tested M=1/2/3/4/8/16/32/256 cases.
This is an operator result, not proof of complete incremental-Decode invariance.
The cold common-prefix probe selects the same final target rows in all tested batch orders.

Its two leading BF16 logits are both 33.75. The recorded margin is zero.
That probe does not reproduce the incremental-Decode divergence.
The first globally divergent operation remains NOT_AVAILABLE unless a later targeted trace identifies it.

### Qualified Looped runtime controls


| Model | Family | Input | Stage | N8 state/output gate | Tuned chunk gate | Reference file |
| --- | --- | --- | --- | --- | --- | --- |
| huginn | async_g8 | 3500 | UNSUPPORTED_OR_IGNORED_ASYNC_REQUEST | True | NOT_MEASURED | NOT_MEASURED |
| huginn | sync_g4 | 3500 | PILOTS_COMPLETE | True | True | capacity_references_huginn_sync_g4_extension.json |
| huginn | async_g4 | 3500 | UNSUPPORTED_OR_IGNORED_ASYNC_REQUEST | True | NOT_MEASURED | NOT_MEASURED |
| ouro | P8_B32768_K17408 | 8192 | PILOTS_COMPLETE | True | True | capacity_references_ouro_p8_b32768_k17408_extension.json |
| ouro | P8_B65536_K17408 | 8192 | PILOTS_COMPLETE | True | True | capacity_references_ouro_p8_b65536_k17408_extension.json |
| ouro | P2_L16384_O32 | 16384 | PILOTS_COMPLETE | True | True | capacity_references_ouro_p2_l16384_o32_extension.json |
| huginn | async_l3900_g4 | 3900 | UNSUPPORTED_OR_IGNORED_ASYNC_REQUEST | True | True | NOT_MEASURED |
| huginn | sync_l3900_g4 | 3900 | PILOTS_COMPLETE | True | True | capacity_references_huginn_sync_l3900_g4_extension.json |

The pinned Huginn engine supports Sync execution only. Graph-row limits of four and eight remain supported controls.
The requested Async path was not active in the earlier successful native-PD trials.
The inherited launcher forwards the Async flag only for Ouro. Direct Huginn Async construction fails its native guard.

The correction preserves those actual Sync measurements and retires unstarted unsupported or redundant jobs.
Explicit Sync v2 namespaces provide the corrected comparisons. No engine guard is removed.
New `effective_execution_*.json` files record the actual StageEngine flags before measurement.

Earlier effective Sync attribution is SOURCE_AUDIT, not a recorded startup flag.
The fixed-depth model and numerical projection fixes remain unchanged.
Ouro also tests larger P budgets and a smaller legal source KV pool.

A graph-row limit is not a claim that every physical GEMM has that row count.
Ouro 16384/32 is a prospectively declared extension cell. The earlier 16384/100 cell remains preserved.
Independent service references remain specific to the execution family. Their costs are not pooled.

The native connector allocates transfer slots separately from compute slots.
P transfer slots equal P compute slots plus `max_draining_requests`, which defaults to eight.
D transfer slots also include `max_receiving_requests`, which defaults to 32.

Therefore a finite optimal P batch is not the same variable as the native transfer-slot count.
KV reservations can bind before either slot limit. The report keeps those refusals separate.
The source-credit controls compare fewer draining slots at unchanged compute slots and wire credit.

A slower P configuration cannot pass by creating a queue. It must retain near-reference P service and actual P kernel saturation.

Huginn D4 uses an explicit preparation-only ACK control.
Each valid import can release its source allocation before the full D cohort is ready.
The existing D barrier still blocks all Decode until every import is valid.

This separates the three-context source pool from the legal four-context D pool.
Its finite D timing is not a measured Serving speedup.

### Native wire-credit controls after workspace qualification


| Model | Stage | Exact state/output gates | Pilot runs |
| --- | --- | --- | --- |
| ouro | PILOTS_COMPLETE | {"ext_wire_ouro_credit1g_qualification8_v1":true,"ext_wire_ouro_credit8g_qualification8_v1":true} | 8 |
| huginn | PILOTS_COMPLETE | {"ext_wire_huginn_credit1g_qualification8_sync_v2":true,"ext_wire_huginn_credit8g_qualification8_sync_v2":true} | 8 |

These controls change only supported native credit limits. They do not create a new transport protocol.
The 64-MiB fragment limit, payload values, and model computation remain unchanged.

### Independent confirmation and deadline limits


| Model | Stage | Selected pilot | Repeat N32 gate | Chunk N32 gate | Planned final runs |
| --- | --- | --- | --- | --- | --- |
| huginn | CONFIRMATION_COMPLETE | ext_huginn_p1_drain0_credit8g_c2048_f1p15_pilot24_sync_v2 | True | True | 18 |
| ouro | CONFIRMATION_COMPLETE | ext_ouro_l16384_p2_drain1_credit32g_c16384_f1p25_pilot24_v1 | True | True | 18 |
| qwen8 | FINAL_RUNS | ext_pilot_qwen8_bi_compiled_none_pd_g1_c4096_f1p0_p4_v2 | True | True | 27 |

All screened candidates remain in the selection JSON files. The selected N24 run is exploratory.
Ouro also profiles its best N24 candidate at each extension input length before final selection.
Only an exact matched profile within the original 5% overhead limit can guide that selection.

Those exploratory profiles do not replace the N32 or N128 gates.
Ouro and Huginn each completed three center-rate clean starts for both selected chunk settings.
Extra heavy traces can be omitted after their clean P-service prerequisite fails.

The queue retains one root-cause trace per chunk, attempted traces, and every service-qualified profile.
The omitted namespaces remain NOT_RUN_FAILED_STRONG_P_SERVICE_PREREQUISITE in the ledger.
A final TRUE verdict still requires three independent starts, actual P kernels, and finite native D-margin validation.

If P compute does not qualify, one small native staggered-D test can still answer the separate D-reserve question.
Its purpose is D_RESERVE_ONLY_WITHOUT_QUALIFIED_P_COMPUTE. A pass does not establish a Prefill-heavy regime.
A queued final run or an independent capacity ratio cannot satisfy this gate.

Missing or deadline-truncated conditions remain NOT_MEASURED or NOT_RUN_USER_11H_DEADLINE.
At the final budget review, 53 unstarted optional Looped tail jobs were omitted.
Every unstarted tail job used the same priority rule. No active GPU experiment stopped for this budget change.

The ledger marks them NOT_RUN_EXTENSION_PRIORITY_TIME_BUDGET. Their immutable configurations remain preserved.

### Profiler failures and their scope


| Profile run | Model | Completed responses | Offered requests | Stop reason | Return code | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| ext_profile_huginn_l3500_c2048_r1_s1_v1 | huginn | 80 | 128 | PROFILE_NO_RESPONSE_PROGRESS_180S | -9 | raw/ext_profile_huginn_l3500_c2048_r1_s1_v1/profile_watchdog.json |
| ext_profile_huginn_l3500_cfull_r1_s2_v1 | huginn | 14 | 128 | PROFILE_NO_RESPONSE_PROGRESS_180S | 143 | raw/ext_profile_huginn_l3500_cfull_r1_s2_v1/profile_watchdog.json |
| ext_profile_ouro_l16384_cfull_r1_s1_v1 | ouro | 30 | 128 | PROFILED_RUN_NO_PROGRESS_WITH_IDLE_GPUS | 143 | raw/ext_profile_ouro_l16384_cfull_r1_s1_v1/manual_no_progress_stop.json |
| ext_profile_qwen8_l8000_c4096_r1_s1_v1 | qwen8 | 33 | 160 | PROFILE_NO_RESPONSE_PROGRESS_180S | -15 | raw/ext_profile_qwen8_l8000_c4096_r1_s1_v1/profile_watchdog.json |

These runs stopped after response progress ceased or after a bounded profiler limit.
The registered owned child sessions stopped. Other users' processes remained unchanged.
Partial responses and profiler files remain preserved. They do not establish complete-run throughput or kernel saturation.

Their matched clean runs remain separate evidence. A profiler-associated stall does not identify its root cause.
The failed Nsight runs do not prove a CUDA, NIXL, or model error.
After the near-deadline AR stall, 4 unstarted AR profiles were omitted.

Every clean start and attempted trace remains preserved. This omission cannot support a three-profile TRUE claim.
The CPU process observer also failed once during a /proc exit race. The observer now handles that race.
Its regression test passed. That observer failure did not stop the GPU campaign.

All main generation is controlled with `ignore_eos=True`.
Internal exact token parity does not establish original-HF fidelity or downstream answer accuracy.
No Dynamic A, model-depth reduction, approximation, quantization, or training runs in this study.

Finite-cohort throughput is not infinite-horizon sustainable capacity.
