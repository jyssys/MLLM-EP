# Atomic Prefill PD: Final Crossover Summary

Compiled at 2026-11-02T01:28:43.086700+00:00. Elapsed time: 21.426 hours.

Branch: `flashvep/looped-pd-atomic-prefill-qps-crossover`.

Compilation source commit: `147c4dc48215dd82cc85f77b044ff27325ea84fc`.

Each measured run records its own source archive, source commit, and configuration hash.
The final publication commit can differ from the measurement commit.
Use `git log -1 --format=%H` on this branch to obtain the publication SHA.

Actual analysis-tree SHA256: `a392ed79e603c437d00c3575d39166f104f681e2144fe757f6ef0524e7bc5c18`.

The compilation HEAD can precede uncommitted analysis changes.
See `state/analysis_source_manifest.json` for the exact file hashes.

Search terminal state: `STOPPED_USER_11H_DEADLINE`.


| Model | Verdict | Qualified final runs | Classified PD runs | Observed categories |
| --- | --- | --- | --- | --- |
| ouro | NO_CONFIRMED_PREFILL_HEAVY_ON_TESTED_GRID | 37 | 259 | {"MIXED_OR_INCONCLUSIVE":77,"DECODE_BOUND":18,"KV_ADMISSION_BOUND":162,"GENERIC_TUNING_ARTIFACT":2} |
| huginn | NO_CONFIRMED_PREFILL_HEAVY_ON_TESTED_GRID | 6 | 137 | {"MIXED_OR_INCONCLUSIVE":12,"KV_ADMISSION_BOUND":125} |
| qwen8 | NO_CONFIRMED_PREFILL_HEAVY_ON_TESTED_GRID | 5 | 14 | {"DECODE_BOUND":1,"KV_ADMISSION_BOUND":3,"MIXED_OR_INCONCLUSIVE":10} |


The strict TRUE label requires three independently confirmed starts at the same cell and rate.
A negative tested-grid result is not proof that recurrent models cannot become Prefill-heavy.
A correctness blocker is not a negative bottleneck result.

## Separate P service pressure from P compute saturation


| Model | Final qualified PD runs | P backlog plus D SLO/KV/finite-rate margin | Near strongest P service | Valid matched P-kernel measurements | P kernel at least 80% |
| --- | --- | --- | --- | --- | --- |
| ouro | 28 | 10 | 0 | 1 | 0 |
| huginn | 6 | 6 | 2 | 0 | 0 |
| qwen8 | 5 | 1 | 5 | 1 | 1 |

These counts can differ. A P service queue does not establish GPU compute saturation.
The D margin column is a screen. It does not replace the separate native staggered-D validation.
A missing matched profile is NOT_MEASURED, not a measured low-busy result.

### Joint conditions in each selected independent start


| Model | Input/output | C | Start | P backlog grows | Near strongest P service | Valid P kernel busy (%) | D credit clear | D TPOT margin | Matched finite D test | Class |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | 3500/32 | 2048 | 0 | True | False | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| huginn | 3500/32 | 2048 | 1 | True | True | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| huginn | 3500/32 | 2048 | 2 | True | False | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| ouro | 16384/32 | 16384 | 0 | True | False | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| ouro | 16384/32 | 16384 | 1 | True | False | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| ouro | 16384/32 | 16384 | 2 | True | False | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |
| qwen8 | 8000/200 | 4096 | 0 | False | True | 93.6353 | True | False | False | DECODE_BOUND |
| qwen8 | 8000/200 | 4096 | 1 | False | True | NOT_MEASURED | False | True | False | KV_ADMISSION_BOUND |
| qwen8 | 8000/200 | 4096 | 2 | True | True | NOT_MEASURED | True | True | True | MIXED_OR_INCONCLUSIVE |

Every online P/D condition must hold in the same Serving start.
The separate D diagnostic must also match its configuration and stress rate.
The table must not combine different starts into one positive result.

The Qwen start-zero profile records 93.6353% P kernel busy within the matched overhead limit.
That clean start fails the P-backlog and D-TPOT-margin conditions. It does not establish TRUE_PREFILL_HEAVY.
Another Qwen start passes the P-service and D-margin screens, but its matched P profile is unavailable.

These separate observations do not establish simultaneous P saturation and safe D reserve.

## Measured center-rate examples


| Model | Execution family | Input/output | Arm/C | Offered N | Realized QPS | TTFT p95 ms | Request TPOT p95 ms | Token ITL p99 ms | SLO attainment | SLO goodput req/s | Classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | HUGINN_SOURCE_DRAIN_CONTROL_SYNC | 3500/32 | PD_G1/C2048 | 128 | 1.0565 | 99444.7078 | 109.4074 | 143.8288 | 0.0234 | 0.0130 | MIXED_OR_INCONCLUSIVE |
| huginn | HUGINN_SOURCE_DRAIN_CONTROL_SYNC | 3500/32 | PD_G1/C3500 | 128 | 1.0565 | 55906.2436 | 135.6001 | 316.1604 | 0.0234 | 0.0162 | MIXED_OR_INCONCLUSIVE |
| ouro | P2_L16384_O32_BOUNDED_DRAIN | 16384/32 | PD_G1/C4096 | 128 | 1.3530 | 83535.8524 | 48.3344 | 64.4375 | 0.0234 | 0.0164 | MIXED_OR_INCONCLUSIVE |
| ouro | P2_L16384_O32_BOUNDED_DRAIN | 16384/32 | PD_G1/C16384 | 128 | 1.3530 | 140990.3289 | 48.1039 | 73.0355 | 0.0078 | 0.0041 | MIXED_OR_INCONCLUSIVE |
| qwen8 | BI_COMPILED_NO_GRAPH_P4 | 8000/200 | PD_G1/C4096 | 160 | 3.5802 | 6000.8841 | 56.8058 | 196.0436 | 0.0437 | 0.1290 | DECODE_BOUND |
| qwen8 | BI_COMPILED_NO_GRAPH_P4 | 8000/200 | PD_G1/C8000 | 160 | 3.5802 | 3708.4276 | 43.0368 | 241.4913 | 0.0813 | 0.2553 | KV_ADMISSION_BOUND |
| ouro | ORIGINAL | 8192/32 | DP/C8192 | 128 | 2.7670 | 10426.7903 | 237.5160 | 1611.4367 | 0.0156 | 0.0356 | COLOCATED_CONTROL |
| ouro | ORIGINAL | 8192/32 | PD_G1/C4096 | 128 | 2.7670 | 76066.6881 | 44.5866 | 88.8134 | 0.0000 | 0.0000 | KV_ADMISSION_BOUND |
| ouro | ORIGINAL | 8192/32 | PD_G1/C8192 | 128 | 2.7670 | 38192.4257 | 52.0791 | 138.8182 | 0.0156 | 0.0234 | KV_ADMISSION_BOUND |

Each row is independent start zero at its declared center rate.
These are individual-run percentiles, not pooled percentiles or averages across starts.
The complete table retains every start in `serving_runs.csv`.

Paired server-start uncertainty appears in `paired_start_comparisons.csv`.

## Answers to the research questions

### Q1. Qwen3-8B and the DuetServe regime

The original 8000/200 native-PD family failed exact output reproducibility across arrival rates.
The original official-async diagnostic also failed.
The extension tested separate native execution families instead of treating those failures as resolved.

Use the final per-family equality and QPS tables below.
The original default target rates were about 1.1433 and 2.2866 requests/s.
A newly qualified family uses its own measured reference rates and frozen calibration.

Its costs cannot be pooled with the failed default family.
[DuetServe Figure 2](https://pages.cs.wisc.edu/~markhill/papers/icml2026_DuetServe.pdf) uses Qwen3-8B, two H100 GPUs, and 8000/200 tokens.
Its aggregated budget is 8192 tokens, and its PD backend is Dynamo.

Our native vLLM/NIXL backend and latency-SLO protocol differ.
No engine difference is attributed solely to recurrence.

### Q2–Q3. Ouro and Huginn

Use the model verdict table and per-run gate table above.
Ouro supports qualified internal 8K and 16K Prompt execution.
Huginn stays within 4096 total tokens and tests outputs as short as 32.

Shorter outputs improve the independent P/D rate ratio.
That ratio alone does not prove usable online D reserve.
KV reservation and native worker progress can limit admission before pure P compute saturation.

### Q4. Atomic versus tuned chunked PD

Atomic means one actual `[0,L)` Prompt span with full recurrent depth.
It does not mean one kernel or one transfer fragment.
The strongest finite C and the best measured Serving C are separate selections.

Wire credit, D workspace, KV pool, and Native Mixed geometry remain explicit controls.
Their effects cannot be attributed to the token-chunk label.
See the paired final table and `paired_start_comparisons.csv` for actual same-arrival comparisons.

The qualified RLT fixes still use 256-row tiles for selected Down projections.
A larger logical Prompt chunk does not enlarge those individual GEMMs.
Other projections and Attention can still respond to the token batch size.

This execution detail prevents a model-only interpretation of the chunk comparison.

### Q5–Q6. First saturation and usable D margin

`bottleneck_classification.csv` preserves every failed and passed prerequisite.
A high TTFT or a busy P GPU is not sufficient.
The decisive evidence includes ready backlog, actual P service, P kernels, D credit, and Decode SLOs.

The finite prepared-D stress tests appear in `D_margin_summary.json` when eligible.
A missing test is NOT_MEASURED, not a reserve pass.
Even a passing resident-KV probe does not certify arbitrary Prefill-continuation admission.

| Model | Evidence | Requests | Staggered ready rate req/s | Request TPOT p95 ms | 0.9 x frozen TPOT limit ms | Exact output parity | Valid KV byte-check requests | Diagnostic passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| huginn | MEASURED_NATIVE_PREPARED_STAGGERED_D | 4 | 0.8701 | 127.6840 | 173.5576 | True | 4 | True |
| ouro | MEASURED_NATIVE_PREPARED_STAGGERED_D | 4 | 0.8720 | 44.4653 | 81.5864 | True | 4 | True |
| qwen8 | MEASURED_PREFIX_PREPARED_STAGGERED_D | 32 | 3.9665 | 31.7221 | 44.8850 | True | NOT_MEASURED | True |

The Looped tests import actual native NIXL state before the timer. Each test uses four resident requests.
The Qwen test uses 32 requests with official resident-prefix preparation. It is not a native-NIXL-import test.
These are finite D-reserve measurements, not online sustainable capacity or safe Prefill-offload admission.

### Q7. Architectural versus runtime causes

Causal attribution remains INCONCLUSIVE.
The engines, kernels, numerical geometry, memory layouts, and admission rules differ.
Huginn's unused uniform KV planes are a measured allocation issue.

P-sized D workspaces and wire-credit limits are generic runtime controls.
None of these facts proves that recurrence itself prevents Prefill-heavy Serving.
The pinned configurations also differ in logical Attention depth and KV-head width.

Full-valid token accounting gives 147,456 bytes for Qwen, 786,432 for Ouro, and 2,787,840 for Huginn.
Huginn's uniform layout reserves 5,406,720 bytes per token.
These are structural storage counts, not measured latency ratios or isolated recurrence effects.

### Q8. Is Dynamic A justified?

No repeated strict TRUE cell qualifies a new Dynamic-A performance claim in this completed evidence.
Do not start a large Dynamic-A campaign from independent capacity ratios alone.
Prioritize the documented correctness, admission, or runtime limitations before another policy study.

No Dynamic A or new offload mechanism is implemented here.

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

## Completion and limitations


| Configuration status | Count |
| --- | --- |
| COMPLETE | 766 |
| FAILED | 15 |
| NOT_RUN | 1 |
| NOT_RUN_BLOCKED_CORRECTNESS | 55 |
| NOT_RUN_DUPLICATE_DIAGNOSTIC | 1 |
| NOT_RUN_EXTENSION_PRIORITY_TIME_BUDGET | 53 |
| NOT_RUN_FAILED_STRONG_P_SERVICE_PREREQUISITE | 6 |
| NOT_RUN_REPEATED_PROFILE_STALL_TIME_BUDGET | 4 |
| NOT_RUN_SUPERSEDED_AR_EXECUTION_FAMILY | 5 |
| NOT_RUN_SUPERSEDED_METADATA | 2 |
| NOT_RUN_UNSUPPORTED_OR_IGNORED_HUGINN_ASYNC | 11 |
| NOT_RUN_USER_11H_DEADLINE | 31 |


All main generation is controlled with `ignore_eos=True`.
Internal exact token parity does not establish original-HF fidelity or downstream answer accuracy.
No Dynamic A, model-depth reduction, approximation, quantization, or training runs in this study.

Finite-cohort throughput is not infinite-horizon sustainable capacity.
All unsuccessful and unrun settings remain in `configuration_matrix.csv` and `requested_search_grid.csv`.
Raw request/token evidence, source archives, and full failure logs remain preserved.

SM occupancy, full D graph-internal kernel breakdown, original-HF fidelity, and downstream accuracy remain NOT_MEASURED.

## Files and reproduction

See `looped_pd_atomic_prefill_reproduce.md` for commands and publication checks.
The evidence root contains all raw CSV/JSONL, source archives, configuration hashes, figures, and failure logs.

Selected figures:

- [Qualified AR exploratory TTFT grid, N24 per point](looped_pd_atomic_prefill_qps_crossover/figures/qwen8_l8000_o200_qps_pilot_BI_COMPILED_NO_GRAPH_P4_a_qps_ttft.png)
- [Ouro 16K/32 finite final TTFT results](looped_pd_atomic_prefill_qps_crossover/figures/ouro_l16384_o32_final_repeat_P2_L16384_O32_BOUNDED_DRAIN_a_qps_ttft.png)
- [Huginn 3500/32 finite final TTFT results](looped_pd_atomic_prefill_qps_crossover/figures/huginn_l3500_o32_final_repeat_HUGINN_SOURCE_DRAIN_CONTROL_SYNC_a_qps_ttft.png)
- [Measured AR request timeline and P-only kernel intervals](looped_pd_atomic_prefill_qps_crossover/figures/ext_profile_qwen8_l8000_c4096_r1_s0_v1_e_actual_request_timeline.png)
