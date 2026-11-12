# Remote-ready Cross-depth PD: bounded feasibility specification

## Objective and scope

Test whether real P-to-D continuation supplies useful recurrent work that existing Native Mixed cannot already combine with Decode.
Do not treat weight sharing, ordinary co-batching, or a different depth label as a new execution mechanism.
Keep the Dense AR campaign separate.

The first recorded audit starts at 2026-11-12 15:41:05 UTC, or 2026-11-13 00:41:05 KST.
The total deadline is 2026-11-12 19:41:05 UTC.
Do not start GPU work after 19:11:05 UTC.
Stop earlier when the explicit equivalence gate fails.
The four-hour limit is a maximum, not a required runtime.

Source namespace: `poc_flashvep/remote_ready_cross_depth_pd/`.
Evidence namespace: `poc_flashvep/reports/ouro_huginn_remote_ready_cross_depth_pd_20261113/`.
Publication branch: `flashvep/remote-ready-cross-depth-pd-poc-20261113`.
Publication parent: `367d50336791b4c540404053eadf924544a2ca0f`.
Keep the actual user checkout and index unchanged.

## Models and invariants

Ouro uses `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`, fixed R4.
Huginn uses `tomg-group-umd/huginn-0125`, revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`, fixed R32.
Keep original BF16 weights and exact Attention dependencies.
Do not introduce early exit, quantization, approximation, KV sharing across depths, or loop skipping.
Keep Huginn prelude, coda, reinjection, and depth-indexed valid KV semantics.

Reuse the pinned numerical corrections when evaluating their matched runtime.
Do not extend an internal runtime-parity result into an original-HF equivalence claim.
The old Huginn HF triage matched 5/8 prompts, so that separate limitation must remain visible.
No new GPU correctness result exists until a new run completes its own gate.

Conditional primary geometries are Ouro4096/32 and Huginn2048/16.
The default GPU pairs are Ouro0/1 and Huginn4/5.
Old RCC128/32 and diagnostic256/16 cohorts remain short mechanism evidence only.
Do not relabel them as the requested long-prompt online workload.

## Prior evidence audit

Read the frozen D-local, RCC fixed-cut, and Prefill-heavy discovery reports.
Verify the existing A import path, scheduler policy, physical model call, and attention metadata.
Read Sarathi-Serve, CDB, and FluidPD from primary sources.
Record source hashes and the exact input evidence paths.
Keep historical GPU measurements distinct from new CPU trace analysis.

## Phase A: capability and real Remote-ready work

Establish the following chain before counting a remote continuation.

1. P completed the configured cut for this transfer generation.
2. The connector transferred the valid prefix KV and hidden carrier.
3. D accepted the imported request after all committed transfer sequences completed.
4. The scheduler admitted the request to `PREFILLING_RECURRENT`.
5. A recorded D recurrent batch selected its rows together with Decode rows.

Join transfer IDs from the Coordinator metrics, Source split logs, and execution descriptors.
Exclude warmup by measured request identity and the measured time window.
Do not join requests only by seed, since warmup reuses seeds.
Check the actual workers and GPU map.

The Coordinator's `d_received` and `d_activated` times are Host message observations.
They are not CUDA completion times or exact scheduler-ready timestamps.
Activation notification can follow admission or first execution.
Do not reject a valid recorded batch merely because a later Host notification has not arrived.
Conversely, do not assume that a request remains ready between recorded executions.

Define an observed mixed opportunity as a completed recurrent submission with selected Decode and imported Prefill spans.
Define a cross-depth-only selected opportunity when no selected Prefill depth equals any selected Decode depth.
Also count Prefill rows at unmatched depths inside batches that contain both matched and unmatched depths.
These are lower-bound observations of work actually selected by the existing scheduler.
They do not enumerate every legal ready set.

Report total Decode core submissions, observed mixed submissions, same-depth-compatible submissions, and cross-depth-only submissions.
Report actual selected Remote-ready rows and requests.
Mark unused opportunities, exact ready queues, Deadline rejection counts, and per-step KV credits `NOT_MEASURED` unless the trace records them.
Do not replace missing counts with zero.

## Equivalence gate

Compare B3 and B4 with the same imported request states, work set, scheduler limits, numerical path, and graph policy.
If existing A plus Native Mixed already imports those states and dispatches the proposed mixed operation, record `EQUIVALENT_MECHANISM`.
A no-migration baseline can have less Prefill ready work, but that does not prove a new capability beyond existing A plus Native Mixed.
Do not claim that all global opportunities were used merely because the execution paths are equivalent.

The requested immediate-stop rule applies when the candidate is equivalent.
At that point, do not repeat the old batching microbenchmark or build a new scheduler.
Preserve the report, tests, evidence manifest, and resource verification.

## Conditional Phase B

Proceed only if Phase A finds real legal work that the strongest existing Mixed path cannot exploit.
Compare B0 Decode-only, B1 Separate, B2 Strong Aligned Same-depth, B3 Existing Mixed, and B4 Candidate.
Charge Same-depth alignment for real computation and waiting.
Do not disable B3 graphs or batching to create a benefit.

Measure `T_incremental = T_mixed - T_decode_only` with matched inputs and timing scope.
Compare it with separately measured Prefill cost.
Record core times, GEMM shapes, kernel counts, graph hits, Attention, packing, and Decode interference.
Record hardware traffic only when counters measure it.
Arithmetic-intensity estimates are structural accounting, not measured HBM traffic.

CUDA events measure stream enclosures, not exclusive kernel unions.
Detailed profiler traces require a matched interference screen.
Do not use a trace that changes the required tails by more than 5% as clean latency evidence.

## Conditional Phase C

Require a new observed opportunity, lower marginal cost, completion-feasible admission, and exact P-to-D state ownership.
Start with N8 and then N32.
N32 means simultaneous submission, not simultaneous GPU residency.
Keep full fixed depth and compare generated tokens, valid KV, hidden state, ACK, and retirement.
No committed state may be freed or returned without a valid protocol.

Use paired traces and at least three independent starts only after the small gate passes.
Expand to N64/N128 only when Decode remains protected.
Report finite-cohort rates separately from sustainable serving capacity.
The Token Suffix control must receive the same cross-depth functionality and safety policy.
If that control is not run, a strict loop-specific GO remains unavailable.

Frozen Ouro limits are TTFT1112.9925646819174ms, request TPOT53.70702898374411ms, additional-work TPOT48.3363260853697ms, and ITL65.08287642151119ms.
Frozen Huginn limits are TTFT1963.9920117333531ms, request TPOT186.5715059141318ms, additional-work TPOT167.91435532271862ms, and ITL192.8212538175284ms.
Also require Native-relative ITL p99 and TPOT p95 ratios at most1.05.
Report Maximum ITL and threshold exceedances separately.
Do not select new thresholds from candidate results.

## Decisions and reporting

GO requires new Remote-ready opportunity, lower marginal cost, exact state, Decode protection, and repeated benefit over strongest Mixed and equal-function Token controls.
CONDITIONAL applies when a useful prerequisite exists but the remaining benefit or safety gate is incomplete.
NO-GO on tested conditions applies to verified mechanism equivalence or measured lack of incremental benefit.
INCONCLUSIVE applies when required evidence cannot resolve the question.
Separate a novelty/equivalence stop from a physical-headroom or measurement failure.

Use `MEASURED`, `TRACE_ANALYSIS`, `OFFLINE_ORACLE`, `CORRECTNESS_ONLY`, `NOT_MEASURED`, and `NOT_RUN` explicitly.
No new GPU run is required after a verified immediate-stop condition.
Do not alter earlier reports or raw artifacts.

## Resource and publication contract

Leave approved burners running throughout CPU-only work.
Verify Process UID, Session ownership, and actual GPU activity.
If conditional timing runs are needed, obtain the existing quiet-lane lease and stop only approved burners.
Do not signal unknown processes.
Restore all eight approved burners after GPU work and verify actual generation.

Use an isolated Git index for related files only.
Commit and push a new branch without a force push.
Verify the final Remote SHA.
Check the original user index, checkout, and external runtime diff again.
