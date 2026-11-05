# Native-path continuation and 2D Token × Loop headroom PoC

## 1. Scope and time contract

This campaign starts on 2026-11-05 at 00:47:27 Asia/Seoul.
The GPU stop boundary is 05:47:27. The final closeout boundary is 06:47:27.
The six-hour budget includes implementation, experiments, analysis, publication, and resource restoration.
The parent evidence commit is `5a4ff4a8a3f13119951b4e2ff202fb3541ac1d0a`.
All new evidence uses `ouro_huginn_native_loop_continuation_2d_oracle/`.
The publication branch is `flashvep/native-loop-continuation-2d-oracle`.
An alternate Git index preserves the existing checkout and staged entries.
The external runtime remains unchanged. Candidate hooks are process-local.

This is a bounded Go/No-Go experiment. It does not implement a new dynamic scheduler.
An unsupported implementation or an expired time budget is not a performance NO-GO.

## 2. Models and frozen evaluation contract

| Model | Revision | Precision / depth | Primary input / output | Low / medium / heavy nominal QPS |
|---|---|---|---|---|
| ByteDance/Ouro-1.4B | 574fa66cb8bf5abdc979642d01cf2b79b16bfab1 | BF16 / R4 | 4096 / 32 | 1.23460875 / 3.4569045 / 6.356 |
| tomg-group-umd/huginn-0125 | bb6621b65e90b6a4b9b29ef88dc83866d450470c | BF16 / R32 | 2048 / 16 | 0.37903355455 / 1.06129395274 / 1.971 |

Use the previous model-specific token IDs and controlled output caps.
The first 128 requests have 128 unique prompt IDs. Their sources are ShareGPT and LongBench.
Preserve generation seeds, arrival offsets, and prompt order within each paired comparison.
Use open-loop Poisson arrivals. Record nominal QPS, realized arrival rate, and completed throughput separately.
Keep prefix caching disabled. Do not claim a natural-EOS workload.
Keep original weights, attention math, qualified numerical fixes, and all recurrent work.
No early exit, quantization, approximation, or weight changes are permitted.
Huginn must remain within its 4096-token context limit.

Copy the full-precision `frozen_SLO.json` from the parent campaign without rounding.
Ouro limits are TTFT 1112.9925646819174 ms, request-average TPOT 53.70702898374411 ms, and ITL p99 65.08287642151119 ms.
Huginn limits are TTFT 1963.9920117333531 ms, request-average TPOT 186.5715059141318 ms, and ITL p99 192.8212538175284 ms.
The additional-work screen uses TPOT p95 at or below 90% of the TPOT limit.
The screen also requires ITL p99 at or below its limit.
Request-level joint goodput uses the frozen TTFT and request-average TPOT limits.
Do not replace that metric with the p95 screen. Report the token-tail gate separately.

## 3. Phase A: native-path continuation

First reproduce the actual native wavefront callback rejection for grouped recurrent submissions.
Do not remove its one-depth progress assertion.
The native non-wavefront staged-progress path supports grouped depths. Audit that path before reuse.
One outer ModelRunner submission can contain several recurrent CUDA graph replays.
Do not interpret an outer submission count as a kernel count or as one monolithic CUDA graph.

Compare these paths:

1. Native Atomic PD.
2. Existing staged Native PD with no offload.
3. Existing full-snapshot PD with no offload.
4. Feature-enabled native zero-offload fallback.
5. Native-path static loop continuation.

The full-prefill request must retain the native execution path where possible.
Only a partial request may pay its state materialization and boundary cost.
The feature-enabled zero-offload comparison must also expose its KV, workspace, and graph-cache footprint.
A dispatch bypass alone does not validate a useful partial continuation.

Initial cuts are Ouro 3 and 2, and Huginn 31 and 28.
Preserve full prompt hidden rows at a partial boundary.
Preserve every valid depth-indexed KV range, Huginn Prelude/Coda state, and injection state.
Retain source buffers until the native transfer acknowledgment permits release.
Do not create a full snapshot on every request if a valid incremental path is available.
Apply a reusable native-path optimization fairly to the Token baseline as well.

Qualify each new execution path at N8 and then N32.
Compare exact output tokens, valid KV, recorded hidden state, recurrent-work accounting, transfer fingerprints, and source/ACK lifetime.
Keep internal runtime parity separate from original-HF fidelity.
Do not expand a failed path into latency experiments.

Measure low and medium load before heavy load.
The zero-offload target is a latency and throughput ratio within approximately ±5% of Native.
Use paired independent starts and uncertainty when time permits.
A single pair within 5% is a screening result, not proof of equivalence.
Separate CPU submission, graph execution, boundary synchronization, snapshot preparation, transfer, commit/ACK, source retention, and D continuation.
CUDA event enclosures and host phase durations are not exclusive kernel execution time.
Measure profiler overhead with a matched control. Reject traces with more than 5% interference.
Do not add overlapping phases or component percentiles to calculate total latency.

TTFT is arrival to first output. Queue latency is arrival to first actual Prefill execution.
Service TTFT is first actual Prefill execution to first output, including transfer and D continuation.
Calculate each request's intervals before percentiles.
Report E2E, TPOT, ITL tails, offered failures, and finite joint SLO-goodput.
Queue relief can improve serving even when isolated Service TTFT does not improve.

Phase A decisions are GO_NATIVE_PATH, CONDITIONAL, NO_GO_NATIVE_PATH, or INCONCLUSIVE.
A GO requires exact continuation and a useful measured cost or serving result.
Zero-offload improvement alone is not a continuation GO.

## 4. Phase B: dependency-aware 2D oracle

Start with one P-to-D handoff. Do not assume free tile migration or arbitrary ping-pong.
The initial candidate has three steps:

1. P computes all T prompt tokens through recurrent depth c.
2. P computes prefix [0,k) through the remaining depths.
3. D receives valid state and computes suffix [k,T) through the remaining depths.

Each tile requires the preceding-depth hidden state and same-depth causal prefix KV.
Huginn also requires valid Prelude, Coda, and injection state.
Construct the dependency graph before treating any candidate as feasible.
Record depth-indexed KV validity, state ownership, payload bytes, transfer submissions, and release order.
A structurally valid tile is not a GPU-qualified tile.

Classify actions as native-qualified, implementation-dependent, or causally invalid.
Do not give invalid actions free execution in an oracle.
Token-only, Loop-only, and 2D endpoints must share transfer costs, scheduling freedoms, and fast paths.
Token cuts may use large chunks and arbitrary legal boundaries.
Use equal tuning budgets for comparable measured controls.

Measure legal primitive costs where possible.
Include attention length, Prelude/Coda work, graph geometry, transfer, synchronization, workspace, and Decode interference.
Recurrent token rows are structural accounting, not FLOPs or elapsed GPU time.
Unmeasured combinations remain ESTIMATED or NOT_MEASURED.

Compare Native PD, Token-only, Loop-only, 2D Token × Loop, and an optimistic compute-only model.
The constrained model must include queues, P/D resource contention, batching assumptions, KV limits, and frozen Decode screens.
Do not sum independently optimal per-request latencies and call that a serving oracle.
Validate predictions against measured Native and legal fixed-cut runs.
An unvalidated model is a modeled estimate, not a Feasible Oracle.
A heuristic optimistic model is not automatically a rigorous upper bound.
Label all relaxed costs and future information explicitly.

Report Token-only versus 2D incremental headroom under the same constraints.
Separate GO_2D_HEADROOM from LOOP_SPECIFIC_NOT_PROVEN.
An identical generic logical-layer boundary can reproduce a loop cut.
No novelty claim follows from a different cut name.

## 5. Resources, evidence, and stopping rules

Use 2GPU 1P1D. Default pairs are 0/1 for Ouro and 4/5 for Huginn.
The shared-host quiet-lane lock serializes clean timing.
Stop only verified approved burners. Do not terminate unrelated user processes.
Run approved burners on idle GPUs outside clean timing.
Keep all GPUs quiet during a clean run if other-device burners can create host interference.
Restore approved burners on physical GPUs 0 through 7 after experiments.
Verify UID, command, session ownership, and actual GPU processes.

Each launch must fit startup and runtime limits before the fixed GPU stop boundary.
Preserve failed and partial jobs. Stop unsafe memory or unexplained correctness failures on the affected path.
Keep immutable configurations, source hashes/archives, GPU mappings, seeds, raw request events, outputs, state comparisons, transfer logs, memory records, and budget receipts.
Keep the real Git index, prior raw data, and external runtime unchanged.

## 6. Figures and final decision

Create five figures from measured or explicitly labeled modeled data:

1. Native versus staged versus native-path Service TTFT and submission/transfer costs.
2. Zero-offload overhead and uncertainty.
3. Loop-cut P saving, D work, migration cost, and Decode interference.
4. Native, Token-only, Loop-only, 2D, and compute-only headroom.
5. Token-position × recurrent-depth feasibility and state costs.

Mark missing results NOT_MEASURED. Do not draw invented kernel idle intervals.
Distinguish measured points, structural accounting, modeled estimates, and relaxed upper bounds.
Report per-model GO_2D_LOOP_SPECIFIC, CONDITIONAL_GO, NO_GO_LOOP_PD, or INCONCLUSIVE.
Require both models for a common-method GO.
Use a 10% improvement as an aspirational headroom target, not a reason to weaken constraints.
Commit only this campaign's code and evidence. Use ordinary push and verify the remote SHA.
The final report must state actual elapsed time, missing tests, correctness scope, memory footprint, measured costs, model validation error, limitations, reproduction commands, publication SHA, and burner restoration.

## 7. Execution deviations recorded at closeout

The last GPU dispatch ended at 05:17:52.787796 on 2026-11-05 Asia/Seoul.
No GPU experiment started after the fixed GPU deadline.
An assistant capacity interruption delayed analysis and publication beyond the overall 06:47:27 deadline.
CPU closeout resumed at 09:42:31. This restart does not create a new GPU budget.
The final report must disclose this overall time overrun.

Huginn partial Token and deeper Loop/2D qualification stopped at the unchanged Wavefront progress guard.
An inherited probe flag changed the effective D scheduler despite the requested native setting.
The closeout CPU test identifies this configuration seam. A corrected GPU path remains untested.
Ouro 2D repetitions also exposed startup-dependent graph-cache coverage.
No post-deadline GPU run tests a graph-cache correction.

The campaign provides structural dependency accounting and a measured fixed-action envelope.
It does not provide a validated dynamic Feasible Oracle or a rigorous compute-only upper bound.
Missing oracle results must remain missing. No acceptance threshold changes.
