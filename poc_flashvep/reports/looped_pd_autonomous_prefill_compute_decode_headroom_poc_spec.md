# Autonomous Looped PD Compute and Decode Headroom Campaign

## 1. Execution contract

The user instruction dated 2026-11-03 defines this campaign.
This specification makes its execution gates explicit.
The experiment starts at 2026-11-03 02:20:45 KST.
The hard completion deadline is 2026-11-03 11:20:45 KST.
This deadline includes analysis, resource restoration, and verified publication.
No prior campaign deadline applies.

Stop new long GPU jobs after 09:20:45 KST.
All measured GPU jobs must stop by 10:05:45 KST.
Reserve at least 75 minutes for artifacts, reports, and push.
A short job can start later only when its explicit timeout ends before the GPU cutoff.
The controller must enforce both cutoffs from persisted absolute timestamps.
The controller must not extend them after an experiment fails.

## 2. Models, source, and preservation

Use only `ByteDance/Ouro-1.4B`, fixed R4, and `tomg-group-umd/huginn-0125`, fixed R32.
Preserve BF16 weights, BF16 KV, complete fixed-depth work, original sampling, and the qualified numerical fixes.
Ouro revision is `574fa66cb8bf5abdc979642d01cf2b79b16bfab1`.
Huginn revision is `bb6621b65e90b6a4b9b29ef88dc83866d450470c`.
Huginn input plus reserved output must not exceed 4096 tokens.
Do not train, approximate, quantize, skip work, or change model weights.

The release baseline is `2d683cce8ebc926c9d410944b705cbcb6f6031a0`.
The main worktree can remain on its existing branch to preserve its staged index.
Create a separate branch from the release baseline without checking it out over dirty files.
Publish only this campaign's source, specification, report, and evidence through a temporary Git index.
Record the actual worktree HEAD separately from the release baseline.
Archive all actual executed source and the external runtime diff for each job.

The qualified external engine is `/home/esjung/external/looped-pd-abc-rlt`.
Keep its installed files read-only during the first controls.
Use process-local, independent flags for any tested runtime changes.
Preserve all previous raw data and reports.
Record the original staged-index signature and external-runtime diff signature.
Refuse automatic publication if either signature changes unexpectedly.

## 3. Starting evidence

Read these complete reports before preparing GPU jobs:

- `looped_pd_kv_admission_profiling_adaptive_prefill_headroom_final_summary.md`
- `looped_pd_atomic_prefill_qps_crossover_final_summary.md`
- `looped_pd_atomic_prefill_qps_search.md`

Inspect the selected raw traces, queue records, memory tables, and figure manifest.
Treat previous numbers as previous-campaign measurements, not new repeated results.
The qualified Ouro N128 interval has 66.582% P kernel union and 78.800% finite-service proximity.
The qualified Huginn N24 interval has 61.855% P kernel union and binding D KV reservations.
Neither result passes the target.

Read LoopServing as a source reference.
Inspect its scheduler, graph, KV ownership, and recurrent execution paths.
Do not replace the native RLT engine or copy an incompatible ABI.
Any reused implementation must retain its license and pass the new state/output gates.

## 4. Ordered hypotheses

1. Fixed 256-row projection tiles create avoidable CPU submission gaps.
   Prediction: replaying the same tile operations reduces host gaps and improves completed Prefill service at unchanged exact outputs.
2. Huginn's long-request reservations and graph/active-row mismatch restrict useful concurrency.
   Prediction: a shorter legal prompt or a supported matched graph capacity reduces binding refusals without sacrificing Decode tails.
3. Native descriptor preparation, fragment progress, and ACK-held slots restrict P progress after computation ends.
   Prediction: supported descriptor/fragment controls reduce exposed host handoff or source-retention spans at unchanged wire payload and state.
4. Other launch, metadata, or synchronization work dominates the remaining gaps.
   Prediction: trace-correlated API spans or focused host samples identify those intervals after the first controls.

These are falsifiable hypotheses, not measured causes.
Keep a decision ledger with evidence, next action, predicted effect, actual result, and rejection reason.

## 5. Measurement definitions

Kernel union is the union of recorded device-kernel intervals divided by the observation duration.
It is not SM occupancy, achieved bandwidth, or exclusive E2E compute percentage.
Record copies and memset separately when the collector exposes them.
Do not add overlapping activity categories into exclusive wall time.
Absence of an activity record is not proof of usable idle time.

Measure host CUDA API spans inside observed GPU gaps when available.
Separate launch APIs, synchronization/query APIs, copies, and unattributed intervals.
An overlap between a CPU span and a GPU gap does not by itself prove causality.
Confirm important causes with matched one-variable controls.
Keep profiling and clean serving as separate starts.

Retain the fixed 5% matched response-wall perturbation gate.
A rejected profile remains diagnostic and cannot supply a clean-start saturation percentage.
Prefer short graph-internal capture windows in actual steady traffic.
If a bounded collector returns no kernels, report NOT_MEASURED.
Do not translate an empty collector output into zero utilization.

Use Nsight device metrics only if the installed tool and permissions support them.
Record SM active/cycle and memory-bandwidth fields only when actually measured.
Do not change system permissions to enable counters.
Operator-counter measurements remain operator evidence, not serving-wide saturation proof.

## 6. Correctness gates

Every changed arithmetic execution path first needs a differential operator test with actual model weights.
Compare identical operands under baseline and candidate execution.
Include rows 1/2/3/4/8/16/256/512/1536/2048 and the selected long-Prefill geometry where legal.
Record exact elements, maximum absolute error, relative L2 error, and greedy/logit checks when available.
Include repeated calls, input changes, batch order, and result-buffer lifetime.
No graph may reuse a stale activation or overwrite a still-live output.

Then require native state/output tests before performance expansion:

- A small 2-request valid-KV/hidden byte gate.
- An 8-request exact output and full-work gate.
- A 32-request independent output/order check for a new optimized serving path.
- Separate Huginn Prelude/Coda/injection and valid-depth checks through the qualified native path.
- Actual recurrent-row accounting, excluding padded GEMM rows and startup requests.

Any unexplained state/output failure blocks only the affected candidate.
Retain the working baseline and continue independent safe diagnoses.
Internal exact parity does not establish original-HF fidelity or downstream accuracy.
Do not describe unmeasured GSM8K/MATH accuracy as preserved accuracy.

## 7. Safe optimization boundaries

Reuse the existing numerical patch unchanged as the arithmetic reference.
One candidate can capture groups of its existing 256-row Linear operations in a CUDA graph.
This candidate must preserve each original GEMM geometry, weights, padding values, and output order.
Copy results into independent storage before reuse.
Do not use a larger GEMM if it changes the qualified BF16 result.
Fall back to the original operation inside an existing graph capture or for unsupported shapes.
Warm all candidate graph shapes before measurement.
Record any measured-phase capture separately and exclude that start from a clean steady-state claim.
Bound graph-memory caches and record their HBM cost.

Supported KV pools, slots, graph-row capacities, token budgets, descriptor limits, and wire credits can change independently.
Retain native ACK, producer events, destination validity, and source-buffer lifetime rules.
No delayed-D-reserve or compact-KV allocator can enter performance evaluation without its own byte/state gate.
Do not reduce useful batching only to manufacture high P utilization.
Compare actual completed useful work and E2E performance, not kernel busy alone.

## 8. Adaptive workload search

Ouro starts from 8192/32 as the repeated reference.
Candidate inputs are 4096, 6144, and 8192 tokens.
Candidate outputs are 32, 64, and 100 where specified by the user.
Use 16384 only as a last diagnostic after the primary range lacks a valid candidate.

Huginn candidates are 1536/32, 2048/16, 2048/32, 2560/16, 2560/32, 3072/16, 3072/32, 3500/32, and 3900/32.
Prioritize a shorter prompt when D reservations bind before P compute.
Prioritize real P batching or a larger legal prompt when admission is clear but P pressure is absent.
Never silently change the context limit or truncate an allegedly full prompt.

Reuse pinned public ShareGPT/LongBench tokens for matched controlled tests.
Record source IDs, source hashes, tokenizer revision, actual token counts, and the explicit prefix operation.
Controlled ignore-EOS lengths are not natural generation or original Azure prompts.
A new public code or long-QA corpus is optional after the critical runtime tests.
Do not spend the main budget acquiring a new dataset.

## 9. Capacity and online arrivals

Measure finite P-only and actual prepared-D service after each material execution change.
Separate GPU event spans from host completion time.
Do not transfer a reference across changed projection or kernel execution settings without measurement.
Use these references to choose initial online rates, not to claim sustainable capacity.

Use open-loop Poisson arrivals with persisted offsets.
Record nominal QPS, realized scheduled QPS, realized submission QPS, and completed request rate separately.
All paired policies receive identical prompts, generation seeds, caps, and arrival offsets.
Initial pilots normally contain 24 requests.
The search selects the next rate from actual backlog, refusals, service, and SLO evidence.
Do not blindly execute a Cartesian product.
Bound every run by count, progress watchdog, absolute campaign cutoff, and memory checks.

Final promising cells need at least 128 requests and three independent process starts.
Cross GPU pairs and alternate policy order when feasible.
The same prompts across starts support runtime repeatability, not new-domain generalization.
Report independent-start 95% paired uncertainty. Do not treat tokens as independent server starts.

## 10. SLO contract

Reuse the previous frozen per-model/per-input TTFT, TPOT, and token-ITL thresholds wherever available.
Retain their original calibration provenance.
Output-length sensitivity does not permit relaxing an existing threshold.
A previously uncalibrated input length needs separate calibration before held-out candidates.
Use the previous rule: two times serial calibration percentiles, taking the tighter per-metric Atomic/chunked result.
Freeze the resulting file before that cell's held-out serving.

The main request SLO uses TTFT and request-average TPOT.
The D-headroom gate also uses token ITL p99 and the fixed TPOT margin.
Failures and timeouts remain in offered-request denominators.
An overloaded finite cohort does not establish stable capacity.

## 11. Same-start target gates

Preserve the 80% actual P kernel-union screen and 85% strongest matched finite P-service screen.
Require positive P-ready backlog growth that excludes D-reservation-blocked requests.
Require throughput plateau or additional device evidence consistent with the identified limiting resource.
Inspect launch gaps, kernel identities, SM metrics, and bandwidth where available.
Do not call a bandwidth-limited device compute-bound.
Record such a result as a separate physical-resource bottleneck.

Evaluate D in the same start and overlapping interval.
Require request TPOT p95 at most 90% of its frozen limit and token ITL p99 within its frozen limit.
Require native legal KV, staging, slot, workspace, and transfer credits for the actual proposed probe.
A structural 64-token block screen does not certify another full long request.
The full target also requires a real correct D Prefill/recurrent probe with no Decode SLO regression.
Separate favorable starts cannot be combined into one target pass.

## 12. Actual D Prefill probe

Use a qualified native request and real checkpoint computations.
Start with 16 or 32 prompt tokens, then 64/128 only after the earlier increment passes.
Preserve complete R4/R32 and exact sampling/state semantics.
Use native legal phase transitions, source ownership, and request-buffer lifetime.
Background native KV handoff must not silently become a different snapshot protocol.
Provide matched probe-OFF and probe-ON runs at the same background inputs and arrivals.
Warm the same graph shapes in both runs.
Record the additional request independently from the original background denominator.
Record actual useful tokens, loop work, GPU event/kernel time, bytes, and HBM.
Account for extra work explicitly in the full-depth audit.

Admission must use observed D readiness and legal capacity, not future arrival or output knowledge.
Stop or reduce additional work when background Decode tails violate frozen limits.
Synthetic GEMM is only a supplementary capacity diagnostic.
If P does not qualify, an independent real D-reserve experiment may still establish D_HEADROOM_ONLY.
That separate experiment cannot establish the joint target.
The complete Dynamic Loop Split scheduler remains out of scope.

## 13. Resource safety and monitoring

Use physical H100 GPUs 0 through 7. Record their UUIDs and topology.
Default clean timing uses one isolated 2-GPU pair.
Stop only approved owned burners before measured work.
Retain other-user process ownership and do not signal unapproved jobs.
Restore approved burners during safe idle periods and after the final measurement.
Do not burn unused GPUs during a clean timing window unless interference is independently bounded.
Never hide shared-host or communication contention.

Use a persistent dispatcher and absolute-deadline watchdog.
Use immutable run namespaces and source archives.
Do not edit source while a fresh worker is validating its hashes.
Use the shared quiet-lane boundary lock for changes to active source.
Keep code failures, OOMs, empty traces, watchdogs, and rejected profiles in the ledger.

## 14. Decisions and stopping rules

Classify each model independently:

- TRUE_PREFILL_COMPUTE_HEAVY_WITH_D_SLACK
- P_COMPUTE_BOUND_ONLY
- D_HEADROOM_ONLY
- NONCOMPUTE_PREFILL_BOTTLENECK
- DECODE_OR_KV_BOUND
- INCONCLUSIVE

A strict positive needs same-start P and D proof, an actual probe, and independent reproduction.
Do not turn a missing measurement into a negative physical claim.
Do not turn a tested-grid negative into proof that recurrence makes the target impossible.
Stop investment in a candidate when useful throughput fails to improve or exactness fails.
Continue other legal candidates within the absolute budget.

## 15. Deliverables and publication

New source directory: `poc_flashvep/looped_pd_autonomous_headroom/`.
Evidence directory: `poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom/`.
Final report: `poc_flashvep/reports/looped_pd_autonomous_prefill_compute_decode_headroom_final_summary.md`.

Preserve the decision ledger, configurations, model/runtime hashes, commands, token IDs, arrival offsets, raw requests, GPU intervals, queue/credit events, and failed logs.
Report TTFT, TPOT, ITL, E2E, input/output throughput, SLO attainment/goodput, HBM, and actual transfer work.
Report additive host E2E phase percentages separately from overlapping device intervals.
Use MEASURED, TRACE_ANALYSIS, ANALYTICAL_ORACLE, DIAGNOSTIC, and NOT_MEASURED labels consistently.

Create five required figure groups:

1. Static PD versus real Native Mixed DP2: QPS against TTFT p95, TPOT p95, ITL p99, and total throughput.
2. Same-start P queues, actual kernel activity, Prefill completions, and identified noncompute waits.
3. Same-start D kernel activity, legal KV credits, active batches, and frozen Decode SLOs.
4. Real additional D Prefill work against background Decode tails, or an explicit unmeasured status.
5. Matched baseline/optimization root cause and useful-work/E2E effects.

Use PNG plus SVG/PDF, explicit run IDs, measurement windows, and a source manifest.
Do not plot invented idle time or fictional positive borrowing points.
Include three-start uncertainty only where three actual starts exist.

Commit and push only this campaign's files through a temporary index.
Use ordinary non-force pushes and verify the remote SHA.
Keep large raw evidence losslessly reconstructable without deleting local originals.
Reserve publication time inside the nine-hour campaign limit.
