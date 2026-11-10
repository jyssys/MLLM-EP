# Completion-aware Adaptive Loop-Split PD reimplementation

## 1. Objective and evidence contract

Reproduce useful P/D Prefill overlap, not only successful state migration.
Determine why the previous scheduler rarely accepted work and delayed accepted requests for many seconds.
Preserve the original BF16 weights and exact Attention.
Use Ouro fixed R4 and Huginn fixed R32.
Execute every recurrent operation exactly once.
Preserve the qualified numerical fixes, Native ACK, and source-buffer lifetime.

The publication parent is `0adb7491c866d6d66cf71ea30dca63f82c46d135`.
The new branch is `flashvep/completion-aware-adaptive-loop-pd`.
The evidence directory is `ouro_huginn_completion_aware_adaptive_loop_pd_reimplementation/`.
The source directory is `poc_flashvep/completion_aware_loop_pd/`.
Earlier code and evidence remain unchanged.

Measured kernel activity, CUDA Event enclosures, host observations, structural work, and modeled costs remain separate evidence classes.
An event enclosure is not exclusive kernel execution.
Output delivery does not identify an individual Decode kernel.
An unobserved interval is not measured idle.

## 2. Clock and resources

The operational start is 2026-11-10 14:51:19 KST.
The operational bound is eight hours, including publication and resource restoration.
The current request specifies no new numerical time limit.
GPU work stops at least 45 minutes before this operational bound.
The bound does not reopen an earlier campaign clock.

Use verified owned H100 GPU pairs, initially Ouro 0/1 and Huginn 4/5.
Clean timing uses one scientific pair at a time.
Stop verified approved Burners on all eight GPUs during clean timing.
Do not stop unrelated jobs.
Restore approved Burners during extended GPU-free intervals and after the campaign.
Check all eight worker sessions, GPU ownership, and generation progress after restoration.
Recheck missing GPU workers without duplicating healthy workers.

## 3. Frozen models and workloads

| Model | Input / output | Depth | Heavy QPS | Low QPS | Medium QPS | P/D slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ouro-1.4B | 4096 / 32 | R4 | 6.356 | 1.23460875 | 3.4569045 | 2/8 |
| Huginn-0125 | 2048 / 16 | R32 | 1.971 | 0.37903355455 | 1.06129395274 | 2/8 |

Retain the existing model revisions and controlled ShareGPT/LongBench tokenized prompt pool.
Use 128 distinct prompt IDs, fixed output caps, ignored EOS, and disabled prefix caching.
Use paired open-loop Poisson arrivals.
Use calibration seed 2026110910 separately from confirmation seeds 2026110911, 2026110912, and 2026110913.
Keep actual host arrival times separate from planned arrival offsets.
Finite-cohort completion rates are not sustainable capacity.

| Model | TTFT limit, ms | Request TPOT limit, ms | Additional-work TPOT p95, ms | ITL p99 limit, ms |
| --- | ---: | ---: | ---: | ---: |
| Ouro | 1112.9925646819174 | 53.70702898374411 | 48.3363260853697 | 65.08287642151119 |
| Huginn | 1963.9920117333531 | 186.5715059141318 | 167.91435532271862 | 192.8212538175284 |

Retain Native-relative Decode-tail comparisons.
The previous 5% non-inferiority target remains a reported design target.
Do not relax the absolute SLO to increase acceptance.

The recovery confirmation uses arrival seeds 2026110931, 2026110932, and 2026110933.
These seeds differ from recovery calibration and N32 selection seed 2026110920.
The preserved prompt IDs remain unchanged.
The recovery policy freezes before these three confirmation arrivals.
Native, Loop, and Token share each model's recovery KV pool and graph policy.
Huginn uses 24832 D blocks in all three recovery arms.
The earlier primary comparison retains its original 23552-block pool and separate namespace.
The two comparisons must not become one combined treatment estimate.

## 4. Root-cause feedback loop

Replay the old first-seed continuation observations before changing the implementation.
The initial regression screen requires no inter-quantum submission gap above one second.
This screen reproduces 23.207262 s for Ouro and 85.872055 s for Huginn.
It is a starvation diagnostic, not an additional Decode SLO.

Rank and test four hypotheses independently:

1. A failed initial arming decision prevents later boundary evaluation.
2. A long-lived continuation occupies the only admission slot.
3. Serial cost predicts execution but omits waiting for future safe opportunities.
4. Persistent safety suspension prevents progress without a completion reservation.

Produce one gate record for each measured request.
Track initial arming, repeated opportunities, offers, preparation, commit, import, quanta, output, ACK, and final release.
Separate actual observed rejection counts from counterfactual constraint ablations.
Do not infer a missing source event as an observed GPU failure.

## 5. Reference review

Inspect local `ref_loopserve` first.
Inspect its scheduler, recurrent execution, graph controller, KV ownership, and lifecycle code.
Do not import its Early Exit or incompatible KV semantics.
Prefer existing qualified kernels and exact connectors over new CUDA kernels.

Read FluidPD at https://arxiv.org/abs/2610.06917.
Check FluidToken, PPI/DPI, completion deadlines, partial Prefill handoff, and KV admission.
Check official code availability without inventing repositories or APIs.
Label any local implementation as a reproduction or adaptation.

Inspect SGLang's official `python/sglang/srt/disaggregation/decode.py`.
Review separate preallocation, transfer-pending ownership, admission, asynchronous publication, and Decode activation.
Reuse design principles only where the qualified runtime preserves equivalent state semantics.
Record retrieved source revisions and hashes.

## 6. Measurement stage

Run paired marker-free Native controls for both models.
Measure an instrumented control with the same configuration and arrivals.
Reuse existing CUDA Event and CUPTI collectors where possible.
Collect P Prefill, D Decode, and D continuation activity separately.
Keep request, ready, admission, commit, transfer, ACK, and first-output timestamps.

CUDA Events can measure stream enclosures without timed-path synchronization.
CUPTI or Nsight can identify individual kernels when progress remains valid.
Measure instrumentation wall-time and ITL-tail effects against the marker-free control.
Use the existing 5% interference contract for qualified performance traces.
Retain invalid traces for diagnostics only.
If kernel evidence remains invalid, report physical idle and its Oracle as `NOT_MEASURED`.
Continue safe controlled interventions rather than claim absent physical headroom.

## 7. Scheduler redesign

Separate source opportunity, completion admission, and execution permission.
Reevaluate legal boundaries after transient rejection where execution has not passed that boundary.
Do not claim that a completed atomic graph can expose an earlier hidden state retrospectively.
Measure the cost of checkpoint-capable execution separately from unmodified Native execution.
Keep the feature disabled path Native.

Completion admission must consider all remaining quanta, handoff, current continuation work, Decode reserve, and scheduling uncertainty.
Compare predicted D completion with continued P execution and observed P queue pressure.
Do not use future arrivals in the online policy.
Log prediction inputs, uncertainty, completion deadlines, and realized error.

Reserve execution opportunities before commit rather than assume unlimited future borrowing.
Use earliest completion deadlines for accepted continuation work.
Protect ready Decode deadlines before each non-preemptive quantum.
Do not preempt a running kernel.
Reject an offer before commit if the conservative completion schedule is infeasible.
Expire stale preparations without changing committed ownership.

Track progress age after commit.
Do not discard a request at an arbitrary timeout.
A predicted completion miss stops further borrowing and remains an experimental failure.
Complete accepted work exactly, or use only a separately qualified exact recovery path.
Do not invent D-to-P recovery when no safe state-return protocol exists.

## 8. Protocol and correctness gates

The protocol is offer, prepare, commit, state transfer, import, execution, Decode transition, ACK, and release.
Every transfer uses an epoch and explicit source/destination ownership.
The source cannot execute the committed suffix again.
The destination cannot execute uncommitted state.
ACK must precede source reuse.

Run CPU tests before GPU qualification.
Cover transient rejection, stale epochs, cancellation before commit, queue limits, and progress deadlines.
Run N8 before N32.
Compare exact output tokens, complete valid KV, hidden state, recurrent work, and wire bytes with Native.
Require actual online mixed cuts where the scheduler selects different cuts.
Label forced cuts as diagnostics, not adaptive success.
Do not expand a correctness-failing path to N128.
Stop an OOM or unsafe-state path and retain its evidence.

## 9. Diagnostic Oracles

Physical-slack Oracle uses measured kernel gaps, non-preemptive quantum costs, Decode deadlines, transfer, and KV credit.
Future-aware results are offline upper bounds.
Missing valid kernel traces block this Oracle, not every GPU experiment.

Forced-offload diagnostics cover Ouro cuts R1/R2/R3 and Huginn R24/R28/R30/R31 where exactness is qualified.
Measure actual D work, completion delay, transfer, Decode impact, and request latency.
An SLO-violating forced run is not successful serving.

Completion-feasible Oracle includes waiting. Check predictions against independent real static cuts.
Report prediction error and uncertainty.
An unvalidated model cannot establish attainable online speedup.
Compare physical, completion-feasible, and actual scheduler capacity without treating them as interchangeable.

The 500-Loop target is an exploration target, not an assumption.
Report each model's structural denominator, physical estimate, SLO-feasible estimate, and actual completed offloads.
Report unknown bounds as unknown rather than zero.

## 10. Serving evaluation

Compare Native, old adaptive Loop, new adaptive Loop, and strong adaptive Token.
Add a hybrid diagnostic only after its exact dependency frontier passes qualification.
Do not restrict Token to inefficient small tiles.
Use equal graph budgets, warm-up policy, KV settings, prompts, and paired arrivals.
Separate Loop rows from FLOPs and causal Attention work.

Use N128 and at least three independent arrival seeds per qualified primary arm.
Retain failed and incomplete runs.
Report per-start values and paired start-level confidence intervals.
Do not treat tokens as independent starts.

Report TTFT p50/p95/p99, E2E p95, TPOT p95, ITL p99/max, and threshold exceedances.
Report completed QPS, input/output Token/s, and joint SLO-goodput.
Report offloaded requests, Loops, recurrent rows, completion p50/p95/p99/max, and scheduling gaps.
Report P queue, valid kernel busy/idle, KV credit, memory, and actual payload.
Report host enclosures separately.
Service TTFT requires a qualified actual execution-start observation.
Do not substitute a coordinator start timestamp.

## 11. Artifacts and verdict

Produce root-cause, reference-review, implementation, physical-headroom, and serving reports.
Produce interactive Native/old/new timelines with common time units and explicit evidence types.
Include all configuration hashes, source hashes, GPU mappings, seeds, failures, and exclusions.
Preserve the user's staged index and external runtime diff.
Use a temporary Git index for the scoped publication if the checkout remains dirty.
Push the new branch and check the remote SHA.

Start the final report with whether the intended TO-BE actually occurs.
Use a model-specific verdict:

- `GO_LOOP_SPECIFIC_BENEFIT`
- `GO_GENERIC_OFFLOADING_ONLY`
- `NO_GO_PHYSICAL_HEADROOM`
- `NO_GO_NET_BENEFIT`
- `INCONCLUSIVE_MEASUREMENT`

## 12. Predeclared diagnostic recovery after the primary comparison

The primary confirmation configurations remain immutable.
The first Ouro Token start records a maximum ITL of 866.017 ms.
Stop later starts of that arm after the current running job finishes.
Do not discard that result or treat the missing start as zero performance.

The primary Token path also creates cold graphs during measurement.
The graph audit records ten measured P captures in its first start.
That fact is a comparison confound. It does not uniquely explain the maximum ITL.

A secondary diagnostic may disable measured graph capture for every arm.
Cached graphs remain active. Cold shapes use the qualified eager path.
All arms retain the same 64-entry cache and startup policy.
Report eager fallback counts and startup costs.

Huginn's primary D pool has 23552 blocks. Each primary request reserves 4128 blocks.
That pool holds five complete requests, despite eight logical slots.
A secondary diagnostic may use 24832 blocks for every compared Huginn arm.
That pool holds six complete requests. It adds 1280 blocks, or 3460300800 bytes.
Require at least 3 GiB of free HBM during startup statistics checks.
Stop the larger-pool path after an HBM guard failure or OOM.
Native controls must use the same larger pool.
Do not attribute a larger-pool benefit to Loop scheduling.
The separate Native idle Oracle retains the original 23552-block configuration.

The secondary Token baseline may select one prefix from the current D advertisement.
It performs destination revalidation at that boundary.
After rejection, it completes the whole remaining suffix on P without repeated small checkpoints.
This baseline differs from repeated-boundary Token negotiation. Report that difference explicitly.

The secondary completion model may use observed full Decode-cycle enclosures.
The cycle model includes host gaps that individual CUDA Event durations omit.
It remains an empirical prediction, not a reserved future execution opportunity.
The recovery model also prices the future Decode work of already accepted continuations.
Those requests are known at admission time. They are not hypothetical future arrivals.
The recovery diagnostic uses continuation deadlines of 1000 ms for Ouro and 1200 ms for Huginn.
These deadlines do not change any frozen TTFT, TPOT, or ITL SLO.
They bound accepted continuation residence. They do not guarantee a request-level TTFT pass.
After a tail suspension, admission may resume in an empty Decode epoch.
Require zero active Decode requests and zero pending committed continuations before this recovery.
A completion-reservation failure does not permit that recovery.
The source may skip an RPC when a fresh D advertisement already rejects new work.
It still reevaluates the next legal Loop boundary.
This local skip does not create a destination lease or change ownership.
Report local skips separately from destination rejection.

An absolute-SLO-envelope diagnostic may use the unchanged frozen ITL limit for admission.
It must retain the frozen additional-work TPOT threshold.
The Native-relative 5% target remains a separate required comparison.
An absolute SLO pass with a failed Native-relative target is not a non-inferiority success.
This diagnostic must not replace the primary results.

Qualify every new protocol behavior with N8 and N32 exact-state tests before clean serving.
Use arrival seed 2026110920 for selection.
Use new arrival seeds 2026110931/32/33 only if selection supports further evaluation.
Do not describe reused primary arrival seeds as unseen validation after a code change.

Missing measurements are not evidence of physical impossibility.
Equivalent logical-layer mechanisms do not establish Loop-specific novelty.
Preserve negative results without weakening correctness or Decode SLO.
